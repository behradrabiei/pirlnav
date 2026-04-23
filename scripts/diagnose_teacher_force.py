"""
Diagnostic: teacher-force the trained IL policy on its training (or val)
episodes and measure how often its argmax matches the expert's action.

Rationale:
    Training loss was driven to ~2e-5 (essentially 0) under inflection-weighted
    cross-entropy.  If the pipeline is wired correctly, argmax(policy(obs))
    should equal expert_action at essentially every step on the TRAIN split
    when we keep the policy on the expert's own trajectory.  Anything
    meaningfully less than ~100% points at a systemic problem (observation
    encoding mismatch, category embedding mismatch, test-time RGB
    augmentation, etc.) rather than covariate shift -- covariate shift only
    bites once the policy deviates from the expert state distribution.
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from habitat.utils.env_utils import construct_envs
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.utils.common import batch_obs

import pirlnav  # noqa: F401 - registers policies / sensors
from pirlnav.config import get_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/experiments/il_objectnav_mp3d.yaml",
    )
    p.add_argument(
        "--ckpt",
        default="data/new_checkpoints/objectnav_il/overfit_v1/ckpt.9.pth",
    )
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--no-augment", action="store_true",
                   help="disable test-time RGB augmentations")
    p.add_argument("--self-drive", action="store_true",
                   help="step env with policy action instead of expert action; "
                        "additionally report first divergence step vs expert")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.config, opts=None)
    cfg.defrost()
    cfg.RUN_TYPE = "eval"
    cfg.NUM_ENVIRONMENTS = 1
    cfg.TEST_EPISODE_COUNT = args.num_episodes
    cfg.TASK_CONFIG.DATASET.SPLIT = args.split
    # Expose the expert action as an observation so we can diff it against
    # the policy's argmax.
    if "DEMONSTRATION_SENSOR" not in cfg.TASK_CONFIG.TASK.SENSORS:
        cfg.TASK_CONFIG.TASK.SENSORS.append("DEMONSTRATION_SENSOR")
    if args.no_augment:
        cfg.POLICY.RGB_ENCODER.use_augmentations_test_time = False
    cfg.freeze()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    envs = construct_envs(cfg, get_env_class(cfg.ENV_NAME))
    obs_space = envs.observation_spaces[0]
    action_space = envs.action_spaces[0]

    policy_cls = baseline_registry.get_policy(cfg.IL.POLICY.name)
    actor_critic = policy_cls.from_config(
        cfg, observation_space=obs_space, action_space=action_space,
    ).to(device)
    actor_critic.eval()

    print(f"[info] loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {
        k.replace("actor_critic.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("actor_critic.")
    }
    missing, unexpected = actor_critic.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} "
              f"(first 3: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} "
              f"(first 3: {unexpected[:3]})")

    n_layers = actor_critic.net.num_recurrent_layers
    hid_sz = cfg.POLICY.STATE_ENCODER.hidden_size

    match_count_total = 0
    step_count_total = 0
    expert_hist = Counter()
    policy_hist = Counter()
    confusion = defaultdict(Counter)

    for ep_idx in range(args.num_episodes):
        observations = envs.reset()
        cur_ep = envs.current_episodes()[0]
        ref_len = (
            len(cur_ep.reference_replay)
            if getattr(cur_ep, "reference_replay", None) is not None
            else -1
        )
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, [])

        hidden = torch.zeros(1, n_layers, hid_sz, device=device)
        prev_actions = torch.zeros(1, 1, dtype=torch.long, device=device)
        masks = torch.zeros(1, 1, dtype=torch.bool, device=device)

        step = 0
        matches = 0
        first_divergence = -1
        last_info = None

        while True:
            # Expert's action at the *current* step, as the task wants it.
            # If we are self-driving and have drifted off the expert's
            # trajectory, the DemonstrationSensor may well return a bogus
            # value (it indexes reference_replay by an internal timestep
            # that tracks the true wall-clock sim step), but we only use
            # it for divergence bookkeeping, not for stepping.
            expert_action = int(batch["next_actions"].flatten()[0].item())
            with torch.no_grad():
                acts, hidden = actor_critic.act(
                    batch, hidden, prev_actions, masks, deterministic=True,
                )
            policy_action = int(acts.flatten()[0].item())
            expert_hist[expert_action] += 1
            policy_hist[policy_action] += 1
            confusion[expert_action][policy_action] += 1
            if policy_action == expert_action:
                matches += 1
            elif first_divergence < 0:
                first_divergence = step
            step += 1

            step_action = policy_action if args.self_drive else expert_action
            outputs = envs.step([step_action])
            obs_list, _rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            last_info = infos[0]
            if dones[0]:
                break
            batch = batch_obs(obs_list, device=device)
            batch = apply_obs_transforms_batch(batch, [])
            prev_actions = torch.tensor(
                [[step_action]], dtype=torch.long, device=device,
            )
            masks = torch.ones(1, 1, dtype=torch.bool, device=device)

        acc = matches / max(step, 1)
        match_count_total += matches
        step_count_total += step
        succ = (
            (last_info or {}).get("success", 0.0) if last_info else 0.0
        )
        spl = (last_info or {}).get("spl", 0.0) if last_info else 0.0
        print(
            f"  ep={cur_ep.episode_id:>4}  cat={cur_ep.object_category:<10} "
            f"steps={step:>4}  matches={matches:>4}  "
            f"acc={acc * 100:5.1f}%  ref_len={ref_len}  "
            f"first_div={first_divergence:>4}  "
            f"succ={succ:.0f}  spl={spl:.2f}"
        )

    print()
    print("=" * 70)
    print(
        f"Teacher-forced argmax-vs-expert on {args.split} split "
        f"({args.num_episodes} episodes):"
    )
    print(f"  Total steps     : {step_count_total}")
    print(f"  Total matches   : {match_count_total}")
    print(
        f"  Agreement       : "
        f"{100 * match_count_total / max(step_count_total, 1):.2f}%"
    )
    print(
        f"  Aug at eval     : "
        f"{cfg.POLICY.RGB_ENCODER.use_augmentations_test_time}"
    )
    print()
    print("Expert action distribution (TF states):")
    total = sum(expert_hist.values())
    for a in sorted(expert_hist):
        print(
            f"  action {a}: {expert_hist[a]:5d} "
            f"({100 * expert_hist[a] / total:5.1f}%)"
        )
    print("Policy argmax distribution (same states):")
    total = sum(policy_hist.values())
    for a in sorted(policy_hist):
        print(
            f"  action {a}: {policy_hist[a]:5d} "
            f"({100 * policy_hist[a] / total:5.1f}%)"
        )
    print()
    print("Confusion (rows=expert, cols=policy argmax):")
    all_a = sorted(set(expert_hist) | set(policy_hist))
    print("   expert\\pol   " + "  ".join(f"{a:>5d}" for a in all_a))
    for ea in all_a:
        row = "  ".join(f"{confusion[ea][pa]:>5d}" for pa in all_a)
        print(f"        {ea:>5d}   {row}")

    envs.close()


if __name__ == "__main__":
    main()
