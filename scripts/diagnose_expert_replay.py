"""
Diagnostic: replay each episode's expert `reference_replay` action sequence
through the sim WITHOUT any policy in the loop, and measure what fraction of
demonstrations actually score as successful under the current task config.

This establishes the *ceiling* of any learned policy: if the expert demos
themselves don't succeed under our SUCCESS / SUCCESS_DISTANCE /
DISTANCE_TO / FORWARD_STEP_SIZE / TURN_ANGLE / ALLOW_SLIDING / agent
radius config, nothing trained on them ever will.
"""

import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from habitat.core.environments import get_env_class
from habitat.utils.env_utils import construct_envs

import pirlnav  # noqa: F401
from pirlnav.config import get_config


ACTION_NAME_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_UP": 4,
    "LOOK_DOWN": 5,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", default="configs/experiments/il_objectnav_mp3d.yaml",
    )
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--success-distance", type=float, default=None,
                   help="override SUCCESS.SUCCESS_DISTANCE (in meters)")
    p.add_argument("--allow-sliding", dest="allow_sliding",
                   action="store_true")
    p.add_argument("--no-allow-sliding", dest="allow_sliding",
                   action="store_false")
    p.set_defaults(allow_sliding=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.config, opts=None)
    cfg.defrost()
    cfg.RUN_TYPE = "eval"
    cfg.NUM_ENVIRONMENTS = 1
    cfg.TEST_EPISODE_COUNT = args.num_episodes
    cfg.TASK_CONFIG.DATASET.SPLIT = args.split
    if args.success_distance is not None:
        cfg.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE = args.success_distance
        cfg.TASK_CONFIG.TASK.SUCCESS_DISTANCE = args.success_distance
    if args.allow_sliding is not None:
        cfg.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = (
            args.allow_sliding
        )
    cfg.freeze()

    envs = construct_envs(cfg, get_env_class(cfg.ENV_NAME))

    total = 0
    succ = 0
    spl_sum = 0.0
    dist_sum = 0.0
    action_stop_from_sensor = 0
    action_stop_from_replay = 0
    per_cat = Counter()
    per_cat_succ = Counter()
    ep_fail_samples = []

    for ep_idx in range(args.num_episodes):
        envs.reset()
        cur = envs.current_episodes()[0]
        replay = list(cur.reference_replay)
        last_info = None
        steps_taken = 0
        # reference_replay[0] is typically the starting pose (action not
        # meaningful); actual actions start at index 1.  We match the
        # training DemonstrationSensor semantics as a fallback.
        start_idx = 1 if len(replay) > 0 and hasattr(replay[0], "action") else 0
        for r in replay[start_idx:]:
            action_name = r.action
            if action_name not in ACTION_NAME_TO_ID:
                # some recorders use uppercase or enum values
                action_name = str(action_name).upper().split(".")[-1]
            a = ACTION_NAME_TO_ID[action_name]
            outs = envs.step([a])
            _obs, _rew, dones, infos = [list(x) for x in zip(*outs)]
            last_info = infos[0]
            steps_taken += 1
            if a == 0:
                action_stop_from_replay += 1
            if dones[0]:
                break

        s = float((last_info or {}).get("success", 0.0))
        spl = float((last_info or {}).get("spl", 0.0))
        d = float((last_info or {}).get("distance_to_goal", float("nan")))
        total += 1
        succ += int(s > 0.5)
        spl_sum += spl
        dist_sum += d if d == d else 0.0
        per_cat[cur.object_category] += 1
        if s > 0.5:
            per_cat_succ[cur.object_category] += 1
        else:
            ep_fail_samples.append(
                (cur.episode_id, cur.object_category, steps_taken,
                 len(replay), d, spl)
            )

        print(
            f"  ep={cur.episode_id:>4} cat={cur.object_category:<10} "
            f"steps={steps_taken:>4} replay_len={len(replay):>4} "
            f"d2goal={d:.3f}m spl={spl:.3f} succ={int(s>0.5)}"
        )

    print()
    print("=" * 70)
    print(
        f"Expert-replay sim evaluation on {args.split} split "
        f"({total} episodes):"
    )
    print(f"  Success rate      : {100 * succ / total:.2f}%  ({succ}/{total})")
    print(f"  Mean SPL          : {spl_sum / total:.4f}")
    print(f"  Mean dist-to-goal : {dist_sum / total:.3f} m")
    print(f"  SUCCESS_DISTANCE  : {cfg.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE}")
    print(f"  DISTANCE_TO       : {cfg.TASK_CONFIG.TASK.DISTANCE_TO_GOAL.DISTANCE_TO}")
    print()
    print("Per-category success:")
    for cat in sorted(per_cat):
        n = per_cat[cat]
        k = per_cat_succ[cat]
        print(f"  {cat:<10} {k}/{n} ({100 * k / n:.1f}%)")
    if ep_fail_samples:
        print("\nFirst few failures (expert replay didn't succeed):")
        for e, c, st, rl, d, sp in ep_fail_samples[:10]:
            print(
                f"  ep={e:>4} cat={c:<10} steps={st:>4} replay_len={rl:>4} "
                f"d2goal_end={d:.3f}m spl={sp:.3f}"
            )
    envs.close()


if __name__ == "__main__":
    main()
