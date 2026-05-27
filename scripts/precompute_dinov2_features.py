"""Precompute frozen DINOv2 CLS features for pirlnav IL episodes.

Because pirlnav's IL training is pure teacher-forcing (the env is stepped with
the expert's ``next_actions``), the RGB observed at step ``t`` of episode
``e`` is a deterministic function of ``(episode_id, t)``. We can therefore
run the DINOv2 forward pass once per step per episode, offline, and cache the
resulting 768-dim CLS token for the online trainer to reuse.

Output layout::

    {cache_root}/{scene}/{episode_id}.pt
        |-- dino_cls : torch.Tensor (T+1, 768) float16
        |-- success  : float  (Habitat Success measure after replay)
        |-- spl      : float  (SPL at end of replay)

where ``T = len(episode.reference_replay) - 1`` (the first entry of
``reference_replay`` is a no-op init; we still record the initial-reset CLS
so the full length matches the ``DemonstrationSensor.timestep`` counter).

RGB preprocessing matches the ResNet path: resize shorter edge, then
center-crop to patch-aligned ``(H, W)`` (default 252x252).

The script is resumable: episodes whose ``.pt`` file already exists are
skipped. Typical runtime on a single RTX 5090 is ~12-15 min for the 302+53
episodes of the one-scene MP3D subset.

Usage::

    python scripts/precompute_dinov2_features.py \
        --config configs/experiments/il_objectnav_mp3d_dinov2.yaml \
        --split train \
        --scene 17DRP5sb8fy \
        --cache-root data/dinov2_cache

    # Recorded poses instead of stepping expert actions (separate cache dir):
    python scripts/precompute_dinov2_features.py \
        --replay-mode poses \
        --cache-root data/dinov2_cache
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import habitat  # noqa: E402

import pirlnav  # noqa: F401,E402
from pirlnav.config import get_config  # noqa: E402
from pirlnav.policy.dinov2_encoder import DINOv2VisualEncoder  # noqa: E402
from pirlnav.policy.transforms import DINOv2Transform  # noqa: E402


ACTION_NAME_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_UP": 4,
    "LOOK_DOWN": 5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/experiments/il_objectnav_mp3d_dinov2.yaml",
        help="Base experiment config (picks up the training task yaml so "
        "ALLOW_SLIDING / SUCCESS_DISTANCE / FORWARD_STEP_SIZE match "
        "training).",
    )
    p.add_argument(
        "--split", default="train", choices=["train", "val", "both"],
        help="Which split to precompute. Defaults to 'train' because the "
        "cache is only valid under teacher-forced IL training; at eval the "
        "policy takes its own actions so cached features are not reusable.",
    )
    p.add_argument("--scene", default="17DRP5sb8fy")
    p.add_argument("--cache-root", default="data/dinov2_cache")
    p.add_argument(
        "--model-name", default="facebook/dinov2-base",
        help="HuggingFace model id for the frozen DINOv2 encoder.",
    )
    p.add_argument("--resize-h", type=int, default=252)
    p.add_argument("--resize-w", type=int, default=252)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--max-episodes", type=int, default=0,
        help="Debug: cap episodes per split (0 = all).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-encode episodes even if a cache file already exists.",
    )
    p.add_argument(
        "--replay-mode",
        default="actions",
        choices=["actions", "poses"],
        help="Step expert actions (default) or teleport to recorded poses. "
        "Pose mode writes under {cache_root}_poses/ to avoid clobbering "
        "action-based caches.",
    )
    return p.parse_args()


DATASET_ROOT = (
    "data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat"
)


def cache_scene_dir(args: argparse.Namespace) -> Path:
    root = Path(args.cache_root)
    if args.replay_mode == "poses":
        root = root.parent / f"{root.name}_poses"
    return root / args.scene


def load_raw_episodes(split: str, scene: str) -> Dict[str, dict]:
    path = Path(DATASET_ROOT) / split / "content" / f"{scene}.json.gz"
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return {str(ep["episode_id"]): ep for ep in data["episodes"]}


def action_name_to_id(action_name: str) -> int:
    if action_name not in ACTION_NAME_TO_ID:
        action_name = str(action_name).upper().split(".")[-1]
    return ACTION_NAME_TO_ID[action_name]


def set_agent_from_dict(env: habitat.Env, agent_state: dict) -> bool:
    import quaternion as qlib  # noqa: F401

    rot = agent_state["rotation"]
    q = qlib.quaternion(rot[3], rot[0], rot[1], rot[2])
    return bool(env.sim.set_agent_state(agent_state["position"], q))


def collect_rgb_actions(env: habitat.Env, ep) -> Tuple[List[np.ndarray], dict]:
    """Replay expert actions and return RGB frames plus final Habitat metrics."""
    env.episode_iterator = iter([ep])
    obs = env.reset()
    cur = env.current_episode
    if str(cur.episode_id) != str(ep.episode_id):
        raise RuntimeError(
            f"wanted episode {ep.episode_id}, got {cur.episode_id}"
        )
    replay = list(cur.reference_replay)
    if not replay:
        return [], env.get_metrics()

    rgbs = [np.asarray(obs["rgb"])]
    metrics = env.get_metrics()
    start_idx = 1 if hasattr(replay[0], "action") else 0
    for r in replay[start_idx:]:
        step_obs = env.step(action_name_to_id(r.action))
        rgbs.append(np.asarray(step_obs["rgb"]))
        metrics = env.get_metrics()
        if env.episode_over:
            break
    return rgbs, metrics


def collect_rgb_poses(
    env: habitat.Env, ep, raw_episode: dict
) -> Tuple[List[np.ndarray], dict]:
    """Teleport to recorded poses, issue STOP at the final pose, return RGBs."""
    env.episode_iterator = iter([ep])
    env.reset()

    rgbs: List[np.ndarray] = []
    for step in raw_episode.get("reference_replay", []):
        state = step.get("agent_state")
        if state is None:
            continue
        obs = env.sim.get_observations_at(
            state["position"],
            state["rotation"],
            keep_agent_at_new_pose=True,
        )
        if obs is None:
            print(
                f"  [warn] ep={ep.episode_id}: agent placement failed, skip step"
            )
            continue
        rgbs.append(np.asarray(obs["rgb"]))

    replay = raw_episode.get("reference_replay", [])
    if replay:
        last = replay[-1].get("agent_state")
        if last is not None:
            set_agent_from_dict(env, last)
            env.step(0)

    return rgbs, env.get_metrics()


def metrics_from_payload(payload) -> Tuple[float, float]:
    if isinstance(payload, dict):
        success = float(payload.get("success", 0.0))
        spl = float(payload.get("spl", 0.0))
        return success, spl
    return 0.0, 0.0


def load_cached_metrics(path: Path) -> Optional[Tuple[float, float]]:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "success" not in payload:
        return None
    return metrics_from_payload(payload)


def build_encoder(args: argparse.Namespace) -> DINOv2VisualEncoder:
    encoder = DINOv2VisualEncoder(
        model_name=args.model_name,
        resize_hw=(args.resize_h, args.resize_w),
        output_dim=768,
    )
    encoder = encoder.to(args.device)
    encoder.eval()
    return encoder


@torch.no_grad()
def encode_rgb(
    encoder: DINOv2VisualEncoder,
    transform: DINOv2Transform,
    rgb_uint8: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Run the online DINOv2 preprocessing transform on a single uint8 RGB
    frame, then forward through the encoder and return the CLS token as
    fp16 on CPU.

    Using ``DINOv2Transform`` (rather than a hand-rolled resize) keeps this
    numerically identical to what ``ObjectNavILMAENet.forward`` does during
    the uncached DINOv2 training run.
    """
    rgb = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).to(device)
    rgb = rgb.unsqueeze(0)  # (H, W, 3) -> (1, H, W, 3)
    rgb = transform(rgb)  # -> (1, 3, H_resize, W_resize) in [0, 1]
    cls = encoder(rgb)
    return cls.squeeze(0).to(torch.float16).cpu()


def build_task_config_for_split(base_config_path: str, split: str):
    """Build a pirlnav task config pinned to a single split for replay,
    with the episode iterator disabled (we drive episode selection
    manually so every episode is visited exactly once)."""
    cfg = get_config(base_config_path, opts=None)
    task_cfg = cfg.TASK_CONFIG.clone()
    task_cfg.defrost()
    task_cfg.DATASET.SPLIT = split
    # Disable iterator cycling/shuffling so env.episodes stays in file order
    # when we drive resets manually.
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = 10**12
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
    task_cfg.freeze()
    return task_cfg


def process_split(
    args: argparse.Namespace,
    encoder: DINOv2VisualEncoder,
    split: str,
) -> None:
    print(f"\n=== split={split} replay_mode={args.replay_mode} ===")
    task_cfg = build_task_config_for_split(args.config, split)
    env = habitat.Env(config=task_cfg)

    raw_by_id: Optional[Dict[str, dict]] = None
    if args.replay_mode == "poses":
        raw_by_id = load_raw_episodes(split, args.scene)

    scene_dir = cache_scene_dir(args)
    scene_dir.mkdir(parents=True, exist_ok=True)
    print(f"  cache dir: {scene_dir}")

    episodes = list(env.episodes)
    num_eps = len(episodes)
    if args.max_episodes > 0:
        num_eps = min(num_eps, args.max_episodes)
        episodes = episodes[:num_eps]
    print(f"  total episodes in split: {num_eps}")

    t0 = time.time()
    done_ct = 0
    skip_encode_ct = 0
    total_frames = 0
    n_success = 0
    n_evaluated = 0
    spl_sum = 0.0
    resize_hw = (args.resize_h, args.resize_w)
    transform = DINOv2Transform(augmentations_name="dinov2", size_hw=resize_hw)

    try:
        for ep_idx, ep in enumerate(episodes):
            out_path = scene_dir / f"{ep.episode_id}.pt"
            skip_encode = out_path.exists() and not args.overwrite

            if skip_encode:
                cached_metrics = load_cached_metrics(out_path)
                if cached_metrics is not None:
                    success_val, spl_val = cached_metrics
                    n_evaluated += 1
                    n_success += int(success_val > 0.5)
                    spl_sum += spl_val
                    skip_encode_ct += 1
                    done_ct += 1
                    continue

            if args.replay_mode == "actions":
                rgbs, metrics = collect_rgb_actions(env, ep)
            else:
                raw = (raw_by_id or {}).get(str(ep.episode_id))
                if raw is None:
                    print(f"  [warn] ep={ep.episode_id} missing in raw json, skip")
                    continue
                rgbs, metrics = collect_rgb_poses(env, ep, raw)

            if not rgbs:
                print(f"  [warn] ep={ep.episode_id} produced no frames, skip")
                continue

            success_val = float(metrics.get("success", 0.0))
            spl_val = float(metrics.get("spl", 0.0))
            n_evaluated += 1
            n_success += int(success_val > 0.5)
            spl_sum += spl_val

            if skip_encode:
                payload = torch.load(out_path, map_location="cpu", weights_only=False)
                if not isinstance(payload, dict) or "dino_cls" not in payload:
                    print(
                        f"  [warn] ep={ep.episode_id}: cache missing dino_cls, "
                        "re-encoding"
                    )
                    skip_encode = False
                else:
                    payload["success"] = success_val
                    payload["spl"] = spl_val
                    torch.save(payload, out_path)
                    skip_encode_ct += 1

            if not skip_encode:
                feats = [
                    encode_rgb(encoder, transform, rgb, args.device)
                    for rgb in rgbs
                ]
                feat_tensor = torch.stack(feats, dim=0).contiguous()
                total_frames += feat_tensor.shape[0]
                torch.save(
                    {
                        "dino_cls": feat_tensor,
                        "success": success_val,
                        "spl": spl_val,
                    },
                    out_path,
                )

            done_ct += 1

            if done_ct % 25 == 0 or done_ct == num_eps:
                elapsed = time.time() - t0
                fps = total_frames / max(elapsed, 1e-6)
                sr = (
                    100.0 * n_success / n_evaluated if n_evaluated > 0 else 0.0
                )
                print(
                    f"  [{split}] {done_ct}/{num_eps} eps "
                    f"(encode-skipped {skip_encode_ct}, frames {total_frames}, "
                    f"success {n_success}/{n_evaluated}={sr:.1f}%, "
                    f"{fps:.1f} fps, {elapsed:.1f}s elapsed)"
                )
    finally:
        env.close()

    elapsed = time.time() - t0
    print(
        f"  done: {done_ct}/{num_eps} eps "
        f"(encode-skipped {skip_encode_ct}), "
        f"{total_frames} frames encoded, {elapsed:.1f}s total"
    )
    if n_evaluated > 0:
        print(
            f"  replay success ({args.replay_mode}): "
            f"{100.0 * n_success / n_evaluated:.2f}% "
            f"({n_success}/{n_evaluated})  mean_spl={spl_sum / n_evaluated:.4f}"
        )
    else:
        print("  replay success: no episodes evaluated")


def main() -> None:
    args = parse_args()
    print(
        f"replay_mode={args.replay_mode} cache_root={args.cache_root} "
        f"scene={args.scene}"
    )
    print(f"Loading encoder {args.model_name!r} on {args.device} ...")
    encoder = build_encoder(args)
    n_params = sum(p.numel() for p in encoder.backbone.parameters())
    print(f"  encoder backbone params: {n_params / 1e6:.1f} M (frozen)")

    Path(args.cache_root).mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        process_split(args, encoder, split)

    print("\nDone.")


if __name__ == "__main__":
    main()
