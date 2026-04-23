"""Precompute frozen DINOv2 CLS features for pirlnav IL episodes.

Because pirlnav's IL training is pure teacher-forcing (the env is stepped with
the expert's ``next_actions``), the RGB observed at step ``t`` of episode
``e`` is a deterministic function of ``(episode_id, t)``. We can therefore
run the DINOv2 forward pass once per step per episode, offline, and cache the
resulting 768-dim CLS token for the online trainer to reuse.

Output layout::

    {cache_root}/{scene}/{episode_id}.pt
        |-- dino_cls : torch.Tensor (T+1, 768) float16

where ``T = len(episode.reference_replay) - 1`` (the first entry of
``reference_replay`` is a no-op init; we still record the initial-reset CLS
so the full length matches the ``DemonstrationSensor.timestep`` counter).

The script is resumable: episodes whose ``.pt`` file already exists are
skipped. Typical runtime on a single RTX 5090 is ~12-15 min for the 302+53
episodes of the one-scene MP3D subset.

Usage::

    python scripts/precompute_dinov2_features.py \
        --config configs/tasks/objectnav_mp3d.yaml \
        --split both \
        --scene 17DRP5sb8fy \
        --cache-root data/dinov2_cache
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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
    p.add_argument("--resize-h", type=int, default=476)
    p.add_argument("--resize-w", type=int, default=630)
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
    return p.parse_args()


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
    print(f"\n=== split={split} ===")
    task_cfg = build_task_config_for_split(args.config, split)
    env = habitat.Env(config=task_cfg)

    scene_dir = Path(args.cache_root) / args.scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(env.episodes)
    num_eps = len(episodes)
    if args.max_episodes > 0:
        num_eps = min(num_eps, args.max_episodes)
        episodes = episodes[:num_eps]
    print(f"  total episodes in split: {num_eps}")

    t0 = time.time()
    done_ct = 0
    skip_ct = 0
    total_frames = 0
    resize_hw = (args.resize_h, args.resize_w)
    transform = DINOv2Transform(augmentations_name="dinov2", size_hw=resize_hw)

    try:
        for ep_idx, ep in enumerate(episodes):
            out_path = scene_dir / f"{ep.episode_id}.pt"

            if out_path.exists() and not args.overwrite:
                skip_ct += 1
                done_ct += 1
                continue

            # Drive episode selection deterministically: tell the iterator
            # which episode to return on the next reset, then reset.
            env.episode_iterator = iter([ep])
            obs = env.reset()
            cur = env.current_episode
            if cur.episode_id != ep.episode_id:
                raise RuntimeError(
                    "episode iterator gave us a different episode than "
                    f"expected: wanted {ep.episode_id}, got {cur.episode_id}"
                )

            replay = list(cur.reference_replay)
            if len(replay) == 0:
                print(f"  [warn] ep={cur.episode_id} has empty reference_replay")
                continue

            feats = [encode_rgb(encoder, transform, obs["rgb"], args.device)]

            start_idx = 1 if hasattr(replay[0], "action") else 0
            for r in replay[start_idx:]:
                action_name = r.action
                if action_name not in ACTION_NAME_TO_ID:
                    action_name = str(action_name).upper().split(".")[-1]
                a = ACTION_NAME_TO_ID[action_name]
                step_obs = env.step(a)
                feats.append(
                    encode_rgb(encoder, transform, step_obs["rgb"], args.device)
                )
                if env.episode_over:
                    break

            feat_tensor = torch.stack(feats, dim=0).contiguous()
            total_frames += feat_tensor.shape[0]
            torch.save({"dino_cls": feat_tensor}, out_path)
            done_ct += 1

            if done_ct % 25 == 0 or done_ct == num_eps:
                elapsed = time.time() - t0
                fps = total_frames / max(elapsed, 1e-6)
                print(
                    f"  [{split}] {done_ct}/{num_eps} eps "
                    f"(skipped {skip_ct}, frames {total_frames}, "
                    f"{fps:.1f} fps, {elapsed:.1f}s elapsed)"
                )
    finally:
        env.close()

    elapsed = time.time() - t0
    print(
        f"  done: {done_ct}/{num_eps} eps ({skip_ct} skipped), "
        f"{total_frames} frames, {elapsed:.1f}s total"
    )


def main() -> None:
    args = parse_args()
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
