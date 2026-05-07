"""
Visualize the two evaluation-time RGB augmentations side-by-side.

PIRLNav's eval-time RGB augmentation pipeline (see
``pirlnav/policy/transforms.py``) is:

    resize -> center-crop -> [optional ColorJitter(0.4, 0.4, 0.4, 0.4)]
                          -> [optional RandomShiftsAug(pad=16)]

This script applies the two augmentations independently to the same
post-resize/center-crop observation, and saves two PNGs:

    <output-dir>/aug_jitter.png   (original | jittered)
    <output-dir>/aug_shift.png    (original | shifted)

By default the script grabs the first RGB observation of episode
``--episode-index`` from the same Habitat task config used by
``global_test.py``.  Pass ``--image PATH`` to skip the env entirely and
use any existing RGB image instead.

Usage:
    python demo_augmentations.py
    python demo_augmentations.py --image data/eval_out/.../some_frame.png
    python demo_augmentations.py --episode-index 7 --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch

import habitat  # noqa: F401  (registers nav task config keys)
from pirlnav.config import get_task_config
from pirlnav.policy.transforms import ResizeTransform, ShiftAndJitterTransform
import pirlnav  # noqa: F401  (registers ObjectNav-v2 task + dataset)
from pirlnav.task.object_nav_task import ObjectGoalNavEpisode


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) float in [0, 1] -> (H, W, 3) uint8 (first item)."""
    img = x[0].detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def hwc_uint8_to_batch_tensor(rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, H, W, 3) uint8 torch tensor (matches transform input)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image; got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return torch.from_numpy(rgb).unsqueeze(0)


def save_pair(original: np.ndarray, augmented: np.ndarray, path: Path, aug_name: str) -> None:
    """Save [original | augmented] as a single side-by-side PNG."""
    if original.shape != augmented.shape:
        raise ValueError(
            f"Shape mismatch for {aug_name}: original {original.shape} vs augmented {augmented.shape}"
        )
    sep = np.full((original.shape[0], 4, 3), 255, dtype=np.uint8)
    panel = np.concatenate([original, sep, augmented], axis=1)

    cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"Saved {aug_name} demo -> {path}")


# ---------------------------------------------------------------------------
# Habitat env helpers (mirror global_test.py)
# ---------------------------------------------------------------------------

def build_env(config_path: str) -> "habitat.Env":
    config = get_task_config(config_paths=config_path)
    return habitat.Env(config=config)


def grab_observation_from_env(config_path: str, episode_index: int) -> np.ndarray:
    """Reset to ``episode_index`` and return the RGB observation as (H, W, 3) uint8."""
    env = build_env(config_path)
    try:
        for _ in range(episode_index):
            env.reset()
        observations = env.reset()
        episode = cast(ObjectGoalNavEpisode, env.current_episode)
        print(f"Loaded episode {episode.episode_id} | scene={episode.scene_id}")

        for key in ("rgb", "robot_head_rgb", "head_rgb"):
            if key in observations:
                rgb = observations[key]
                if rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.uint8)
                return rgb
        raise RuntimeError(
            f"No RGB observation found in obs keys: {list(observations.keys())}"
        )
    finally:
        env.close()


def load_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default="configs/tasks/objectnav_mp3d.yaml",
        help="Habitat task config path (used when --image is not provided).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which episode to pull the RGB observation from.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional path to an RGB image; if set, skips Habitat env entirely.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("teleop_runs"),
        help="Where to write aug_jitter.png and aug_shift.png.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square size used by the OVRL ResNet path; must match training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch RNG (drives ColorJitter and RandomShiftsAug).",
    )
    args = parser.parse_args()

    if args.image is not None:
        print(f"Using provided image: {args.image}")
        rgb = load_image(args.image)
    else:
        print(
            f"Loading first observation from episode {args.episode_index} "
            f"(config: {args.config_path}); this may take a moment..."
        )
        rgb = grab_observation_from_env(args.config_path, args.episode_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = hwc_uint8_to_batch_tensor(rgb)

    # The "original" reference passed through the same resize + center-crop
    # so it lines up pixel-for-pixel with the augmented outputs.
    baseline = ResizeTransform(args.image_size).apply(x.clone())
    original_img = to_uint8(baseline)

    torch.manual_seed(args.seed)
    jittered = ShiftAndJitterTransform("jitter", args.image_size).apply(x.clone())
    jittered_img = to_uint8(jittered)

    torch.manual_seed(args.seed)
    shifted = ShiftAndJitterTransform("shift", args.image_size).apply(x.clone())
    shifted_img = to_uint8(shifted)

    save_pair(
        original_img,
        jittered_img,
        args.output_dir / "aug_jitter.png",
        "ColorJitter(0.4, 0.4, 0.4, 0.4)",
    )
    save_pair(
        original_img,
        shifted_img,
        args.output_dir / "aug_shift.png",
        "RandomShiftsAug(pad=16)",
    )


if __name__ == "__main__":
    main()
