#!/usr/bin/env python3
"""Patch missing scene_dataset_config field into every episode in an
ObjectNav dataset bundle.

habitat-lab/habitat/core/env.py line 94-95 overwrites
SIMULATOR.SCENE_DATASET with episode.scene_dataset_config at runtime.
Episodes from the THDA 70k bundle do not carry this field, which causes
attrs to default it to "", which in turn causes habitat-sim to crash with:
  AssertionError: ESP_CHECK failed: Scene Dataset `` does not exist.

This script adds:
    "scene_dataset_config": <value>
to every episode dict in every content/<scene>.json.gz under the given
dataset split directory. Idempotent: re-running with the same value is a
no-op (it overwrites but does not duplicate).

Stdlib only (gzip + json). Runs on the login node without the container.

Usage
-----
    python scripts/patch_scene_dataset_config.py \\
        --data /projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/\\
objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat/train \\
        --value "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def patch_split(split_dir: Path, value: str) -> None:
    content_dir = split_dir / "content"
    if not content_dir.is_dir():
        raise FileNotFoundError(f"content/ directory not found under {split_dir}")

    scene_files = sorted(content_dir.glob("*.json.gz"))
    if not scene_files:
        raise FileNotFoundError(f"No .json.gz files under {content_dir}")

    total_patched = 0
    total_already = 0

    for path in scene_files:
        with gzip.open(path, "rt") as f:
            payload = json.load(f)

        episodes = payload.get("episodes", [])
        patched = 0
        already = 0
        for ep in episodes:
            if ep.get("scene_dataset_config") == value:
                already += 1
            else:
                ep["scene_dataset_config"] = value
                patched += 1

        if patched > 0:
            with gzip.open(path, "wt") as f:
                json.dump(payload, f)

        total_patched += patched
        total_already += already
        print(
            f"[patch] {path.name:>32s}  patched={patched:>4d}  already_ok={already:>4d}",
            file=sys.stderr,
        )

    print(f"\nDone. Total episodes patched: {total_patched}  already correct: {total_already}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a dataset split directory containing content/",
    )
    p.add_argument(
        "--value",
        default="data/scene_datasets/mp3d/mp3d.scene_dataset_config.json",
        help="Value to write into scene_dataset_config on every episode.",
    )
    args = p.parse_args()
    patch_split(args.data, args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
