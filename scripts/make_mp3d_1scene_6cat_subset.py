#!/usr/bin/env python3
"""Build a single-scene, 6-category MP3D ObjectNav subset from the THDA 70k
demonstrations, remapped to HM3D's 6-class ObjectNav taxonomy, with a
reproducible stratified train/val split.

The output matches pirlnav's `ObjectNavDatasetV2` layout, replicated for
both splits:

    <out_root>/train/train.json.gz                  # top-level tables, 0 eps
    <out_root>/train/content/<scene>.json.gz        # train eps + goals
    <out_root>/val/val.json.gz                      # top-level tables, 0 eps
    <out_root>/val/content/<scene>.json.gz          # held-out eps + goals

MP3D's THDA dataset labels sofas as "sofa" (HM3D uses the same name), plants
as "plant", and tvs as "tv_monitor" -- which already matches HM3D's
canonical categories -- so no string-level remap is needed. We only remap
the numeric `category_to_task_category_id` table so the ObjectGoalSensor
produces IDs in [0, 5] (6 HM3D classes), consistent with any pretrained
model's embedding size.

Notes on the split:
 - Stratified by `object_category` so each class lands in both splits.
 - Deterministic: driven by (seed, category, original episode order).
 - Per-episode `goals` is also populated (dup of `goals_by_category`) so
   the eval path, which coerces the dataset to `ObjectNav-v1`, can find the
   goals directly on each episode.
"""

import argparse
import gzip
import json
import os
import random
from collections import defaultdict
from pathlib import Path

SRC_DEFAULT = (
    "/data/hm3d_datasets/MP3D/data/datasets/objectnav/"
    "objectnav_mp3d_thda_70k/objectnav/objectnav_mp3d_thda_70k/train"
)

OUT_DEFAULT = (
    "data/datasets/objectnav/objectnav_mp3d/"
    "objectnav_mp3d_1scene_6cat"
)

HM3D_CATEGORIES = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
HM3D_TASK_ID = {c: i for i, c in enumerate(HM3D_CATEGORIES)}


def load_gz_json(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def dump_gz_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def stratified_split(episodes, val_frac: float, seed: int):
    """Split `episodes` into (train, val) lists, stratified by
    object_category. Deterministic given the seed.

    Each category's episodes are shuffled independently and the last
    ``ceil(n * val_frac)`` episodes are held out for val. `ceil` (rather
    than `floor`) ensures that a category with a small count still
    contributes at least one val episode if val_frac > 0.
    """
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for ep in episodes:
        by_cat[ep["object_category"]].append(ep)

    train, val = [], []
    per_cat_stats = {}
    for cat, eps in sorted(by_cat.items()):
        shuffled = list(eps)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = 0 if val_frac <= 0 else min(n, max(1, int(round(n * val_frac))))
        val_cat = shuffled[:n_val]
        train_cat = shuffled[n_val:]
        train.extend(train_cat)
        val.extend(val_cat)
        per_cat_stats[cat] = (len(train_cat), len(val_cat))
    return train, val, per_cat_stats


def write_split(
    split_name: str,
    out_root: Path,
    scene: str,
    episodes,
    task_map,
    mp3d_map,
    goals_by_category,
):
    # Re-index episode ids within the split so they are contiguous and
    # deterministic (ObjectNavDatasetV2.from_json also overwrites these,
    # but it's nice to have them sensible in the JSON too).
    episodes = [dict(ep) for ep in episodes]  # shallow copy so we don't mutate caller's list
    for i, ep in enumerate(episodes):
        ep["episode_id"] = str(i)
        # Populate per-episode `goals` so the eval path (which uses
        # ObjectNav-v1) can read them directly. goals_key is
        # f"{basename(scene_id)}_{object_category}".
        goals_key = f"{os.path.basename(ep['scene_id'])}_{ep['object_category']}"
        ep["goals"] = goals_by_category.get(goals_key, [])

    top_out = {
        "episodes": [],
        "category_to_task_category_id": task_map,
        "category_to_mp3d_category_id": mp3d_map,
    }
    scene_out = {
        "episodes": episodes,
        "category_to_task_category_id": task_map,
        "category_to_mp3d_category_id": mp3d_map,
        "goals_by_category": goals_by_category,
    }

    top_path = out_root / split_name / f"{split_name}.json.gz"
    scene_path = out_root / split_name / "content" / f"{scene}.json.gz"
    dump_gz_json(top_out, top_path)
    dump_gz_json(scene_out, scene_path)
    return top_path, scene_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Scene id, e.g. 17DRP5sb8fy")
    p.add_argument("--src", default=SRC_DEFAULT,
                   help="Source THDA train split (contains train.json.gz + content/)")
    p.add_argument("--out", default=OUT_DEFAULT,
                   help="Output dataset root (script creates <out>/train/ and optionally <out>/val/)")
    p.add_argument(
        "--scene-dataset-config",
        default="data/scene_datasets/mp3d/mp3d.scene_dataset_config.json",
        help="Path (relative to repo root) written into every episode's "
             "scene_dataset_config field -- habitat-sim 0.2.5 requires this.",
    )
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of episodes (per category) to hold out for val. "
                        "Set to 0 to disable val split.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the stratified split.")
    args = p.parse_args()

    src_root = Path(args.src)
    src_top = src_root / "train.json.gz"
    src_scene = src_root / "content" / f"{args.scene}.json.gz"
    for pth in (src_top, src_scene):
        if not pth.is_file():
            raise FileNotFoundError(pth)

    src_top_data = load_gz_json(src_top)
    src_scene_data = load_gz_json(src_scene)

    src_task_map = src_top_data["category_to_task_category_id"]
    src_mp3d_map = src_top_data["category_to_mp3d_category_id"]
    missing = [c for c in HM3D_CATEGORIES if c not in src_task_map]
    if missing:
        raise RuntimeError(f"Source MP3D dataset is missing categories {missing}")

    new_task_map = dict(HM3D_TASK_ID)
    new_mp3d_map = {c: src_mp3d_map[c] for c in HM3D_CATEGORIES}

    total_before = len(src_scene_data.get("episodes", []))
    kept_eps = [ep for ep in src_scene_data["episodes"]
                if ep.get("object_category") in HM3D_TASK_ID]
    for ep in kept_eps:
        ep["scene_dataset_config"] = args.scene_dataset_config

    # Keys in goals_by_category are "<basename(scene_id)>_<category>".  We
    # cannot `rsplit("_", 1)` because category names like `tv_monitor` and
    # `chest_of_drawers` contain underscores.  Match by suffix against the
    # set of categories we want instead.
    gbc_src = src_scene_data.get("goals_by_category", {})
    gbc_out = {}
    for key, goals in gbc_src.items():
        for cat in HM3D_TASK_ID:
            if key.endswith(f"_{cat}"):
                gbc_out[key] = goals
                break

    train_eps, val_eps, per_cat = stratified_split(
        kept_eps, val_frac=args.val_frac, seed=args.seed,
    )

    out_root = Path(args.out)
    train_top, train_scene = write_split(
        "train", out_root, args.scene,
        train_eps, new_task_map, new_mp3d_map, gbc_out,
    )
    if args.val_frac > 0 and len(val_eps) > 0:
        val_top, val_scene = write_split(
            "val", out_root, args.scene,
            val_eps, new_task_map, new_mp3d_map, gbc_out,
        )
    else:
        val_top = val_scene = None

    print(f"Source episodes in scene: {total_before}")
    print(f"Kept (6 classes):         {len(kept_eps)}")
    print(f"Split seed={args.seed}, val_frac={args.val_frac}:")
    print(f"  train: {len(train_eps)}   val: {len(val_eps)}")
    print("  per-category (train / val):")
    for c in HM3D_CATEGORIES:
        t, v = per_cat.get(c, (0, 0))
        print(f"    {c:>12}: {t:>3} / {v:>3}")
    print(f"Wrote {train_top}")
    print(f"Wrote {train_scene}")
    if val_scene is not None:
        print(f"Wrote {val_top}")
        print(f"Wrote {val_scene}")
    print(f"goals_by_category entries: {len(gbc_out)} (of {len(gbc_src)})")


if __name__ == "__main__":
    main()
