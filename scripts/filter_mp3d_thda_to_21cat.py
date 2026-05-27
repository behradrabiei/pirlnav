#!/usr/bin/env python3
"""Filter the THDA 70k MP3D ObjectNav bundle down to the canonical 21 classes.

The THDA 70k dataset that ships at::

    <SRC>/objectnav_mp3d_thda_70k/objectnav/objectnav_mp3d_thda_70k/{split}/
        {split}.json.gz                # top-level tables, 0 episodes
        content/<scene>.json.gz        # 56 per-scene episode bundles

contains a 28-entry ``category_to_task_category_id`` table.  The first 21
entries are the canonical MP3D ObjectNav benchmark classes (matching
``pirlnav.task.semantic_map.OBJECTNAV_CATEGORIES``); the last 7 are
THDA-paper "treasure hunt" augmented goals (foodstuff, stationery, fruit,
plaything, hand_tool, game_equipment, kitchenware) that have no
``mpcat40_idx -> task_id`` mapping in the canonical pirlnav table, so they
would never appear in the object-cloud sensor's class-id stream during
training.

This script writes a sibling bundle that contains only the canonical 21
classes::

    <DST>/objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat/{split}/
        {split}.json.gz                # canonical-21 tables, 0 episodes
        content/<scene>.json.gz        # episodes/goals filtered to canonical 21

Properties of the output:
  * ``category_to_task_category_id`` keeps the original task ids 0-20 (no
    renumbering), so ``ObjectGoalSensor.high[0] == 20``.
  * ``goals_by_category`` is filtered by suffix match (cat names like
    ``tv_monitor`` and ``chest_of_drawers`` contain underscores so we cannot
    just split on ``_``).
  * ``episodes`` are filtered by ``object_category in CANONICAL_21``.
  * Drops episodes that are missing ``reference_replay`` (they would be
    skipped by the IL trainer anyway).
  * Idempotent: re-runnable, writes to a separate output root.

Stdlib only (``gzip`` + ``json``); runs on the login node without the
container.

Usage
-----

::

    python scripts/filter_mp3d_thda_to_21cat.py \\
        --src /projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k/objectnav/objectnav_mp3d_thda_70k \\
        --dst /projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat \\
        --splits train
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


# Canonical 21-class MP3D ObjectNav table (must match
# pirlnav.task.semantic_map.OBJECTNAV_CATEGORIES order/ids).
CANONICAL_21: List[str] = [
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
]
CANONICAL_SET = set(CANONICAL_21)


def load_gz_json(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def dump_gz_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def _filter_category_table(table: Dict[str, int]) -> Dict[str, int]:
    """Restrict to canonical 21 keys, preserving the original numeric ids."""
    out = {k: int(v) for k, v in table.items() if k in CANONICAL_SET}
    missing = sorted(CANONICAL_SET - set(out.keys()))
    if missing:
        raise RuntimeError(
            f"Source dataset is missing canonical categories: {missing}"
        )
    return out


def _filter_goals_by_category(
    gbc: Dict[str, list],
) -> Dict[str, list]:
    """Keep only goals_by_category entries whose suffix is one of the 21.

    Keys look like ``<basename(scene_id)>_<category>``.  We cannot
    ``rsplit("_", 1)`` because some category names (``tv_monitor``,
    ``chest_of_drawers``, ``gym_equipment``) contain underscores -- so we
    suffix-match instead.
    """
    out: Dict[str, list] = {}
    for key, goals in gbc.items():
        for cat in CANONICAL_SET:
            if key.endswith(f"_{cat}"):
                out[key] = goals
                break
    return out


def _filter_scene_payload(payload: dict) -> tuple[dict, dict]:
    """Returns ``(filtered_payload, stats_dict)``."""
    if "category_to_task_category_id" not in payload:
        raise RuntimeError("scene payload missing category_to_task_category_id")

    new_task_map = _filter_category_table(payload["category_to_task_category_id"])
    new_mp3d_map = _filter_category_table(payload["category_to_mp3d_category_id"])
    if set(new_task_map.keys()) != set(new_mp3d_map.keys()):
        raise RuntimeError(
            "Filtered task and mp3d maps have different keys: "
            f"{set(new_task_map.keys()) ^ set(new_mp3d_map.keys())}"
        )

    src_episodes = payload.get("episodes", []) or []
    src_count = len(src_episodes)
    by_cat_before = Counter(ep.get("object_category", "<missing>") for ep in src_episodes)

    kept_episodes: List[dict] = []
    skipped_no_replay = 0
    for ep in src_episodes:
        if ep.get("object_category") not in CANONICAL_SET:
            continue
        if not ep.get("reference_replay"):
            skipped_no_replay += 1
            continue
        # habitat/core/env.py line 94-95 overwrites SIMULATOR.SCENE_DATASET
        # with episode.scene_dataset_config at runtime.  The THDA source
        # episodes omit this field; attrs defaults it to "", which causes
        # habitat-sim to crash with "Scene Dataset `` does not exist".
        # Set it explicitly so it resolves to the bind-mounted config file.
        ep = dict(ep)
        if not ep.get("scene_dataset_config"):
            ep["scene_dataset_config"] = (
                "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
            )
        kept_episodes.append(ep)

    by_cat_after = Counter(ep["object_category"] for ep in kept_episodes)

    src_gbc = payload.get("goals_by_category", {}) or {}
    new_gbc = _filter_goals_by_category(src_gbc)

    out_payload = {
        "episodes": kept_episodes,
        "category_to_task_category_id": new_task_map,
        "category_to_mp3d_category_id": new_mp3d_map,
        "goals_by_category": new_gbc,
    }
    if "content_scenes_path" in payload:
        out_payload["content_scenes_path"] = payload["content_scenes_path"]

    stats = {
        "src_episodes": src_count,
        "kept_episodes": len(kept_episodes),
        "skipped_no_replay": skipped_no_replay,
        "src_goals_by_category": len(src_gbc),
        "kept_goals_by_category": len(new_gbc),
        "by_cat_before": by_cat_before,
        "by_cat_after": by_cat_after,
    }
    return out_payload, stats


def _filter_top_payload(payload: dict) -> dict:
    """Top-level <split>.json.gz only carries category tables (0 episodes).

    We trim the tables to canonical 21 and keep the empty episodes list.
    """
    new_task_map = _filter_category_table(payload["category_to_task_category_id"])
    new_mp3d_map = _filter_category_table(payload["category_to_mp3d_category_id"])
    out = {
        "category_to_task_category_id": new_task_map,
        "category_to_mp3d_category_id": new_mp3d_map,
        "episodes": payload.get("episodes", []),
    }
    return out


def process_split(src_split: Path, dst_split: Path, split: str) -> dict:
    if not src_split.is_dir():
        raise FileNotFoundError(f"Source split not found: {src_split}")
    src_top = src_split / f"{split}.json.gz"
    src_content = src_split / "content"
    if not src_top.is_file():
        raise FileNotFoundError(f"Missing top-level {src_top}")
    if not src_content.is_dir():
        raise FileNotFoundError(f"Missing content dir {src_content}")

    top_payload = load_gz_json(src_top)
    new_top = _filter_top_payload(top_payload)
    dump_gz_json(new_top, dst_split / f"{split}.json.gz")

    scene_files = sorted(src_content.glob("*.json.gz"))
    if not scene_files:
        raise RuntimeError(f"No per-scene files in {src_content}")

    total_src = 0
    total_kept = 0
    total_skipped_replay = 0
    total_src_gbc = 0
    total_kept_gbc = 0
    by_cat_after_total: Counter = Counter()
    by_cat_dropped_total: Counter = Counter()

    for scene_path in scene_files:
        scene_payload = load_gz_json(scene_path)
        new_payload, stats = _filter_scene_payload(scene_payload)
        out_path = dst_split / "content" / scene_path.name
        dump_gz_json(new_payload, out_path)

        total_src += stats["src_episodes"]
        total_kept += stats["kept_episodes"]
        total_skipped_replay += stats["skipped_no_replay"]
        total_src_gbc += stats["src_goals_by_category"]
        total_kept_gbc += stats["kept_goals_by_category"]
        by_cat_after_total.update(stats["by_cat_after"])
        for cat, cnt in stats["by_cat_before"].items():
            if cat not in CANONICAL_SET:
                by_cat_dropped_total[cat] += cnt

        print(
            f"[{split}] {scene_path.name:>32s}  "
            f"src={stats['src_episodes']:>4d}  "
            f"kept={stats['kept_episodes']:>4d}  "
            f"dropped_class={stats['src_episodes'] - stats['kept_episodes'] - stats['skipped_no_replay']:>4d}  "
            f"dropped_no_replay={stats['skipped_no_replay']:>3d}  "
            f"gbc={stats['kept_goals_by_category']:>2d}/{stats['src_goals_by_category']:>2d}",
            file=sys.stderr,
        )

    return {
        "split": split,
        "n_scene_files": len(scene_files),
        "src_episodes": total_src,
        "kept_episodes": total_kept,
        "skipped_no_replay": total_skipped_replay,
        "src_goals_by_category": total_src_gbc,
        "kept_goals_by_category": total_kept_gbc,
        "by_cat_after": by_cat_after_total,
        "by_cat_dropped": by_cat_dropped_total,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Source bundle root, e.g. .../objectnav_mp3d_thda_70k/objectnav/"
            "objectnav_mp3d_thda_70k.  Must contain <split>/{<split>.json.gz, "
            "content/<scene>.json.gz}."
        ),
    )
    p.add_argument(
        "--dst",
        type=Path,
        required=True,
        help=(
            "Destination bundle root.  Will be created (with parents) and "
            "the same {split}/ layout will be written under it."
        ),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Which split directories to process under src/ (default: train).",
    )
    args = p.parse_args()

    if args.dst.resolve() == args.src.resolve():
        print("ERROR: --dst must differ from --src to avoid in-place overwrite.",
              file=sys.stderr)
        return 1

    aggregate: List[dict] = []
    for split in args.splits:
        src_split = args.src / split
        dst_split = args.dst / split
        if not src_split.is_dir():
            print(f"[skip] split not found: {src_split}", file=sys.stderr)
            continue
        result = process_split(src_split, dst_split, split)
        aggregate.append(result)

    print()
    print("=" * 76)
    for r in aggregate:
        dropped = r["src_episodes"] - r["kept_episodes"] - r["skipped_no_replay"]
        keep_pct = (
            100.0 * r["kept_episodes"] / r["src_episodes"]
            if r["src_episodes"] else 0.0
        )
        print(f"split={r['split']}  scenes={r['n_scene_files']}  "
              f"src={r['src_episodes']}  kept={r['kept_episodes']} "
              f"({keep_pct:.2f}%)  "
              f"dropped_class={dropped}  "
              f"dropped_no_replay={r['skipped_no_replay']}")
        print(f"  goals_by_category: {r['kept_goals_by_category']} / {r['src_goals_by_category']}")
        if r["by_cat_dropped"]:
            print(f"  dropped THDA-extra classes: "
                  + ", ".join(f"{c}={n}" for c, n in r['by_cat_dropped'].most_common()))
        print(f"  per-canonical-class kept (top): "
              + ", ".join(f"{c}={n}" for c, n in r['by_cat_after'].most_common(7)))
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
