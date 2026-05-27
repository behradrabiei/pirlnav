#!/usr/bin/env python3
"""Compute the IL inflection-weight coefficient for an ObjectNav demo dataset.

The ``InflectionWeightSensor`` in
``pirlnav/task/sensors.py`` returns ``INFLECTION_COEF`` exactly when
``replay[t].action != replay[t-1].action`` (an "inflection point") and
``1.0`` otherwise.  The standard PIRLNav / habitat-baselines convention is::

    INFLECTION_COEF = total_steps / inflection_steps

so the cumulative weight at inflection points equals the cumulative weight
at non-inflection points and the BC loss is not dominated by long stretches
of MOVE_FORWARD.

Inputs match the on-disk layout written by ``ObjectNavDatasetV2``::

    <DATA_PATH>/<split>.json.gz                    (top-level tables, 0 eps)
    <DATA_PATH>/content/<scene>.json.gz            (per-scene episodes)

Either a single file or the dataset root works; the script auto-detects.

Usage
-----
Compute on the 21-class filtered THDA bundle on the host::

    python scripts/compute_inflection_coef.py \
        --data /projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat/train

Or after binding into the container path::

    python scripts/compute_inflection_coef.py \
        --data data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat/train

The script only uses Python stdlib (``gzip`` + ``json``), so it runs on the
login node without activating the container.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_episode_files(data_path: Path) -> Iterable[Path]:
    """Yield every per-scene ``content/<scene>.json.gz`` under a dataset split.

    Accepts either:
      * a directory containing ``content/``                (standard layout)
      * a single ``.json.gz`` file (per-scene or top-level)
    """
    if data_path.is_file():
        yield data_path
        return

    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    content_dir = data_path / "content"
    if content_dir.is_dir():
        scene_files = sorted(content_dir.glob("*.json.gz"))
        if not scene_files:
            raise FileNotFoundError(
                f"No per-scene .json.gz files under {content_dir}"
            )
        yield from scene_files
        return

    # Fallback: maybe the user pointed at <data>/content/ directly.
    scene_files = sorted(data_path.glob("*.json.gz"))
    if scene_files:
        yield from scene_files
        return

    raise FileNotFoundError(
        f"Could not find a content/ directory or any .json.gz under {data_path}"
    )


def _count_inflections_in_episode(reference_replay: List[dict]) -> Tuple[int, int]:
    """Returns ``(total_steps, inflection_steps)`` for one episode.

    Mirrors ``InflectionWeightSensor._get_observation`` exactly:
      * step 0 is never an inflection (no previous action)
      * step t in [1, T) is an inflection iff actions[t] != actions[t-1]
    """
    actions = [step["action"] for step in reference_replay]
    total = len(actions)
    if total < 2:
        return total, 0
    inflections = sum(
        1 for t in range(1, total) if actions[t] != actions[t - 1]
    )
    return total, inflections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Path to a dataset split directory (containing content/), the "
            "content/ directory itself, or a single per-scene .json.gz."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Cap on number of per-scene files to scan (-1 = all). Useful "
             "for a quick sanity check on a huge dataset.",
    )
    args = parser.parse_args()

    files = list(_iter_episode_files(args.data))
    if args.max_files > 0:
        files = files[: args.max_files]

    total_steps = 0
    total_inflections = 0
    total_episodes = 0
    skipped_no_replay = 0

    for path in files:
        with gzip.open(path, "rt") as f:
            payload = json.load(f)
        episodes = payload.get("episodes", [])
        for ep in episodes:
            replay = ep.get("reference_replay")
            if not replay:
                skipped_no_replay += 1
                continue
            t, infl = _count_inflections_in_episode(replay)
            total_steps += t
            total_inflections += infl
            total_episodes += 1
        print(
            f"[scan] {path.name:>32s}  episodes_so_far={total_episodes:>7d}  "
            f"steps={total_steps:>10d}  inflections={total_inflections:>9d}",
            file=sys.stderr,
        )

    if total_inflections == 0:
        print("ERROR: no inflection points found; cannot compute coefficient.",
              file=sys.stderr)
        return 1

    coef = total_steps / total_inflections
    inflection_frac = total_inflections / total_steps

    print()
    print("=" * 72)
    print(f"Files scanned          : {len(files)}")
    print(f"Episodes (with replay) : {total_episodes}")
    print(f"Episodes skipped       : {skipped_no_replay}")
    print(f"Total steps            : {total_steps}")
    print(f"Inflection steps       : {total_inflections}  "
          f"({100.0 * inflection_frac:.3f}% of steps)")
    print(f"INFLECTION_COEF        : {coef:.12f}")
    print("=" * 72)
    print()
    print("Paste the value into your task config:")
    print()
    print("  TASK:")
    print("    INFLECTION_WEIGHT_SENSOR:")
    print(f"      INFLECTION_COEF: {coef:.12f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
