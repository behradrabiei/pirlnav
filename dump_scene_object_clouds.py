"""Dump ground-truth per-scene object clouds from ``sim.semantic_annotations()``.

For every scene referenced by the dataset, walks the simulator's semantic
annotations, keeps the instances whose mpcat40 category maps to one of the 21
MP3D ObjectNav goal classes, and writes ``{scene_name}.npz`` with the same
schema as ``teleop_object_cloud.save_cloud`` -- so a future
``CachedObjectCloudSensor`` can consume either source interchangeably.

This is a strictly stronger alternative to teleop for the "complete object
cloud" experiment: it captures every annotated goal-class instance in the
scene, not just the ones the operator happened to walk past.

For each scene we create ``<output_dir>/<scene_name>/`` containing:

* ``<scene_name>.npz`` -- the cloud data (training-ready).
* ``cloud_topdown.png`` -- a world-aligned top-down visualization of the
  cloud, auto-fit to the scene's bounds.

Usage:
    python dump_scene_object_clouds.py \
        --config-path configs/tasks/objectnav_mp3d.yaml \
        --output-dir data/object_clouds/mp3d
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

import habitat

import pirlnav  # noqa: F401  (registers ObjectNav-v2 task + dataset)
from pirlnav.config import get_task_config
from pirlnav.task.semantic_map import (
    MPCAT40_TO_TASK,
    OBJECTNAV_CATEGORIES,
    PALETTE,
)


TASK_NAMES: List[str] = [name for name, _ in OBJECTNAV_CATEGORIES]


def extract_scene_cloud(sim) -> Optional[Dict[str, np.ndarray]]:
    """Iterate ``sim.semantic_annotations().objects`` and collect goal-class
    instances. Each kept instance contributes a single ``(x, y, z)`` AABB
    centre tagged with its ObjectNav task id (0..20).

    Returns ``None`` if no goal-class instances are annotated in the scene.
    """
    positions: List[np.ndarray] = []
    task_ids: List[int] = []
    for obj in sim.semantic_annotations().objects:
        if obj is None or obj.category is None:
            continue
        try:
            mp = int(obj.category.index("mpcat40"))
        except (ValueError, AttributeError):
            continue
        if not 0 <= mp < len(MPCAT40_TO_TASK):
            continue
        task = int(MPCAT40_TO_TASK[mp])
        if task < 0:
            continue
        positions.append(np.asarray(obj.aabb.center, dtype=np.float32))
        task_ids.append(task)

    if not positions:
        return None
    return {
        "obj_pos": np.stack(positions, axis=0),
        "task_ids": np.array(task_ids, dtype=np.int64),
        "labels": [TASK_NAMES[t] for t in task_ids],
    }


def render_scene_topdown(
    obj_pos: np.ndarray,
    task_ids: np.ndarray,
    labels: List[str],
    side_px: int = 768,
    margin_m: float = 2.0,
) -> np.ndarray:
    """Render a world-aligned top-down view of a full-scene object cloud.

    The viewport is auto-sized to include every object plus ``margin_m`` metres
    of padding on each side. World ``-Z`` (north) maps to canvas-up, ``+X``
    (east) to canvas-right -- the same convention as
    ``teleop_object_cloud.render_topdown``. Each centroid is drawn as a
    goal-class-coloured filled circle with its class name beside it.

    Returns an ``(side_px, side_px, 3) uint8`` RGB image.
    """
    canvas = np.full((side_px, side_px, 3), 30, dtype=np.uint8)
    if obj_pos.shape[0] == 0:
        return canvas

    x = obj_pos[:, 0]
    z = obj_pos[:, 2]
    x_min, x_max = float(x.min()) - margin_m, float(x.max()) + margin_m
    z_min, z_max = float(z.min()) - margin_m, float(z.max()) + margin_m
    span = max(x_max - x_min, z_max - z_min, 1e-3)
    x_c = 0.5 * (x_min + x_max)
    z_c = 0.5 * (z_min + z_max)
    res = span / side_px  # metres per pixel

    def world_to_px(wx: float, wz: float):
        col = int(round((wx - x_c) / res + side_px / 2))
        row = int(round((wz - z_c) / res + side_px / 2))
        return col, row

    half = int(np.ceil(span / 2))
    for k in range(-half, half + 1):
        gx, _ = world_to_px(x_c + k, z_c)
        cv2.line(canvas, (gx, 0), (gx, side_px), (50, 50, 50), 1)
        _, gz = world_to_px(x_c, z_c + k)
        cv2.line(canvas, (0, gz), (side_px, gz), (50, 50, 50), 1)

    for pos, tid, lbl in zip(obj_pos, task_ids, labels):
        px, py = world_to_px(float(pos[0]), float(pos[2]))
        if not (0 <= px < side_px and 0 <= py < side_px):
            continue
        color = tuple(int(c) for c in PALETTE[int(tid) + 2])
        cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (px + 8, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"N={obj_pos.shape[0]}  span={span:.1f}m  res={res * 100:.1f}cm/px"
        "   (up = -Z / north)",
        (10, side_px - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (220, 220, 220), 1, cv2.LINE_AA,
    )
    return canvas


def save_cloud(
    data: Optional[Dict[str, np.ndarray]],
    scene_dir: Path,
    scene_name: str,
    scene_id: str,
) -> bool:
    """Persist the cloud + a top-down PNG into ``scene_dir``.

    Writes ``<scene_name>.npz`` (schema matches ``teleop_object_cloud``) and
    ``cloud_topdown.png`` (RGB->BGR via cv2). Returns True iff anything was
    saved.
    """
    if data is None:
        print(f"  empty cloud for {scene_id}; skipping")
        return False

    scene_dir.mkdir(parents=True, exist_ok=True)
    npz_path = scene_dir / f"{scene_name}.npz"
    png_path = scene_dir / "cloud_topdown.png"

    np.savez_compressed(
        str(npz_path),
        obj_pos=data["obj_pos"],
        task_ids=data["task_ids"],
        labels=np.array(data["labels"], dtype=object),
        scene_id=np.array(scene_id),
    )
    topdown_rgb = render_scene_topdown(
        data["obj_pos"], data["task_ids"], data["labels"],
    )
    cv2.imwrite(str(png_path), cv2.cvtColor(topdown_rgb, cv2.COLOR_RGB2BGR))
    print(
        f"  saved {len(data['task_ids']):4d} objects -> {npz_path} (+png)"
    )
    return True


def _matches_filter(scene_id: str, filters: Optional[List[str]]) -> bool:
    if not filters:
        return True
    return any(f in scene_id for f in filters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path", default="configs/tasks/objectnav_mp3d.yaml",
        help="Habitat task config; only the DATASET section is used to "
             "enumerate scenes.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/object_clouds/mp3d"),
        help="Destination dir; one '<scene_name>.npz' is written per scene.",
    )
    parser.add_argument(
        "--scene-ids", nargs="*", default=None,
        help="Optional substrings; if set, only scenes whose path contains "
             "any of them are dumped.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-dump even when '<scene>.npz' already exists.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = get_task_config(config_paths=args.config_path)
    config.defrost()
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 10 ** 9
    config.freeze()
    env = habitat.Env(config=config)

    try:
        n_eps = len(env.episodes)
        if n_eps == 0:
            print("Dataset contains no episodes; nothing to dump.")
            return

        print(f"Scanning {n_eps} episode(s) for unique scenes...")
        seen: Set[str] = set()
        n_saved = 0
        n_skipped = 0
        for _ in range(n_eps):
            env.reset()
            scene_path = env.current_episode.scene_id
            scene_name = Path(scene_path).stem
            if scene_name in seen:
                continue
            if not _matches_filter(scene_path, args.scene_ids):
                seen.add(scene_name)
                continue
            seen.add(scene_name)

            scene_dir = args.output_dir / scene_name
            npz_path = scene_dir / f"{scene_name}.npz"
            if npz_path.exists() and not args.overwrite:
                print(f"[skip] {scene_name}: {npz_path} already exists "
                      "(pass --overwrite to redo)")
                n_skipped += 1
                continue

            print(f"[scene] {scene_name}")
            cloud = extract_scene_cloud(env.sim)
            if save_cloud(cloud, scene_dir, scene_name, scene_path):
                n_saved += 1

        print(
            f"Done. {n_saved} scene(s) saved, {n_skipped} pre-existing, "
            f"{len(seen)} unique scene(s) visited."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
