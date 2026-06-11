"""
Headless semantic-segmentation + object-cloud sanity check.

The online object-cloud variant
(configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full_252.yaml)
builds its ego object cloud from the live habitat-sim SEMANTIC_SENSOR + DEPTH
each step. This script verifies, per scene and without a display, that the
semantic pipeline is actually producing useful labels before committing a
multi-day training run. For each scene it builds the full training sensor
stack, steps a short slice of the expert reference_replay, and asserts:

  1. the raw `semantic` frame carries many distinct instance ids,
  2. `build_instance_to_task_id(sim)` maps some instances into a goal class
     [0, 20] (otherwise the scene's annotations are missing / mis-indexed),
  3. the packed `ego_object_cloud` observation accumulates non-padding rows
     (task_id != -1) with task ids in [0, 20] once the agent has moved,
  4. `depth` is unnormalized (NORMALIZE_DEPTH=False -> max > 1.0 metres),
     which the back-projection in pirlnav/task/object_cloud.py relies on.

For visual validation it also saves, per scene, a side-by-side PNG
[RGB | semantic-by-task-class | semantic-by-instance] of the most informative
frame seen during the replay (the one with the most goal-class pixels), so you
can eyeball that goal objects are segmented where you'd expect.

Run inside the container with the standard full-MP3D binds:
  python scripts/verify_semantic_segmentation.py                       # auto-pick 3 scenes
  python scripts/verify_semantic_segmentation.py --scenes 17DRP5sb8fy  # specific scene(s)
  python scripts/verify_semantic_segmentation.py --num-scenes 5 --max-actions 80
  python scripts/verify_semantic_segmentation.py --out-dir probe_logs/semseg --no-save-images

Exit code is 0 iff every checked scene PASSes, else 1 (handy in CI / sweeps).
"""

import argparse
import glob
import gzip
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DEFAULT_CONFIG = (
    "configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full_252.yaml"
)
DEFAULT_DATA_PATH = (
    "data/datasets/objectnav/objectnav_mp3d/"
    "objectnav_mp3d_thda_70k_21cat/{split}/{split}.json.gz"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="MP3D scene ids to check. If omitted, auto-pick the first "
        "--num-scenes scenes that have a content file.",
    )
    p.add_argument("--num-scenes", type=int, default=3,
                   help="How many scenes to auto-pick when --scenes is unset.")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Experiment yaml (only the TASK_CONFIG is exercised).")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                   help="DATA_PATH template; {split} is filled by habitat-lab.")
    p.add_argument("--split", default="train")
    p.add_argument("--max-actions", type=int, default=60,
                   help="Replay actions to step before checking the cloud.")
    p.add_argument("--max-objects", type=int, default=100,
                   help="Override EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS.")
    p.add_argument("--out-dir", default="probe_logs/semseg_verify",
                   help="Directory for the saved RGB/semantic PNG panels.")
    p.add_argument("--save-images", dest="save_images", action="store_true",
                   default=True, help="Save RGB/semantic panels (default).")
    p.add_argument("--no-save-images", dest="save_images", action="store_false",
                   help="Skip writing the visual panels.")
    return p.parse_args()


def discover_scenes(data_path, split, n):
    content_dir = os.path.join(
        os.path.dirname(data_path.format(split=split)), "content"
    )
    paths = sorted(glob.glob(os.path.join(content_dir, "*.json.gz")))
    if not paths:
        raise FileNotFoundError(f"no per-scene content files under {content_dir}")
    return [os.path.basename(p)[: -len(".json.gz")] for p in paths[:n]]


def content_has_replay(content_path):
    """Cheap pre-check that the per-scene content file decodes and contains
    at least one episode with a non-empty reference_replay."""
    with gzip.open(content_path, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))
    return any(ep.get("reference_replay") for ep in data.get("episodes", []))


def replay_action_names(episode, max_actions):
    """Action-name slice from the env's *current* episode so the stepped
    actions actually correspond to its start pose (mirrors
    scripts/test_scene_replay.py: replay[0] is the start pose, actions
    begin at index 1)."""
    replay = list(episode.reference_replay or [])
    if not replay:
        return None
    start_idx = 1 if hasattr(replay[0], "action") else 0
    names = []
    for step in replay[start_idx: max_actions + start_idx]:
        names.append(str(step.action).upper().split(".")[-1])
    return names or None


def _colorize_semantic(sem, table):
    """Return (task_class_rgb, instance_rgb, present_task_names) for a raw
    first-person semantic frame, coloring goal classes via the shared PALETTE.
    """
    import numpy as np
    from pirlnav.task.semantic_map import (
        PALETTE, OCCUPIED, OBJECTNAV_CATEGORIES, NUM_CATEGORIES,
    )

    if sem.ndim == 3:
        sem = sem[..., 0]
    inst = np.clip(sem.astype(np.int64), 0, len(table) - 1)
    task = table[inst]

    # Goal classes -> task+2 channel in PALETTE; everything else -> OCCUPIED.
    channel = np.where(task >= 0, task + 2, OCCUPIED).astype(np.int64)
    task_rgb = PALETTE[channel].astype(np.uint8)

    # Per-instance pseudo-color so all object boundaries are visible; id 0
    # (typically background/structure) stays black.
    ids = sem.astype(np.int64)
    inst_rgb = np.stack(
        [(ids * 131) % 256, (ids * 197) % 256, (ids * 251) % 256], axis=-1
    ).astype(np.uint8)
    inst_rgb[ids == 0] = 0

    present = sorted({int(t) for t in np.unique(task) if 0 <= t <= NUM_CATEGORIES - 1})
    names = [OBJECTNAV_CATEGORIES[t][0] for t in present]
    return task_rgb, inst_rgb, names


def _save_panel(path, rgb, sem_task_rgb, sem_inst_rgb):
    import cv2
    import numpy as np

    rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
    panel = np.concatenate([rgb, sem_task_rgb, sem_inst_rgb], axis=1)
    labels = ["RGB", "semantic (task class)", "semantic (instance)"]
    w = rgb.shape[1]
    for i, text in enumerate(labels):
        cv2.putText(panel, text, (i * w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def check_scene(scene, args):
    import numpy as np
    import habitat
    import pirlnav  # noqa: F401  (registers ObjectNav-v2, sensors, etc.)
    from pirlnav.config import get_config
    from pirlnav.task.semantic_map import build_instance_to_task_id, NUM_CATEGORIES

    content_path = os.path.join(
        os.path.dirname(args.data_path.format(split=args.split)),
        "content", f"{scene}.json.gz",
    )
    if not os.path.exists(content_path):
        print(f"[verify] {scene}: SETUP_FAIL no content file {content_path}")
        return False
    if not content_has_replay(content_path):
        print(f"[verify] {scene}: SKIP no episode with reference_replay")
        return False

    cfg = get_config(args.config, opts=None)
    cfg.defrost()
    cfg.TASK_CONFIG.DATASET.SPLIT = args.split
    cfg.TASK_CONFIG.DATASET.DATA_PATH = args.data_path
    cfg.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene]
    cfg.TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS = args.max_objects
    cfg.freeze()

    env = habitat.Env(config=cfg.TASK_CONFIG)
    try:
        obs = env.reset()
        action_names = replay_action_names(
            env.current_episode, args.max_actions
        )
        if not action_names:
            print(f"[verify] {scene}: SKIP current episode has no "
                  "reference_replay")
            return False

        if "semantic" not in obs:
            print(f"[verify] {scene}: FAIL no 'semantic' observation "
                  "(SEMANTIC_SENSOR not enabled?)")
            return False
        if "ego_object_cloud" not in obs:
            print(f"[verify] {scene}: FAIL no 'ego_object_cloud' observation")
            return False

        # (1) raw semantic frame carries many distinct instance ids.
        n_inst = int(np.unique(np.asarray(obs["semantic"])).size)

        # (4) depth is unnormalized (metres), as the projection assumes.
        depth_max = float(np.asarray(obs["depth"]).max())

        # (2) instance -> goal-class mapping is populated for this scene.
        table = build_instance_to_task_id(env.sim)
        n_goal_instances = int(np.sum((table >= 0) & (table <= NUM_CATEGORIES - 1)))

        # Track the frame with the most goal-class pixels for the saved panel.
        def goal_pixels(o):
            sem = np.asarray(o["semantic"])
            sem = sem[..., 0] if sem.ndim == 3 else sem
            inst = np.clip(sem.astype(np.int64), 0, len(table) - 1)
            return int(np.sum(table[inst] >= 0))

        best = (goal_pixels(obs), obs) if args.save_images else None

        # (3) step the replay, then inspect the accumulated ego cloud.
        for name in action_names:
            obs = env.step({"action": name})
            if args.save_images:
                gp = goal_pixels(obs)
                if gp > best[0]:
                    best = (gp, obs)
            if env.episode_over:
                break

        if args.save_images:
            os.makedirs(args.out_dir, exist_ok=True)
            b_obs = best[1]
            task_rgb, inst_rgb, names = _colorize_semantic(
                np.asarray(b_obs["semantic"]), table
            )
            out_path = os.path.join(args.out_dir, f"{scene}_semseg.png")
            _save_panel(out_path, b_obs["rgb"], task_rgb, inst_rgb)
            print(f"[verify] {scene}: saved {out_path} "
                  f"(goal classes in frame: {', '.join(names) or 'none'})")

        cloud = np.asarray(obs["ego_object_cloud"])
        tids = cloud[:, 0]
        non_pad = tids != -1.0
        n_objects = int(np.sum(non_pad))
        ok_ids = bool(
            n_objects == 0
            or (tids[non_pad].min() >= 0 and tids[non_pad].max() <= NUM_CATEGORIES - 1)
        )

        passed = (
            n_inst > 1
            and depth_max > 1.0
            and n_goal_instances > 0
            and n_objects > 0
            and ok_ids
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"[verify] {scene}: {status} "
            f"semantic_instances={n_inst} goal_instances={n_goal_instances} "
            f"depth_max={depth_max:.2f}m cloud_objects={n_objects}/{args.max_objects} "
            f"task_ids_in_range={ok_ids}"
        )
        return passed
    finally:
        env.close()


def main():
    args = parse_args()
    try:
        scenes = args.scenes or discover_scenes(
            args.data_path, args.split, args.num_scenes
        )
    except Exception as exc:
        print(f"[verify] SETUP_FAIL scene discovery: {exc}")
        return 1

    print(f"[verify] checking {len(scenes)} scene(s): {', '.join(scenes)}")
    results = {}
    for scene in scenes:
        try:
            results[scene] = check_scene(scene, args)
        except Exception:
            print(f"[verify] {scene}: ERROR during check")
            traceback.print_exc()
            results[scene] = False

    n_pass = sum(1 for v in results.values() if v)
    print(f"[verify] summary: {n_pass}/{len(results)} scenes passed")
    return 0 if results and n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
