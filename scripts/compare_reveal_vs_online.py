"""
Compare the ORACLE_REVEAL object cloud against the online (rendered) one.

Replays the *same* expert episode twice per scene -- once with the online
depth+semantic pipeline (il_objectnav_mp3d_dinov2_object_cloud_full_252.yaml)
and once with the progressive-reveal oracle
(il_objectnav_mp3d_dinov2_oracle_reveal_full_252.yaml) -- matches tracked
objects by semantic instance id, and reports per scene:

  * objects in each cloud at episode end, matched / only-online /
    only-reveal with per-category breakdown,
  * reveal-timing delta for matched objects (step first seen by the oracle
    minus step first seen online; negative = oracle reveals earlier),
  * centroid delta for matched objects (GT AABB center vs the online
    mask-area-weighted surface centroid -- expected nonzero, we quantify it),
  * a 3-panel top-down PNG [online | reveal | overlay].

Use the results to tune REVEAL_MIN_ANGULAR_SIZE / REVEAL_OCCLUSION_MARGIN if
the reveal gate is systematically over- or under-permissive.

CAVEAT -- the online baseline is NOT ground truth. A per-frame probe
(2026-06-09, scene 1LXtFkjw3qL) showed two systematic artifacts in the
online pipeline: (a) DEPTH_SENSOR clamps at MAX_DEPTH=5m, so objects seen
beyond 5m are deposited at exactly 5m along the view ray; (b) the semantic
mesh (*_semantic.ply) has holes the render mesh (.glb) does not, so
instance ids bleed through walls and get the occluder's depth -- phantom
objects meters from the real geometry. Every reveal "miss" investigated
was one of these online artifacts, not a gate failure. Treat
recall-of-online as a smoke signal, and diagnose misses against the
vertex-cache positions before blaming the reveal gates.

Run inside the container with the standard full-MP3D binds:
  python scripts/compare_reveal_vs_online.py                      # 2 scenes
  python scripts/compare_reveal_vs_online.py --scenes 17DRP5sb8fy --max-actions 200
"""

import argparse
import os
import sys
import traceback
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from verify_semantic_segmentation import (  # noqa: E402
    DEFAULT_DATA_PATH,
    content_has_replay,
    discover_scenes,
    replay_action_names,
)

ONLINE_CONFIG = (
    "configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full_252.yaml"
)
REVEAL_CONFIG = (
    "configs/experiments/il_objectnav_mp3d_dinov2_oracle_reveal_full_252.yaml"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="*", default=None,
                   help="MP3D scene ids; auto-picks --num-scenes if unset.")
    p.add_argument("--num-scenes", type=int, default=2)
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--split", default="train")
    p.add_argument("--max-actions", type=int, default=200)
    p.add_argument("--max-objects", type=int, default=100)
    p.add_argument("--out-dir", default="probe_logs/reveal_vs_online")
    return p.parse_args()


def run_episode(scene, args, config_path, snapshot_fn):
    """Replay one episode; track per-instance first-seen step and final state.

    ``snapshot_fn(sensor)`` returns the current ``{inst_id: (task_id,
    world_pos(3,))}`` view of that config's cloud. Returns objects as
    ``{inst_id: (task_id, world_pos, ego_pos, first_step)}`` plus the final
    packed observation and agent pose.
    """
    import numpy as np
    import quaternion
    import habitat
    import pirlnav  # noqa: F401  (registers ObjectNav-v2, sensors, etc.)
    from pirlnav.config import get_config

    cfg = get_config(config_path, opts=None)
    cfg.defrost()
    tc = cfg.TASK_CONFIG
    tc.DATASET.SPLIT = args.split
    tc.DATASET.DATA_PATH = args.data_path
    tc.DATASET.CONTENT_SCENES = [scene]
    tc.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    tc.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS = args.max_objects
    cfg.freeze()

    env = habitat.Env(config=tc)
    try:
        obs = env.reset()
        names = replay_action_names(env.current_episode, args.max_actions)
        if not names:
            raise RuntimeError(f"{scene}: current episode has no replay")
        sensor = env.task.sensor_suite.sensors["ego_object_cloud"]

        first_step = {}

        def record(step):
            snap = snapshot_fn(sensor)
            for inst in snap.keys() - first_step.keys():
                first_step[inst] = step
            return snap

        snap = record(0)
        for step, name in enumerate(names, start=1):
            obs = env.step({"action": name})
            snap = record(step)
            if env.episode_over:
                break

        state = env.sim.get_agent_state()
        R = quaternion.as_rotation_matrix(state.rotation)
        agent_pos = np.asarray(state.position, dtype=np.float64)
        objects = {
            inst: (tid, pos, ((pos - agent_pos) @ R).astype(np.float32),
                   first_step[inst])
            for inst, (tid, pos) in snap.items()
        }
        result = {
            "objects": objects,
            "packed": np.asarray(obs["ego_object_cloud"]).copy(),
            "episode_id": env.current_episode.episode_id,
            "agent_pos": agent_pos,
        }
        if sensor._reveal is not None:
            # Full oracle list (revealed or not) for only-online diagnostics.
            reveal = sensor._reveal
            result["oracle_all"] = {
                int(inst): (reveal.obj_pos[i].copy(), float(reveal.radii[i]))
                for i, inst in enumerate(sensor._instance_ids)
            }
        return result
    finally:
        env.close()


def online_snapshot(sensor):
    cloud = sensor._cloud
    return {
        inst: (int(st.task_id), st.centroid.copy())
        for inst, st in cloud._objects.items()
    }


def reveal_snapshot(sensor):
    import numpy as np

    reveal = sensor._reveal
    return {
        int(sensor._instance_ids[i]): (
            int(reveal.task_ids[i]), reveal.obj_pos[i].copy()
        )
        for i in np.nonzero(reveal.revealed)[0]
    }


def _render_overlay(on_objs, rv_objs, side_px=480, window_m=12.0):
    """Top-down agent-frame overlay: online = circle, reveal = cross, matched
    pairs joined by a line. Colors follow the goal-category palette."""
    import cv2
    import numpy as np
    from pirlnav.task.semantic_map import PALETTE

    canvas = np.full((side_px, side_px, 3), 30, dtype=np.uint8)
    res = window_m / side_px
    center = side_px // 2

    def to_px(ego):
        return (int(round(ego[0] / res + center)),
                int(round(ego[2] / res + center)))

    def in_bounds(pt):
        return 0 <= pt[0] < side_px and 0 <= pt[1] < side_px

    for inst in sorted(set(on_objs) | set(rv_objs)):
        on = on_objs.get(inst)
        rv = rv_objs.get(inst)
        tid = (on or rv)[0]
        color = tuple(int(c) for c in PALETTE[tid + 2])
        p_on = to_px(on[2]) if on else None
        p_rv = to_px(rv[2]) if rv else None
        if p_on and p_rv and in_bounds(p_on) and in_bounds(p_rv):
            cv2.line(canvas, p_on, p_rv, (160, 160, 160), 1, cv2.LINE_AA)
        if p_on and in_bounds(p_on):
            cv2.circle(canvas, p_on, 5, color, 2, cv2.LINE_AA)
        if p_rv and in_bounds(p_rv):
            x, y = p_rv
            cv2.line(canvas, (x - 4, y - 4), (x + 4, y + 4), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (x - 4, y + 4), (x + 4, y - 4), color, 2, cv2.LINE_AA)

    cv2.arrowedLine(canvas, (center, center), (center, center - 30),
                    (255, 60, 60), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(canvas, "o online   x reveal", (8, side_px - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _save_panel(path, online, reveal, overlay):
    import cv2
    import numpy as np
    from pirlnav.task.object_cloud import render_ego_cloud_topdown

    panels = [
        render_ego_cloud_topdown(online["packed"]),
        render_ego_cloud_topdown(reveal["packed"]),
        overlay,
    ]
    labels = [
        f"online (n={len(online['objects'])})",
        f"oracle reveal (n={len(reveal['objects'])})",
        "overlay (o online, x reveal)",
    ]
    panel = np.concatenate(panels, axis=1)
    w = panels[0].shape[1]
    for i, text in enumerate(labels):
        cv2.putText(panel, text, (i * w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, panel)


def compare_scene(scene, args):
    import numpy as np
    from pirlnav.task.object_cloud import TASK_NAMES

    online = run_episode(scene, args, ONLINE_CONFIG, online_snapshot)
    reveal = run_episode(scene, args, REVEAL_CONFIG, reveal_snapshot)
    assert online["episode_id"] == reveal["episode_id"], "episode mismatch"
    pose_delta = float(
        np.linalg.norm(online["agent_pos"] - reveal["agent_pos"])
    )
    if pose_delta > 0.01:
        print(f"  WARNING final agent pose differs by {pose_delta:.3f}m "
              "(physics changed actuation?)")

    on_objs, rv_objs = online["objects"], reveal["objects"]
    matched = sorted(set(on_objs) & set(rv_objs))
    only_on = sorted(set(on_objs) - set(rv_objs))
    only_rv = sorted(set(rv_objs) - set(on_objs))

    by_cat = lambda ids, objs: Counter(TASK_NAMES[objs[k][0]] for k in ids)
    fmt = lambda c: ", ".join(f"{k}:{v}" for k, v in sorted(c.items())) or "none"

    print(f"[compare] {scene} episode={online['episode_id']}")
    print(f"  objects: online={len(on_objs)} reveal={len(rv_objs)} "
          f"matched={len(matched)}")
    print(f"  only-online ({len(only_on)}): {fmt(by_cat(only_on, on_objs))}")
    print(f"  only-reveal ({len(only_rv)}): {fmt(by_cat(only_rv, rv_objs))}")
    # Diagnose each miss: a large gap between the annotation AABB center and
    # the online (pixel-observed) centroid means the annotation box is bad
    # and the rays aim at the wrong place -- a data wart, not a gate bug.
    oracle_all = reveal.get("oracle_all", {})
    for k in only_on:
        tid, centroid = on_objs[k][0], on_objs[k][1]
        if k in oracle_all:
            gt_pos, gt_r = oracle_all[k]
            gap = float(np.linalg.norm(gt_pos - centroid))
            print(f"    miss {TASK_NAMES[tid]:<14s} inst={k:<5d} "
                  f"aabb_center_vs_online={gap:5.2f}m radius={gt_r:.2f}m")
        else:
            print(f"    miss {TASK_NAMES[tid]:<14s} inst={k:<5d} "
                  "NOT IN ORACLE LIST (annotation filter mismatch!)")
    if matched:
        timing = np.array(
            [rv_objs[k][3] - on_objs[k][3] for k in matched], dtype=np.float64
        )
        centroid = np.array(
            [np.linalg.norm(rv_objs[k][1] - on_objs[k][1]) for k in matched]
        )
        print(f"  reveal-timing delta (steps, neg = oracle earlier): "
              f"mean={timing.mean():+.1f} median={np.median(timing):+.1f} "
              f"p90={np.percentile(np.abs(timing), 90):.1f}")
        print(f"  centroid delta GT-vs-online (m): mean={centroid.mean():.3f} "
              f"median={np.median(centroid):.3f} max={centroid.max():.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{scene}_reveal_vs_online.png")
    overlay = _render_overlay(
        {k: v[:3] for k, v in on_objs.items()},
        {k: v[:3] for k, v in rv_objs.items()},
    )
    _save_panel(out_path, online, reveal, overlay)
    print(f"  saved {out_path}")

    recall = len(matched) / max(len(on_objs), 1)
    passed = recall >= 0.9
    print(f"  {'PASS' if passed else 'FAIL'} recall-of-online={recall:.2%}")
    return passed


def main():
    args = parse_args()
    try:
        scenes = args.scenes or discover_scenes(
            args.data_path, args.split, args.num_scenes
        )
        scenes = [
            s for s in scenes
            if content_has_replay(os.path.join(
                os.path.dirname(args.data_path.format(split=args.split)),
                "content", f"{s}.json.gz",
            ))
        ]
    except Exception as exc:
        print(f"[compare] SETUP_FAIL scene discovery: {exc}")
        return 1

    print(f"[compare] {len(scenes)} scene(s): {', '.join(scenes)} | "
          f"{args.max_actions} replay actions, MAX_OBJECTS={args.max_objects}")
    results = {}
    for scene in scenes:
        try:
            results[scene] = compare_scene(scene, args)
        except Exception:
            print(f"[compare] {scene}: ERROR")
            traceback.print_exc()
            results[scene] = False

    n_pass = sum(results.values())
    print(f"[compare] summary: {n_pass}/{len(results)} scenes passed")
    return 0 if results and n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
