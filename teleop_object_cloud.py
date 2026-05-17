"""Teleoperate in MP3D and incrementally build an in-memory object cloud.

Each tracked object is a world-space ``(x, y, z)`` centroid plus an ObjectNav
task id (``0..20``), with the centroid refined across views weighted by the
object's visible mask area.

Per step we render a 4-panel view ``[RGB | depth | semantic | top-down cloud]``
to ``frame.png`` and accumulate frames into ``teleop_object_cloud.mp4`` on
quit. The top-down panel is *world-aligned* (north = -Z = up), agent-centered
with a sliding viewport, and draws a red arrow showing the agent's heading.

Controls: w forward, a left, d right, f stop, q quit.

Usage:
    python teleop_object_cloud.py --scene-id 17DRP5sb8fy
"""

from __future__ import annotations

import argparse
import sys
import termios
import tty
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import quaternion  # noqa: F401  (registers np.quaternion dtype)

import habitat

import pirlnav  # noqa: F401  (registers ObjectNav-v2 task + dataset)
from pirlnav.config import get_task_config
from pirlnav.task.object_cloud import (
    ObjectCloud,
    make_camera_intrinsics,
)
from pirlnav.task.semantic_map import (
    OCCUPIED,
    PALETTE,
    build_instance_to_task_id,
)


# ---------------------------------------------------------------------------
# Environment setup (mirrors teleop_semantic_map.py exactly)
# ---------------------------------------------------------------------------

def build_env(
    config_path: str, scene_id: str, episode_index: int = 0
) -> Tuple[habitat.Env, dict]:
    """Build an Env on the ``episode_index``-th episode whose scene matches."""
    config = get_task_config(config_paths=config_path)
    config.defrost()
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    # Pixel-align semantic with RGB/depth so masks index depth correctly.
    config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = config.SIMULATOR.RGB_SENSOR.WIDTH
    config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = config.SIMULATOR.RGB_SENSOR.HEIGHT
    config.SIMULATOR.SEMANTIC_SENSOR.HFOV = config.SIMULATOR.RGB_SENSOR.HFOV
    config.SIMULATOR.SEMANTIC_SENSOR.POSITION = config.SIMULATOR.RGB_SENSOR.POSITION
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 10 ** 9
    config.freeze()
    env = habitat.Env(config=config)

    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0 (got {episode_index})")
    n_eps = len(env.episodes)
    matches_seen = 0
    for _ in range(n_eps):
        observations = env.reset()
        if scene_id in env.current_episode.scene_id:
            if matches_seen == episode_index:
                return env, observations
            matches_seen += 1
    raise RuntimeError(
        f"Wanted match #{episode_index} for scene '{scene_id}', but only "
        f"{matches_seen} matching episode(s) found in {n_eps} dataset episodes."
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def depth_to_rgb(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Visualise metric depth via TURBO."""
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    norm = np.clip((depth_m - min_depth) / max(max_depth - min_depth, 1e-6), 0.0, 1.0)
    bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def semantic_to_rgb(semantic: np.ndarray, instance_to_task: np.ndarray) -> np.ndarray:
    """Color each pixel by its task class (0..20), or OCCUPIED for the rest."""
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    inst = np.clip(semantic.astype(np.int64), 0, len(instance_to_task) - 1)
    task = instance_to_task[inst]
    channel = np.where(task >= 0, task + 2, OCCUPIED).astype(np.int8)
    return PALETTE[channel]


def render_topdown(
    cloud: Dict[str, np.ndarray],
    agent_pos: np.ndarray,
    agent_rot: "np.quaternion",
    side_px: int = 480,
    window_m: float = 12.0,
) -> np.ndarray:
    """Agent-centered, world-aligned top-down view (north = -Z = up).

    The canvas does NOT rotate with the agent; only the viewport slides so
    the agent stays at the center. Object centroids are drawn as filled
    circles using the same goal-class colours as the semantic panel.
    """
    canvas = np.full((side_px, side_px, 3), 30, dtype=np.uint8)
    res = window_m / side_px

    def world_to_px(x: float, z: float) -> Tuple[int, int]:
        # Image rows grow downward; we want world -Z (forward) to map to a
        # smaller row so "north = up" on the canvas (matches the docstring).
        col = int(round((x - float(agent_pos[0])) / res + side_px / 2))
        row = int(round((z - float(agent_pos[2])) / res + side_px / 2))
        return col, row

    half = int(window_m / 2)
    for k in range(-half, half + 1):
        gx, _ = world_to_px(float(agent_pos[0]) + k, float(agent_pos[2]))
        cv2.line(canvas, (gx, 0), (gx, side_px), (50, 50, 50), 1)
        _, gz = world_to_px(float(agent_pos[0]), float(agent_pos[2]) + k)
        cv2.line(canvas, (0, gz), (side_px, gz), (50, 50, 50), 1)

    for pos, tid, lbl in zip(cloud["obj_pos"], cloud["task_ids"], cloud["labels"]):
        px, py = world_to_px(float(pos[0]), float(pos[2]))
        if not (0 <= px < side_px and 0 <= py < side_px):
            continue
        # Goal-class colours live at PALETTE rows 2..22 (rows 0/1 are FREE/OCCUPIED).
        color = tuple(int(c) for c in PALETTE[int(tid) + 2])
        cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (px + 8, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)

    cx, cy = side_px // 2, side_px // 2
    R = quaternion.as_rotation_matrix(agent_rot)
    fwd = R @ np.array([0.0, 0.0, -1.0])
    fwd_xz = np.hypot(fwd[0], fwd[2]) + 1e-9
    arrow_len = 30
    ax = int(round(fwd[0] / fwd_xz * arrow_len))
    ay = int(round(fwd[2] / fwd_xz * arrow_len))
    cv2.arrowedLine(canvas, (cx, cy), (cx + ax, cy + ay),
                    (255, 60, 60), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(canvas, (cx, cy), 5, (255, 60, 60), -1, cv2.LINE_AA)
    cv2.putText(canvas, "world", (10, side_px - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def render_topdown_ego(
    ego: Dict[str, np.ndarray],
    side_px: int = 480,
    window_m: float = 12.0,
) -> np.ndarray:
    """Agent-rotated top-down: agent at center, agent's forward = canvas up.

    Plots the agent-frame centroids straight from ``ObjectCloud.to_ego_dict``,
    so this panel is exactly what ``SimpleObjectCloudEncoder`` ingests.
    """
    canvas = np.full((side_px, side_px, 3), 30, dtype=np.uint8)
    res = window_m / side_px

    def ego_to_px(ex: float, ez: float) -> Tuple[int, int]:
        # Habitat agent frame: -Z is forward. Map ego_z=-1 (front) to a smaller
        # row so "forward = up" on the canvas; ego_x=+1 maps to the right.
        col = int(round(ex / res + side_px / 2))
        row = int(round(ez / res + side_px / 2))
        return col, row

    half = int(window_m / 2)
    for k in range(-half, half + 1):
        gx, _ = ego_to_px(float(k), 0.0)
        cv2.line(canvas, (gx, 0), (gx, side_px), (50, 50, 50), 1)
        _, gz = ego_to_px(0.0, float(k))
        cv2.line(canvas, (0, gz), (side_px, gz), (50, 50, 50), 1)

    for pos, tid, lbl in zip(ego["obj_pos"], ego["task_ids"], ego["labels"]):
        px, py = ego_to_px(float(pos[0]), float(pos[2]))
        if not (0 <= px < side_px and 0 <= py < side_px):
            continue
        color = tuple(int(c) for c in PALETTE[int(tid) + 2])
        cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (px + 8, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)

    cx, cy = side_px // 2, side_px // 2
    arrow_len = 30
    cv2.arrowedLine(canvas, (cx, cy), (cx, cy - arrow_len),
                    (255, 60, 60), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(canvas, (cx, cy), 5, (255, 60, 60), -1, cv2.LINE_AA)
    cv2.putText(canvas, "ego (encoder view)", (10, side_px - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def compose_frame(
    rgb: np.ndarray,
    depth_rgb: np.ndarray,
    semantic_rgb: np.ndarray,
    topdown_rgb: np.ndarray,
    topdown_ego_rgb: np.ndarray,
    step_idx: int,
    n_objects: int,
    goal_label: str,
    out_h: int = 480,
) -> np.ndarray:
    """Tile ``[RGB | depth | semantic | top-down (world) | top-down (ego)]``."""
    def fit_h(img: np.ndarray, nearest: bool = False) -> np.ndarray:
        w = max(1, int(round(img.shape[1] * out_h / img.shape[0])))
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        return cv2.resize(img, (w, out_h), interpolation=interp)

    rgb_p = fit_h(rgb)
    dep_p = fit_h(depth_rgb)
    sem_p = fit_h(semantic_rgb, nearest=True)
    td_p = cv2.resize(topdown_rgb, (out_h, out_h), interpolation=cv2.INTER_NEAREST)
    td_ego_p = cv2.resize(topdown_ego_rgb, (out_h, out_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.concatenate([rgb_p, dep_p, sem_p, td_p, td_ego_p], axis=1)
    cv2.putText(canvas,
                f"step {step_idx}  |  objects {n_objects}  |  goal: {goal_label}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Teleop loop
# ---------------------------------------------------------------------------

KEY_TO_ACTION: Dict[str, str] = {
    "w": "MOVE_FORWARD",
    "a": "TURN_LEFT",
    "d": "TURN_RIGHT",
    "f": "STOP",
}


def read_single_key() -> str:
    """Single keystroke from stdin in cbreak mode (no Enter required)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


def save_cloud(cloud: ObjectCloud, path: Path, scene_id: str) -> None:
    """Persist the world-frame object cloud as a compressed NumPy archive.

    Keys
    ----
    obj_pos   : (N, 3) float32  world-frame centroids (x, y, z)
    task_ids  : (N,)   int64    ObjectNav task class id (0..20)
    labels    : (N,)   str      human-readable class names
    scene_id  : ()     str      habitat scene identifier
    """
    d = cloud.to_dict()
    if d["n_objects"] == 0:
        print("Object cloud is empty; skipping save.")
        return
    np.savez_compressed(
        str(path),
        obj_pos=d["obj_pos"],
        task_ids=d["task_ids"],
        labels=np.array(d["labels"], dtype=object),
        scene_id=np.array(scene_id),
    )
    print(f"Saved object cloud ({d['n_objects']} objects) to {path}")


def save_video(frames: List[np.ndarray], path: Path, fps: int = 5) -> None:
    if not frames:
        print("No frames to write; skipping video.")
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"Failed to open video writer for {path}")
        return
    try:
        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(f"Saved video ({len(frames)} frames) to {path}")


def _normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


def run_episode(
    env: habitat.Env,
    observations: dict,
    output_dir: Path,
    video_fps: int = 5,
    min_mask_pixels: int = 100,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = env.current_episode
    goal_label = getattr(episode, "object_category", "unknown") or "unknown"
    instance_to_task = build_instance_to_task_id(env.sim)
    print(f"Scene  : {episode.scene_id}")
    print(f"Goal   : {goal_label}")
    print(f"Loaded scene with {int((instance_to_task >= 0).sum())} goal-class instances.")

    sim_cfg = env._config.SIMULATOR
    rgb_cfg = sim_cfg.RGB_SENSOR
    depth_cfg = sim_cfg.DEPTH_SENSOR
    fx, fy, cx, cy = make_camera_intrinsics(
        int(rgb_cfg.WIDTH), int(rgb_cfg.HEIGHT), float(rgb_cfg.HFOV)
    )
    depth_min, depth_max = float(depth_cfg.MIN_DEPTH), float(depth_cfg.MAX_DEPTH)

    cloud = ObjectCloud(instance_to_task, min_mask_pixels=min_mask_pixels)

    video_path = output_dir / "teleop_object_cloud.mp4"
    frame_path = output_dir / "frame.png"
    recorded: List[np.ndarray] = []
    step_idx = 0
    try:
        while not env.episode_over:
            rgb = _normalise_rgb(observations["rgb"])
            depth = np.asarray(observations["depth"])
            semantic = np.asarray(observations["semantic"])

            agent_state = env.sim.get_agent_state()
            sensor_state = agent_state.sensor_states["depth"]
            new_ids, upd_ids = cloud.update(
                depth_m=depth,
                semantic=semantic,
                sensor_pos=np.asarray(sensor_state.position, dtype=np.float64),
                sensor_rot=sensor_state.rotation,
                fx=fx, fy=fy, cx=cx, cy=cy,
                depth_min=depth_min, depth_max=depth_max,
            )
            cloud_dict = cloud.to_dict()
            ego_dict = cloud.to_ego_dict(
                agent_pos=np.asarray(agent_state.position, dtype=np.float64),
                agent_rot=agent_state.rotation,
            )

            frame = compose_frame(
                rgb=rgb,
                depth_rgb=depth_to_rgb(depth, depth_min, depth_max),
                semantic_rgb=semantic_to_rgb(semantic, instance_to_task),
                topdown_rgb=render_topdown(
                    cloud_dict,
                    agent_pos=np.asarray(agent_state.position),
                    agent_rot=agent_state.rotation,
                ),
                topdown_ego_rgb=render_topdown_ego(ego_dict),
                step_idx=step_idx,
                n_objects=cloud_dict["n_objects"],
                goal_label=goal_label,
                out_h=max(rgb.shape[0], 320),
            )
            recorded.append(frame)
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if ego_dict["n_objects"] > 0:
                d = np.linalg.norm(ego_dict["obj_pos"], axis=1)
                j = int(d.argmin())
                ex, ey, ez = ego_dict["obj_pos"][j]
                nearest = (
                    f"nearest='{ego_dict['labels'][j]}' "
                    f"ego=({ex:+.2f},{ey:+.2f},{ez:+.2f})"
                )
            else:
                nearest = "nearest=-"
            print(
                f"step {step_idx:3d} | objects {cloud_dict['n_objects']:3d} | "
                f"+{len(new_ids)} new ~{len(upd_ids)} upd | {nearest} | "
                f"[w]fwd [a]left [d]right [f]stop [q]quit: ",
                end="", flush=True,
            )
            key = read_single_key().lower()
            print(key)

            if key == "q":
                print("Quit requested.")
                break
            if key not in KEY_TO_ACTION:
                continue
            observations = env.step(KEY_TO_ACTION[key])
            step_idx += 1
    finally:
        save_video(recorded, video_path, fps=video_fps)
        scene_name = Path(episode.scene_id).stem
        save_cloud(cloud, output_dir / f"{scene_name}.npz", str(episode.scene_id))
    print(f"Finished after {step_idx} step(s). Last frame at {frame_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="configs/tasks/objectnav_mp3d.yaml")
    parser.add_argument("--scene-id", default="17DRP5sb8fy",
                        help="Substring of episode.scene_id to match.")
    parser.add_argument("--episode-index", type=int, default=0,
                        help="Pick the N-th episode whose scene_id matches "
                             "--scene-id (0 = first match).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("teleop_runs/object_cloud"))
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--min-mask-pixels", type=int, default=100,
                        help="Skip objects whose visible mask is smaller than this.")
    args = parser.parse_args()

    env, observations = build_env(
        args.config_path, args.scene_id, episode_index=args.episode_index
    )
    try:
        run_episode(
            env, observations,
            output_dir=args.output_dir,
            video_fps=args.video_fps,
            min_mask_pixels=args.min_mask_pixels,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
