"""Teleoperate in MP3D and build an egocentric semantic + occupancy map.

Map shape (H, W, 23):
    0      free
    1      occupied / unknown class
    2..22  the 21 MP3D ObjectNav goal classes (chair, table, picture, ...)

The map is agent-centric and agent-oriented (agent at center, facing up).
Internally we keep a larger world-anchored label map and rotate+crop it each
step. Cells follow "latest wins", except a cell already labeled with a goal
class is only allowed to flip to FREE or to another goal class -- never
back to the generic OCCUPIED.

Controls: w forward, a left, d right, f stop, q quit.

Usage:
    python teleop_semantic_map.py --scene-id 17DRP5sb8fy
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
from pirlnav.task.semantic_map import (
    FREE,
    NAME_TO_TASK,
    NUM_CATEGORIES,
    NUM_CHANNELS,
    OBJECTNAV_CATEGORIES,
    OCCUPIED,
    PALETTE,
    SemanticMapper,
    UNKNOWN,
    build_instance_to_task_id,
    label_map_to_rgb,
    load_class_embeddings,
    smooth_label_map,
)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def build_env(
    config_path: str, scene_id: str, episode_index: int = 0
) -> Tuple[habitat.Env, dict]:
    """Build an Env on the `episode_index`-th episode whose scene matches `scene_id`."""
    config = get_task_config(config_paths=config_path)
    config.defrost()
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    # Pixel-align semantic with RGB/depth.
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


def value_map_from_labels(
    label_map: np.ndarray,
    goal_task_id: int,
    class_embeddings: np.ndarray,
) -> np.ndarray:
    """(H, W) labels -> (H, W) float32 cosine similarity to the goal class.

    `class_embeddings` is the L2-normalized (NUM_CATEGORIES, D) matrix from
    `load_class_embeddings`. Cells whose label is not a goal class
    (UNKNOWN/FREE/OCCUPIED) get value 0.
    """
    sims = class_embeddings @ class_embeddings[goal_task_id]  # (NUM_CATEGORIES,)
    out = np.zeros(label_map.shape, dtype=np.float32)
    is_goal = label_map >= 2
    out[is_goal] = sims[label_map[is_goal] - 2]
    return out


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def semantic_to_rgb(semantic: np.ndarray, instance_to_task: np.ndarray) -> np.ndarray:
    """Color each pixel by its task class (0..20), or OCCUPIED for the rest."""
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    inst = np.clip(semantic.astype(np.int64), 0, len(instance_to_task) - 1)
    task = instance_to_task[inst]
    channel = np.where(task >= 0, task + 2, OCCUPIED).astype(np.int8)
    return PALETTE[channel]


def depth_to_rgb(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Visualise metric depth via TURBO."""
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    norm = np.clip((depth_m - min_depth) / max(max_depth - min_depth, 1e-6), 0.0, 1.0)
    bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def value_map_to_rgb(
    value_map: np.ndarray,
    label_map: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    unknown_color=(20, 20, 20),
) -> np.ndarray:
    """TURBO heatmap of the value map over `[vmin, vmax]`.

    Cells whose label is not a goal class (i.e. `label_map < 2`) are painted
    `unknown_color` so that "no information" is visually distinct from a
    genuine low cosine-sim cell.
    """
    norm = np.clip((value_map - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[label_map < 2] = unknown_color
    return rgb


def compose_frame(
    rgb: np.ndarray,
    depth_rgb: np.ndarray,
    semantic_rgb: np.ndarray,
    map_rgb: np.ndarray,
    value_rgb: np.ndarray,
    step_idx: int,
    goal_label: str,
    out_h: int = 480,
) -> np.ndarray:
    """Tile [RGB | depth | semantic | map | value] at height `out_h`."""
    def fit_h(img: np.ndarray, nearest: bool = False) -> np.ndarray:
        w = max(1, int(round(img.shape[1] * out_h / img.shape[0])))
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        return cv2.resize(img, (w, out_h), interpolation=interp)

    rgb_p = fit_h(rgb)
    dep_p = fit_h(depth_rgb)
    sem_p = fit_h(semantic_rgb, nearest=True)
    mp_p = cv2.resize(map_rgb, (out_h, out_h), interpolation=cv2.INTER_NEAREST)
    val_p = cv2.resize(value_rgb, (out_h, out_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.concatenate([rgb_p, dep_p, sem_p, mp_p, val_p], axis=1)
    cv2.putText(canvas, f"step {step_idx}  |  goal: {goal_label}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Agent markers at the centers of both the map and value panels.
    map_cx = rgb_p.shape[1] + dep_p.shape[1] + sem_p.shape[1] + out_h // 2
    val_cx = map_cx + out_h
    cy = out_h // 2
    for cx in (map_cx, val_cx):
        cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy - 12), (255, 255, 255), 2, cv2.LINE_AA)
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


def save_map(mapper: "SemanticMapper", path: Path, scene_id: str) -> None:
    """Persist the world-frame label map as a compressed NumPy archive.

    Only the world-frame map and its anchoring metadata are stored; the
    egocentric crop dimensions belong to the training config (and to
    ``SemanticMapSensor``), not to the cached map itself. At reload time the
    consumer should derive ``H_g, W_g`` from ``global_map.shape``.

    Keys
    ----
    global_map : (H_g, W_g)  int8    world-anchored label map (-1 UNKNOWN … 22)
    origin_x   : ()          float64 world-X coordinate of the map origin
    origin_z   : ()          float64 world-Z coordinate of the map origin
    resolution : ()          float64 metres per cell
    scene_id   : ()          str     habitat scene identifier
    """
    known = int((mapper.global_map >= 0).sum())
    if known == 0:
        print("Map is fully unknown; skipping save.")
        return
    np.savez_compressed(
        str(path),
        global_map=mapper.global_map,
        origin_x=np.float64(mapper.origin_x),
        origin_z=np.float64(mapper.origin_z),
        resolution=np.float64(mapper.resolution),
        scene_id=np.array(scene_id),
    )
    goal_cells = int((mapper.global_map >= 2).sum())
    print(f"Saved map ({known} known cells, {goal_cells} goal-class cells) to {path}")


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
    H: int,
    W: int,
    resolution: float,
    output_dir: Path,
    tasks_json: Path,
    video_fps: int = 5,
    smooth_k: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = env.current_episode
    goal_label = getattr(episode, "object_category", "unknown") or "unknown"
    instance_to_task = build_instance_to_task_id(env.sim)
    class_embeddings = load_class_embeddings(tasks_json)
    goal_task_id = NAME_TO_TASK.get(goal_label, -1)
    if goal_task_id < 0:
        print(f"WARNING: goal '{goal_label}' is not in the 21 ObjectNav classes; "
              "value map will be all-gray.")
    print(f"Scene  : {episode.scene_id}")
    print(f"Goal   : {goal_label} (task id {goal_task_id})")
    print(f"Loaded scene with {int((instance_to_task >= 0).sum())} goal-class instances.")

    mapper = SemanticMapper(
        H=H, W=W, resolution=resolution,
        start_pos=np.asarray(env.sim.get_agent_state().position, dtype=np.float64),
    )

    sim_cfg = env._config.SIMULATOR
    hfov_rad = np.deg2rad(float(sim_cfg.DEPTH_SENSOR.HFOV))
    min_depth = float(sim_cfg.DEPTH_SENSOR.MIN_DEPTH)
    max_depth = float(sim_cfg.DEPTH_SENSOR.MAX_DEPTH)

    video_path = output_dir / "teleop_semantic_map.mp4"
    frame_path = output_dir / "map_frame.png"
    recorded: List[np.ndarray] = []
    step_idx = 0
    try:
        while not env.episode_over:
            rgb = _normalise_rgb(observations["rgb"])
            depth = np.asarray(observations["depth"])
            semantic = np.asarray(observations["semantic"])

            agent_state = env.sim.get_agent_state()
            sensor_state = agent_state.sensor_states["depth"]
            mapper.update(
                depth_m=depth,
                semantic=semantic,
                sensor_pos=np.asarray(sensor_state.position, dtype=np.float64),
                sensor_rot=sensor_state.rotation,
                agent_y=float(agent_state.position[1]),
                hfov_rad=hfov_rad,
                instance_to_task=instance_to_task,
                min_depth=min_depth,
                max_depth=max_depth,
            )

            label_map = mapper.egocentric_view(
                agent_pos=np.asarray(agent_state.position, dtype=np.float64),
                agent_rot=agent_state.rotation,
            )
            if smooth_k > 1:
                label_map = smooth_label_map(label_map, smooth_k)

            if goal_task_id >= 0:
                value_map = value_map_from_labels(
                    label_map, goal_task_id, class_embeddings
                )
            else:
                value_map = np.zeros(label_map.shape, dtype=np.float32)

            frame = compose_frame(
                rgb=rgb,
                depth_rgb=depth_to_rgb(depth, min_depth, max_depth),
                semantic_rgb=semantic_to_rgb(semantic, instance_to_task),
                map_rgb=label_map_to_rgb(label_map),
                value_rgb=value_map_to_rgb(value_map, label_map),
                step_idx=step_idx,
                goal_label=goal_label,
                out_h=max(rgb.shape[0], 320),
            )
            recorded.append(frame)
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            n_goal = int((label_map >= 2).sum())
            v_max = float(value_map.max()) if n_goal else 0.0
            print(
                f"step {step_idx:3d} | known {int((label_map >= 0).sum()):5d} | "
                f"goal {n_goal:4d} | value max {v_max:+.3f} | "
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
        save_map(mapper, output_dir / f"{scene_name}.npz", str(episode.scene_id))
    print(f"Finished after {step_idx} step(s). Last map_frame.png at {frame_path}")


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
                             "--scene-id (0 = first match; episodes typically "
                             "differ in start pose + goal object).")
    parser.add_argument("--map-h", type=int, default=256)
    parser.add_argument("--map-w", type=int, default=256)
    parser.add_argument("--map-res", type=float, default=0.025,
                        help="Meters per map cell.")
    parser.add_argument("--output-dir", type=Path, default=Path("teleop_runs/semantic_map"))
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--smooth-k", type=int, default=4,
                        help="Mode-filter window size for the egocentric view "
                             "(0/1 disables; only affects viz, not the world map).")
    parser.add_argument("--tasks-json", type=Path, default=Path("tasks.json"),
                        help="Path to tasks.json with per-class CLIP text_embedding.")
    args = parser.parse_args()

    env, observations = build_env(
        args.config_path, args.scene_id, episode_index=args.episode_index
    )
    try:
        run_episode(
            env, observations,
            H=args.map_h, W=args.map_w, resolution=args.map_res,
            output_dir=args.output_dir,
            tasks_json=args.tasks_json,
            video_fps=args.video_fps, smooth_k=args.smooth_k,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
