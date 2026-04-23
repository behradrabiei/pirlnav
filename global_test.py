"""
Teleoperation script for ObjectNav with a 12-bin goal-direction compass feature.

At every timestep, computes a 12-D vector where each bin corresponds to a
heading (0, 30, ..., 330 degrees) measured counterclockwise from the agent's
forward direction. For each goal in the episode, the bin scores are:

    score_i += max(0, cos(bin_angle_i - bearing_to_goal)) / (1 + distance)

summed across all goal instances. Scores are rectified so goals do not
actively penalize bins pointing away from them.

Controls:
    w / up    : move forward
    a / left  : turn left
    d / right : turn right
    f         : finish / stop episode
    q / esc   : quit

Usage:
    python objectnav_teleop_compass.py \
        --config-path benchmark/nav/objectnav/objectnav_hm3d.yaml \
        --episode-index 0 \
        --output-dir ./teleop_runs
"""

from __future__ import annotations

import argparse
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, cast

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import quaternion  # noqa: F401  (registers np.quaternion dtype)

import habitat
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.utils.visualizations import maps

from pirlnav.task.object_nav_task import ObjectGoalNavEpisode
import pirlnav  # noqa: F401  (registers ObjectNav-v2 task + dataset)
from pirlnav.config import get_task_config


# ---------------------------------------------------------------------------
# Compass feature
# ---------------------------------------------------------------------------

N_BINS = 12
BIN_ANGLES_RAD = np.arange(N_BINS) * (2 * np.pi / N_BINS)  # 0, 30, ..., 330 deg
FORWARD_LOCAL = np.array([0.0, 0.0, -1.0])  # Habitat convention: -z is forward


def compute_compass_feature(
    agent_position: np.ndarray,
    agent_rotation: np.quaternion,
    goal_positions: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Compute the 12-bin compass feature vector for the current agent pose.

    Args:
        agent_position: (3,) agent position in world frame [x, y, z].
        agent_rotation: agent rotation quaternion (np.quaternion).
        goal_positions: (M, 3) goal centroids in world frame.

    Returns:
        compass: (12,) rectified, distance-weighted cosine-similarity scores.
        per_goal_info: list of (bearing_rad, distance) for logging/visualization.
    """
    # Agent forward in world frame, projected onto the x-z ground plane.
    R = quaternion.as_rotation_matrix(agent_rotation)
    fwd_world = R @ FORWARD_LOCAL
    fwd_xz = np.array([fwd_world[0], fwd_world[2]])
    fwd_norm = np.linalg.norm(fwd_xz)
    if fwd_norm < 1e-9:
        # Agent is looking straight up or down; no meaningful ground heading.
        return np.zeros(N_BINS, dtype=np.float32), []
    fwd_xz /= fwd_norm

    compass = np.zeros(N_BINS, dtype=np.float32)
    per_goal_info: List[Tuple[float, float]] = []

    for goal_pos in goal_positions:
        delta = goal_pos - agent_position
        delta_xz = np.array([delta[0], delta[2]])
        d = float(np.linalg.norm(delta_xz))
        if d < 1e-6:
            continue
        goal_dir = delta_xz / d

        # Signed angle from agent-forward to goal-direction in the x-z plane.
        # Positive = counterclockwise about +y viewed from above (agent's left).
        # In Habitat, top-down view is looking down the +y axis, and world +z
        # points toward the bottom of that view, so the 2D cross between
        # (world_x, world_z) vectors has the opposite sign from CCW-about-+y.
        # We negate it so `bearing > 0` corresponds to the agent's left.
        cross = fwd_xz[1] * goal_dir[0] - fwd_xz[0] * goal_dir[1]
        dot = fwd_xz[0] * goal_dir[0] + fwd_xz[1] * goal_dir[1]
        bearing = float(np.arctan2(cross, dot))

        sims = np.clip(np.cos(BIN_ANGLES_RAD - bearing), 0.0, None)
        compass += (sims / (1.0 + d)).astype(np.float32)
        per_goal_info.append((bearing, d))

    return compass, per_goal_info


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@dataclass
class FrameBundle:
    rgb: np.ndarray            # (H, W, 3) uint8
    top_down_map: np.ndarray   # (H, W, 3) uint8
    compass: np.ndarray        # (12,) float32
    per_goal_info: List[Tuple[float, float]]
    step_idx: int
    num_goals: int             # total goal instances in the episode
    goal_label: str            # object category name


def render_compass_panel(
    compass: np.ndarray,
    per_goal_info: List[Tuple[float, float]],
    size_px: int = 400,
) -> np.ndarray:
    """Render the 12-bin compass as a polar wedge plot.

    Bin 0 points up (agent forward), positive angles go counterclockwise,
    matching the convention of the compass feature.
    """
    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
    ax = fig.add_subplot(111, projection="polar")

    # Normalize for color mapping; keep true values for bar heights so the
    # relative magnitudes are visible across timesteps.
    max_val = float(compass.max()) if compass.max() > 1e-9 else 1.0
    colors = plt.cm.viridis(compass / max_val)

    # Wedges centered on each bin angle.
    width = 2 * np.pi / N_BINS
    ax.bar(
        BIN_ANGLES_RAD,
        compass,
        width=width,
        bottom=0.0,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        align="center",
    )

    # Mark agent forward (bin 0 direction).
    ax.plot([0, 0], [0, max_val * 1.05], color="red", linewidth=2.0)

    # Overlay raw goal directions as small markers on the outer ring.
    for bearing, dist in per_goal_info:
        ax.plot(
            bearing,
            max_val * 1.1,
            marker="o",
            color="orange",
            markersize=6,
            markeredgecolor="black",
        )

    # Orient so that 0 rad (agent forward) is up, and positive angles go CCW.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_yticklabels([])
    ax.set_xticks(BIN_ANGLES_RAD)
    ax.set_xticklabels([f"{int(np.rad2deg(a))}" for a in BIN_ANGLES_RAD], fontsize=7)
    ax.set_title("Goal compass (12 bins)", fontsize=10, pad=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def compose_frame(bundle: FrameBundle, out_h: int = 480) -> np.ndarray:
    """Tile [RGB | compass | top-down map] into a single image."""
    def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        scale = h / img.shape[0]
        w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    rgb = resize_to_h(bundle.rgb, out_h)
    compass_img = render_compass_panel(bundle.compass, bundle.per_goal_info)
    compass_img = resize_to_h(compass_img, out_h)
    tdm = resize_to_h(bundle.top_down_map, out_h)

    # Annotate step index + goal count + goal label on the top-left.
    canvas = np.concatenate([rgb, compass_img, tdm], axis=1)
    cv2.putText(
        canvas,
        f"step {bundle.step_idx}  |  goal: {bundle.goal_label} ({bundle.num_goals} instance{'s' if bundle.num_goals != 1 else ''})",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def build_env(config_path: str) -> habitat.Env:
    """Create an ObjectNav env with the TopDownMap measure enabled."""
    config = get_task_config(config_paths=config_path)
    config.defrost()
    config.TASK.MEASUREMENTS = list(config.TASK.MEASUREMENTS) + ["TOP_DOWN_MAP"]
    config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 512
    config.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
    config.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
    config.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
    config.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
    config.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
    config.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
    config.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = True
    config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = False
    config.freeze()
    return habitat.Env(config=config)


def extract_goal_centroids(episode: NavigationEpisode) -> np.ndarray:
    """Pull out the (M, 3) array of goal object centroids in world frame."""
    positions = [np.asarray(g.position, dtype=np.float64) for g in episode.goals]
    if len(positions) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(positions, axis=0)


def get_rgb(observations: dict) -> np.ndarray:
    """Find the RGB observation under the usual key names; fall back to black."""
    for key in ("rgb", "robot_head_rgb", "head_rgb"):
        if key in observations:
            img = observations[key]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return img
    return np.zeros((256, 256, 3), dtype=np.uint8)


def get_top_down_map(info: dict, target_h: int = 480) -> np.ndarray:
    """Colorize the TopDownMap measure output. Returns black image if missing."""
    tdm_info = info.get("top_down_map")
    if tdm_info is None:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)
    return maps.colorize_draw_agent_and_fit_to_height(tdm_info, target_h)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

KEY_TO_ACTION = {
    "w": "MOVE_FORWARD",
    "a": "TURN_LEFT",
    "d": "TURN_RIGHT",
    "f": "STOP",
}


def read_single_key() -> str:
    """Read a single keystroke from stdin without waiting for Enter.

    Puts the terminal in cbreak mode for the duration of one read, then
    restores the previous settings.  Ctrl+C is re-raised as KeyboardInterrupt.
    """
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
    if ch == "\x03":  # Ctrl+C
        raise KeyboardInterrupt
    return ch


def run_episode(env: habitat.Env, episode_index: int, output_dir: Path) -> None:
    # Skip to the requested episode.
    for _ in range(episode_index):
        env.reset()
    observations = env.reset()
    episode = cast(ObjectGoalNavEpisode, env.current_episode)

    goal_centroids = extract_goal_centroids(episode)
    goal_label = episode.object_category or "unknown"
    print(f"Episode {episode.episode_id} | scene={episode.scene_id}")
    print(f"  goal: {goal_label} | {len(goal_centroids)} instance(s) | start={episode.start_position}")

    output_dir.mkdir(parents=True, exist_ok=True)

    step_idx = 0
    while not env.episode_over:
        agent_state = env.sim.get_agent_state()
        agent_pos = np.asarray(agent_state.position, dtype=np.float64)
        agent_rot = agent_state.rotation

        compass, per_goal = compute_compass_feature(
            agent_pos, agent_rot, goal_centroids
        )
        info = env.get_metrics()
        tdm = get_top_down_map(info)

        bundle = FrameBundle(
            rgb=get_rgb(observations),
            top_down_map=tdm,
            compass=compass,
            per_goal_info=per_goal,
            step_idx=step_idx,
            num_goals=len(goal_centroids),
            goal_label=goal_label,
        )
        frame = compose_frame(bundle)

        # Save + display.
        out_path = output_dir / "frame.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Print the compass vector for the terminal log.
        formatted = ", ".join(f"{v:5.3f}" for v in compass)
        print(f"step {step_idx:3d} | compass = [{formatted}]")
        print("  frame.png updated | [w]forward  [a]left  [d]right  [f]stop  [q]quit: ",
              end="", flush=True)

        key = read_single_key().lower()
        print(key)  # echo the key since cbreak suppresses it

        if key == "q":
            print("Quit requested.")
            break
        if key not in KEY_TO_ACTION:
            continue

        action = KEY_TO_ACTION[key]
        observations = env.step(action)
        step_idx += 1

    print(f"Finished after {step_idx} step(s). Last frame saved to {output_dir / 'frame.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default="configs/tasks/objectnav_mp3d.yaml",
        help="Habitat task config path (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which episode in the dataset to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("teleop_runs"),
        help="Where to save frame.png (single file, overwritten each step).",
    )
    args = parser.parse_args()

    env = build_env(args.config_path)
    try:
        run_episode(env, args.episode_index, args.output_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main()