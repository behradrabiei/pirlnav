#!/usr/bin/env python3
"""Replay IL training demonstrations and record validation videos.

Each frame tiles RGB, first-person semantic, goal compass, online semantic
BEV, online object cloud, and the habitat top-down map so sensor outputs
can be checked against the expert trajectory. This script is visualization
only and does not affect IL training. By default the top-down, semantic BEV,
object-cloud, and goal-compass panels overlay the next 5 expert poses; use
``--horizon-steps 0`` to disable.

Usage (requires conda env ``pirlnav``)::

    conda activate pirlnav
    python replay_il_training_demos.py --replay-mode actions
    python replay_il_training_demos.py --replay-mode poses
    python replay_il_training_demos.py --replay-mode actions --allow-sliding
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import quaternion  # noqa: F401

import habitat
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

import pirlnav  # noqa: F401
from pirlnav.config import get_config
from pirlnav.task.object_cloud import render_ego_cloud_topdown
from pirlnav.task.semantic_map import (
    OCCUPIED,
    PALETTE,
    build_instance_to_task_id,
    label_map_to_rgb,
)

ACTION_NAME_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_UP": 4,
    "LOOK_DOWN": 5,
}

MODE_SUBDIRS = {
    "actions": "action_replay",
    "poses": "pose_replay",
}


def video_output_dir(out_root: str, replay_mode: str, allow_sliding: bool) -> Path:
    subdir = MODE_SUBDIRS[replay_mode]
    if allow_sliding:
        subdir = f"{subdir}_sliding"
    return Path(out_root) / subdir

DATASET_ROOT = (
    "data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat"
)


def ensure_conda_env() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") == "pirlnav":
        return
    for conda_sh in (
        "/workspace/conda/etc/profile.d/conda.sh",
        os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh"),
        os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh"),
    ):
        if not os.path.isfile(conda_sh):
            continue
        print(
            f"[replay] CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}; "
            "re-exec under pirlnav ..."
        )
        cmd = (
            f"source {conda_sh} && conda activate pirlnav && "
            f"exec python {' '.join(repr(a) for a in sys.argv)}"
        )
        os.execv("/bin/bash", ["bash", "-lc", cmd])
    print(
        "[replay] warning: not in conda env 'pirlnav' and could not auto-activate.",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/experiments/il_objectnav_mp3d.yaml",
    )
    p.add_argument(
        "--replay-mode",
        required=True,
        choices=["actions", "poses"],
    )
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--scene", default="17DRP5sb8fy")
    p.add_argument("--out-root", default="demo_replays")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--episode-ids", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--allow-sliding",
        action="store_true",
        help="Enable ALLOW_SLIDING (default: False, matches IL training).",
    )
    p.add_argument(
        "--success-distance",
        type=float,
        default=1.0,
        help="SUCCESS_DISTANCE in meters (default: 1.0).",
    )
    p.add_argument(
        "--horizon-steps",
        type=int,
        default=5,
        help="Expert lookahead on top-down, semantic map, and object cloud (0=off).",
    )
    return p.parse_args()


HORIZON_COLOR_RGB = (0, 255, 0)
HORIZON_TURN_DIR_COLOR_RGB = (255, 140, 0)
HORIZON_LINE_THICKNESS = 3
HORIZON_ARROW_THICKNESS = 3
HORIZON_TURN_DIR_THICKNESS = 4
EGO_CLOUD_WINDOW_M = 12.0
EGO_HORIZON_MIN_WINDOW_M = 3.0
EGO_HORIZON_FIT_MARGIN = 1.4
EGO_TURN_IN_PLACE_M = 0.22
EGO_TURN_ARC_RADIUS_PX = 44
EGO_TURN_DIR_ARC_RADIUS_PX = 52
EGO_TURN_DIR_MIN_RAD = np.radians(3.0)
EGO_TURN_TICK_LEN_PX = 22
# Default SemanticMapSensor.MAP_RESOLUTION (agent-frame BEV in replay config).
SEMANTIC_MAP_RESOLUTION_M = 0.025
# Goal-compass BEV: how much of the half-side the 100% radial wedge fills.
COMPASS_BEV_MAX_RADIUS_FRAC = 0.40
# Goal-compass BEV: fallback metric window (m) when horizon is missing/empty.
COMPASS_BEV_BASE_WINDOW_M = 6.0


_VIRIDIS_LUT_RGB: Optional[np.ndarray] = None


def _viridis_lut_rgb() -> np.ndarray:
    """256-entry uint8 viridis LUT, matching the eval-video polar chart."""
    global _VIRIDIS_LUT_RGB
    if _VIRIDIS_LUT_RGB is None:
        from matplotlib import cm as mpl_cm

        rgba = mpl_cm.get_cmap("viridis")(np.linspace(0.0, 1.0, 256))
        _VIRIDIS_LUT_RGB = (rgba[..., :3] * 255.0).astype(np.uint8)
    return _VIRIDIS_LUT_RGB


def _viridis_rgb(value01: float) -> Tuple[int, int, int]:
    idx = int(np.clip(value01, 0.0, 1.0) * 255.0 + 0.5)
    r, g, b = _viridis_lut_rgb()[idx]
    return int(r), int(g), int(b)


@dataclass
class ExpertHorizonWaypoint:
    """Future expert pose for replay horizon overlay (viz only)."""

    map_coord: Tuple[int, int]
    map_angle: float
    ego_xz: Tuple[float, float]
    ego_fwd_xz: Tuple[float, float]


def _quaternion_to_topdown_angle(rotation: Sequence[float]) -> float:
    from habitat.tasks.utils import cartesian_to_polar
    from habitat.utils.geometry_utils import quaternion_rotate_vector

    q = np.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
    heading_vector = quaternion_rotate_vector(
        q.inverse(), np.array([0.0, 0.0, -1.0])
    )
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return float(phi + np.pi)


def _rotation_matrix(rotation: Sequence[float]) -> np.ndarray:
    q = np.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
    return quaternion.as_rotation_matrix(q)


def _forward_in_agent_ego(
    rotation: Sequence[float], agent_rot: Sequence[float]
) -> Tuple[float, float]:
    fwd_world = _rotation_matrix(rotation) @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    fwd_ego = fwd_world @ _rotation_matrix(agent_rot)
    return float(fwd_ego[0]), float(fwd_ego[2])


def expert_horizon_from_replay(
    reference_replay: Sequence[Any],
    replay_idx: int,
    agent_pos: Sequence[float],
    agent_rot: Sequence[float],
    horizon_steps: int,
    map_shape: Tuple[int, int],
    pathfinder,
) -> List[ExpertHorizonWaypoint]:
    """Build up to ``horizon_steps`` future expert waypoints (replay viz only)."""
    if horizon_steps <= 0:
        return []

    p0 = np.asarray(agent_pos, dtype=np.float64)
    R0 = _rotation_matrix(agent_rot)
    waypoints: List[ExpertHorizonWaypoint] = []
    for k in range(1, horizon_steps + 1):
        j = replay_idx + k
        if j >= len(reference_replay):
            break
        state = _agent_state_dict(reference_replay[j])
        if state is None:
            continue
        pos = state["position"]
        rot = state["rotation"]
        delta = np.asarray(pos, dtype=np.float64) - p0
        ego = delta @ R0
        ego_xz = (float(ego[0]), float(ego[2]))
        ego_fwd = _forward_in_agent_ego(rot, agent_rot)
        ax, ay = maps.to_grid(
            float(pos[2]),
            float(pos[0]),
            map_shape,
            pathfinder=pathfinder,
        )
        waypoints.append(
            ExpertHorizonWaypoint(
                map_coord=(int(ax), int(ay)),
                map_angle=_quaternion_to_topdown_angle(rot),
                ego_xz=ego_xz,
                ego_fwd_xz=ego_fwd,
            )
        )
    return waypoints


def _ego_horizon_fit_window_m(
    waypoints: Sequence[ExpertHorizonWaypoint],
    base_window_m: float = EGO_CLOUD_WINDOW_M,
) -> float:
    """Zoom object-cloud view to the expert lookahead (avoids huge spokes)."""
    if not waypoints:
        return base_window_m
    extent = max(
        float(np.hypot(wp.ego_xz[0], wp.ego_xz[1])) for wp in waypoints
    )
    if extent < 0.08:
        return EGO_HORIZON_MIN_WINDOW_M
    fit = extent * EGO_HORIZON_FIT_MARGIN + 0.6
    return float(min(base_window_m, max(EGO_HORIZON_MIN_WINDOW_M, fit)))


def _ego_xz_to_px(
    ex: float,
    ez: float,
    *,
    side_px: int,
    window_m: float,
) -> Tuple[int, int]:
    res = window_m / side_px
    center = side_px // 2
    col = int(round(ex / res + center))
    row = int(round(ez / res + center))
    return col, row


def _ego_heading_theta(fwd_xz: Tuple[float, float]) -> float:
    """Heading radians in canvas coords (forward = up, matches object cloud)."""
    fx, fz = fwd_xz
    # Object cloud: +ex -> right, -ez -> up  =>  atan2(fx, -fz)
    return float(np.arctan2(fx, -fz + 1e-9))


def _heading_tick_end(
    origin: Tuple[int, int],
    fwd_xz: Tuple[float, float],
    length_px: int,
) -> Tuple[int, int]:
    th = _ego_heading_theta(fwd_xz)
    col, row = origin
    return (
        int(col + np.sin(th) * length_px),
        int(row - np.cos(th) * length_px),
    )


def _shortest_angle_delta(a0: float, a1: float) -> float:
    return float((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)


def _this_step_yaw_delta(thetas: Sequence[float]) -> float:
    """Rotation about to happen at this timestep: 0 -> theta of next expert pose.

    The current agent heading is 0 in its own ego frame (forward = up), and
    ``thetas[0]`` is the heading of the first future expert pose, so the
    signed shortest angle between them is exactly ``thetas[0]``.
    """
    if not thetas:
        return 0.0
    return _shortest_angle_delta(0.0, thetas[0])


def _ring_point(
    center: Tuple[int, int], radius_px: int, theta: float
) -> Tuple[int, int]:
    cx, cy = center
    return (
        int(cx + np.sin(theta) * radius_px),
        int(cy - np.cos(theta) * radius_px),
    )


def _draw_yaw_arc(
    canvas: np.ndarray,
    center: Tuple[int, int],
    radius_px: int,
    theta_start: float,
    delta_rad: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    *,
    arrowhead: bool = False,
) -> None:
    """Draw a yaw arc on a ring around ``center`` (replay viz only)."""
    if abs(delta_rad) < np.radians(2.0):
        return
    n = max(int(abs(np.degrees(delta_rad)) / 10), 3)
    pts = [
        _ring_point(center, radius_px, theta_start + delta_rad * t)
        for t in np.linspace(0.0, 1.0, n)
    ]
    cv2.polylines(
        canvas,
        [np.asarray(pts, dtype=np.int32)],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )
    if arrowhead and len(pts) >= 2:
        tail = pts[max(0, len(pts) - 3)]
        cv2.arrowedLine(
            canvas,
            tail,
            pts[-1],
            color,
            max(thickness, 2),
            cv2.LINE_AA,
            tipLength=0.55,
        )


def _draw_turn_direction_arc(
    canvas: np.ndarray,
    center: Tuple[int, int],
    thetas: Sequence[float],
    *,
    radius_px: int = EGO_TURN_DIR_ARC_RADIUS_PX,
) -> None:
    """Orange arc + arrowhead for the turn happening at *this* timestep.

    Starts at current heading (0 in agent ego frame, forward = up) and
    extends by the signed yaw to the first future expert pose, so the
    arrowhead points in the immediate turn direction (left or right).
    """
    delta = _this_step_yaw_delta(thetas)
    if abs(delta) < EGO_TURN_DIR_MIN_RAD:
        return
    _draw_yaw_arc(
        canvas,
        center,
        radius_px,
        0.0,
        delta,
        HORIZON_TURN_DIR_COLOR_RGB,
        HORIZON_TURN_DIR_THICKNESS,
        arrowhead=True,
    )


def _horizon_is_turn_in_place(waypoints: Sequence[ExpertHorizonWaypoint]) -> bool:
    if not waypoints:
        return False
    extent = max(np.hypot(wp.ego_xz[0], wp.ego_xz[1]) for wp in waypoints)
    return extent < EGO_TURN_IN_PLACE_M


def _horizon_has_turn_direction(waypoints: Sequence[ExpertHorizonWaypoint]) -> bool:
    if not waypoints:
        return False
    thetas = [_ego_heading_theta(wp.ego_fwd_xz) for wp in waypoints]
    return abs(_this_step_yaw_delta(thetas)) >= EGO_TURN_DIR_MIN_RAD


def _draw_ego_horizon_turn(
    canvas: np.ndarray,
    waypoints: Sequence[ExpertHorizonWaypoint],
    *,
    side_px: int,
) -> None:
    """Turn-in-place: ring, arcs between consecutive steps, equal-length heading ticks."""
    color = HORIZON_COLOR_RGB
    cx, cy = side_px // 2, side_px // 2
    center = (cx, cy)
    thetas = [_ego_heading_theta(wp.ego_fwd_xz) for wp in waypoints]

    cv2.circle(canvas, center, EGO_TURN_ARC_RADIUS_PX, color, 2, cv2.LINE_AA)

    for i in range(len(thetas) - 1):
        delta = _shortest_angle_delta(thetas[i], thetas[i + 1])
        _draw_yaw_arc(
            canvas,
            center,
            EGO_TURN_ARC_RADIUS_PX,
            thetas[i],
            delta,
            color,
            thickness=HORIZON_LINE_THICKNESS,
        )

    for wp in waypoints:
        end = _heading_tick_end(center, wp.ego_fwd_xz, EGO_TURN_TICK_LEN_PX)
        cv2.arrowedLine(
            canvas,
            center,
            end,
            color,
            HORIZON_ARROW_THICKNESS,
            cv2.LINE_AA,
            tipLength=0.35,
        )

    _draw_turn_direction_arc(canvas, center, thetas)


def _draw_ego_horizon_move(
    canvas: np.ndarray,
    waypoints: Sequence[ExpertHorizonWaypoint],
    *,
    side_px: int,
    window_m: float,
) -> None:
    """Moving expert: consecutive segments + ticks; yaw arcs when a step barely moves."""
    color = HORIZON_COLOR_RGB
    tick_len_px = 11
    res = window_m / side_px
    px_pts = [
        _ego_xz_to_px(wp.ego_xz[0], wp.ego_xz[1], side_px=side_px, window_m=window_m)
        for wp in waypoints
    ]
    thetas = [_ego_heading_theta(wp.ego_fwd_xz) for wp in waypoints]

    for i in range(len(waypoints) - 1):
        dist_m = float(
            np.hypot(
                waypoints[i + 1].ego_xz[0] - waypoints[i].ego_xz[0],
                waypoints[i + 1].ego_xz[1] - waypoints[i].ego_xz[1],
            )
        )
        if dist_m < EGO_TURN_IN_PLACE_M:
            mid_ex = 0.5 * (waypoints[i].ego_xz[0] + waypoints[i + 1].ego_xz[0])
            mid_ez = 0.5 * (waypoints[i].ego_xz[1] + waypoints[i + 1].ego_xz[1])
            mid = _ego_xz_to_px(mid_ex, mid_ez, side_px=side_px, window_m=window_m)
            delta = _shortest_angle_delta(thetas[i], thetas[i + 1])
            _draw_yaw_arc(
                canvas,
                mid,
                18,
                thetas[i],
                delta,
                color,
                thickness=HORIZON_LINE_THICKNESS,
            )
            continue
        if np.hypot(px_pts[i + 1][0] - px_pts[i][0], px_pts[i + 1][1] - px_pts[i][1]) >= 3:
            cv2.line(
                canvas,
                px_pts[i],
                px_pts[i + 1],
                color,
                HORIZON_LINE_THICKNESS,
                cv2.LINE_AA,
            )

    for i, (wp, (col, row)) in enumerate(zip(waypoints, px_pts)):
        if not (0 <= col < side_px and 0 <= row < side_px):
            continue
        cv2.circle(canvas, (col, row), 3, color, -1, cv2.LINE_AA)
        end = _heading_tick_end((col, row), wp.ego_fwd_xz, tick_len_px)
        cv2.arrowedLine(
            canvas,
            (col, row),
            end,
            color,
            HORIZON_ARROW_THICKNESS,
            cv2.LINE_AA,
            tipLength=0.4,
        )

    center = (side_px // 2, side_px // 2)
    _draw_turn_direction_arc(canvas, center, thetas)


def _draw_ego_horizon(
    canvas: np.ndarray,
    waypoints: Sequence[ExpertHorizonWaypoint],
    *,
    side_px: int,
    window_m: float,
) -> None:
    if not waypoints:
        return
    if _horizon_is_turn_in_place(waypoints):
        _draw_ego_horizon_turn(canvas, waypoints, side_px=side_px)
    else:
        _draw_ego_horizon_move(canvas, waypoints, side_px=side_px, window_m=window_m)


def _semantic_map_span_m(label_map: np.ndarray) -> float:
    """Metric width of the egocentric semantic map (forward = up)."""
    return float(label_map.shape[1] * SEMANTIC_MAP_RESOLUTION_M)


def render_semantic_map_panel(
    label_map: np.ndarray,
    horizon: Optional[Sequence[ExpertHorizonWaypoint]],
    *,
    side_px: int = 480,
) -> np.ndarray:
    """Semantic BEV with optional expert horizon overlay (replay viz only)."""
    rgb = label_map_to_rgb(label_map)
    canvas = _square_panel(rgb, side_px, nearest=True)
    if not horizon:
        return canvas
    window_m = _ego_horizon_fit_window_m(
        horizon, base_window_m=_semantic_map_span_m(label_map)
    )
    _draw_ego_horizon(canvas, horizon, side_px=side_px, window_m=window_m)
    _draw_horizon_panel_caption(canvas, horizon, side_px)
    return canvas


def _draw_horizon_panel_caption(
    canvas: np.ndarray,
    horizon: Sequence[ExpertHorizonWaypoint],
    side_px: int,
) -> None:
    turn = _horizon_is_turn_in_place(horizon)
    text = f"next 1-{len(horizon)}" + (" turn" if turn else "")
    if _horizon_has_turn_direction(horizon):
        text += " | orange=this step turn"
    cv2.putText(
        canvas,
        text,
        (8, side_px - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def render_ego_cloud_panel(
    packed: np.ndarray,
    horizon: Optional[Sequence[ExpertHorizonWaypoint]],
    *,
    side_px: int = 480,
) -> np.ndarray:
    """Object cloud with optional expert horizon (fixed 12 m window)."""
    canvas = render_ego_cloud_topdown(
        packed, side_px=side_px, window_m=EGO_CLOUD_WINDOW_M
    )
    if not horizon:
        return canvas
    _draw_ego_horizon(
        canvas, horizon, side_px=side_px, window_m=EGO_CLOUD_WINDOW_M
    )
    _draw_horizon_panel_caption(canvas, horizon, side_px)
    return canvas


def render_goal_compass_bev_panel(
    compass: np.ndarray,
    horizon: Optional[Sequence[ExpertHorizonWaypoint]],
    *,
    side_px: int = 480,
) -> np.ndarray:
    """Ego BEV of the 12-bin goal compass with optional expert horizon.

    Forward = up, agent at center (same convention as the semantic-map and
    object-cloud panels). Each bin renders as a filled viridis wedge whose
    radial extent encodes ``compass[i] / max(compass)`` (per-frame
    normalization, matching ``render_goal_compass_panel``). When a horizon is
    supplied it is drawn on top using ``_draw_ego_horizon`` so the three ego
    panels share the same green polyline / heading-tick / orange-turn-arc
    style. Replay visualization only -- does not feed the policy.
    """
    compass = np.asarray(compass, dtype=np.float32).reshape(-1)
    n_bins = int(compass.shape[0])
    max_val = float(compass.max()) if compass.size else 0.0
    norm = max(max_val, 1e-9)

    if horizon:
        window_m = _ego_horizon_fit_window_m(
            horizon, base_window_m=COMPASS_BEV_BASE_WINDOW_M
        )
    else:
        window_m = COMPASS_BEV_BASE_WINDOW_M

    canvas = np.full((side_px, side_px, 3), 30, dtype=np.uint8)
    res = window_m / side_px
    center = side_px // 2

    for k in range(-int(window_m // 2), int(window_m // 2) + 1):
        offset = int(round(k / res + center))
        if 0 <= offset < side_px:
            cv2.line(canvas, (offset, 0), (offset, side_px), (50, 50, 50), 1)
            cv2.line(canvas, (0, offset), (side_px, offset), (50, 50, 50), 1)

    max_radius_px = max(8, int(COMPASS_BEV_MAX_RADIUS_FRAC * side_px))
    cv2.circle(
        canvas, (center, center), max_radius_px, (80, 80, 80), 1, cv2.LINE_AA
    )

    if n_bins > 0:
        half_bin = np.pi / n_bins
        arc_samples = 8
        for i in range(n_bins):
            score01 = float(compass[i]) / norm
            L_i = score01 * max_radius_px
            if L_i < 1.0:
                continue
            theta_center = -i * (2.0 * np.pi / n_bins)
            thetas = np.linspace(
                theta_center - half_bin, theta_center + half_bin, arc_samples
            )
            pts = np.empty((arc_samples + 1, 2), dtype=np.int32)
            pts[0] = (center, center)
            pts[1:, 0] = np.round(center + L_i * np.sin(thetas)).astype(np.int32)
            pts[1:, 1] = np.round(center - L_i * np.cos(thetas)).astype(np.int32)
            cv2.fillPoly(canvas, [pts], _viridis_rgb(score01), cv2.LINE_AA)

    cv2.arrowedLine(
        canvas,
        (center, center),
        (center, center - 30),
        (255, 60, 60),
        2,
        cv2.LINE_AA,
        tipLength=0.3,
    )
    cv2.circle(canvas, (center, center), 4, (255, 60, 60), -1, cv2.LINE_AA)

    if horizon:
        _draw_ego_horizon(
            canvas, horizon, side_px=side_px, window_m=window_m
        )

    cv2.putText(
        canvas,
        "Goal compass (GT)",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    caption = f"max={max_val:.2f}"
    if horizon:
        caption += f" | next 1-{len(horizon)}"
        if _horizon_is_turn_in_place(horizon):
            caption += " turn"
        if _horizon_has_turn_direction(horizon):
            caption += " | orange=this step turn"
    cv2.putText(
        canvas,
        caption,
        (8, side_px - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return canvas


def build_task_config(
    config_path: str,
    split: str,
    allow_sliding: bool,
    success_distance: float,
) -> habitat.config.Config:
    cfg = get_config(config_path, opts=None)
    task_cfg = cfg.TASK_CONFIG.clone()
    task_cfg.defrost()
    task_cfg.DATASET.SPLIT = split
    sim_sensors = list(task_cfg.SIMULATOR.AGENT_0.SENSORS)
    for name in ("DEPTH_SENSOR", "SEMANTIC_SENSOR"):
        if name not in sim_sensors:
            sim_sensors.append(name)
    task_cfg.SIMULATOR.AGENT_0.SENSORS = sim_sensors
    task_cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    task_cfg.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = allow_sliding
    task_sensors = list(task_cfg.TASK.SENSORS)
    for name in (
        "GOAL_COMPASS_SENSOR",
        "SEMANTIC_MAP_SENSOR",
        "EGO_OBJECT_CLOUD_SENSOR",
    ):
        if name not in task_sensors:
            task_sensors.append(name)
    task_cfg.TASK.SENSORS = task_sensors
    task_cfg.TASK.SEMANTIC_MAP_SENSOR.CACHE_ROOT = ""
    task_cfg.TASK.EGO_OBJECT_CLOUD_SENSOR.CACHE_ROOT = ""
    actions = list(task_cfg.TASK.POSSIBLE_ACTIONS)
    if "REPLAY_TELEPORT" not in actions:
        actions.append("REPLAY_TELEPORT")
    task_cfg.TASK.POSSIBLE_ACTIONS = actions
    task_cfg.TASK.SUCCESS.SUCCESS_DISTANCE = success_distance
    task_cfg.TASK.SUCCESS_DISTANCE = success_distance
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = 10**12
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
    measurements = list(task_cfg.TASK.MEASUREMENTS)
    if "TOP_DOWN_MAP" not in measurements:
        measurements.append("TOP_DOWN_MAP")
    task_cfg.TASK.MEASUREMENTS = measurements
    task_cfg.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 512
    task_cfg.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
    task_cfg.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
    task_cfg.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
    task_cfg.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = False
    task_cfg.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
    task_cfg.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
    task_cfg.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = False
    task_cfg.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = False
    task_cfg.freeze()
    return task_cfg


def load_raw_episodes(split: str, scene: str) -> Dict[str, dict]:
    path = Path(DATASET_ROOT) / split / "content" / f"{scene}.json.gz"
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return {str(ep["episode_id"]): ep for ep in data["episodes"]}


def action_name_to_id(action_name: str) -> int:
    if action_name not in ACTION_NAME_TO_ID:
        action_name = str(action_name).upper().split(".")[-1]
    return ACTION_NAME_TO_ID[action_name]


def normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


def _draw_topdown_horizon(
    colorized_map: np.ndarray,
    agent_map_coord: Tuple[int, int],
    waypoints: Sequence[ExpertHorizonWaypoint],
    *,
    tick_len: int = 10,
) -> None:
    """Draw expert polyline and heading ticks on a colorized top-down map."""
    if not waypoints:
        return
    color = HORIZON_COLOR_RGB
    pts = [agent_map_coord] + [wp.map_coord for wp in waypoints]
    for i in range(len(pts) - 1):
        cv2.line(
            colorized_map,
            pts[i][::-1],
            pts[i + 1][::-1],
            color,
            HORIZON_LINE_THICKNESS,
            cv2.LINE_AA,
        )
    for wp in waypoints:
        col, row = wp.map_coord[::-1]
        phi = wp.map_angle
        end_col = int(col + np.sin(phi) * tick_len)
        end_row = int(row - np.cos(phi) * tick_len)
        cv2.arrowedLine(
            colorized_map,
            (col, row),
            (end_col, end_row),
            color,
            HORIZON_ARROW_THICKNESS,
            cv2.LINE_AA,
            tipLength=0.35,
        )


def top_down_panel(
    metrics: dict,
    target_h: int,
    horizon: Optional[Sequence[ExpertHorizonWaypoint]] = None,
) -> np.ndarray:
    tdm_info = metrics.get("top_down_map")
    if tdm_info is None:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)

    top_down_map = maps.colorize_topdown_map(
        tdm_info["map"],
        tdm_info.get("fog_of_war_mask"),
    )
    if horizon:
        _draw_topdown_horizon(
            top_down_map,
            tuple(tdm_info["agent_map_coord"]),
            horizon,
        )
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=tdm_info["agent_map_coord"],
        agent_rotation=tdm_info["agent_angle"],
        agent_radius_px=min(top_down_map.shape[0:2]) // 32,
    )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    old_h, old_w, _ = top_down_map.shape
    top_down_height = target_h
    top_down_width = int(float(top_down_height) / old_h * old_w)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    if top_down_map.shape[0] != target_h or top_down_map.shape[1] != target_h:
        top_down_map = cv2.resize(
            top_down_map, (target_h, target_h), interpolation=cv2.INTER_AREA
        )
    return top_down_map


def _resize_to_h(img: np.ndarray, out_h: int, nearest: bool = False) -> np.ndarray:
    w = max(1, int(round(img.shape[1] * out_h / img.shape[0])))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, (w, out_h), interpolation=interp)


def _square_panel(img: np.ndarray, side_px: int, nearest: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, (side_px, side_px), interpolation=interp)


def semantic_image_to_rgb(
    semantic: np.ndarray, instance_to_task: np.ndarray
) -> np.ndarray:
    """Color each pixel by ObjectNav task class (matches teleop_semantic_map)."""
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    inst = np.clip(semantic.astype(np.int64), 0, len(instance_to_task) - 1)
    task = instance_to_task[inst]
    channel = np.where(task >= 0, task + 2, OCCUPIED).astype(np.int8)
    return PALETTE[channel]


def render_frame(
    obs: dict,
    metrics: dict,
    step_idx: int,
    goal: str,
    success: int,
    instance_to_task: np.ndarray,
    out_h: int = 480,
    horizon: Optional[Sequence[ExpertHorizonWaypoint]] = None,
    horizon_steps: int = 0,
) -> np.ndarray:
    rgb_panel = _resize_to_h(normalise_rgb(obs["rgb"]), out_h)
    sem_img_panel = _resize_to_h(
        semantic_image_to_rgb(obs["semantic"], instance_to_task), out_h, nearest=True
    )
    compass_panel = render_goal_compass_bev_panel(
        np.asarray(obs["goal_compass"], dtype=np.float32),
        horizon,
        side_px=out_h,
    )
    sem_map_panel = render_semantic_map_panel(
        np.asarray(obs["semantic_map"]), horizon, side_px=out_h
    )
    cloud_panel = render_ego_cloud_panel(
        np.asarray(obs["ego_object_cloud"], dtype=np.float32),
        horizon,
        side_px=out_h,
    )
    map_panel = top_down_panel(metrics, out_h, horizon=horizon)
    canvas = np.concatenate(
        [rgb_panel, sem_img_panel, compass_panel, sem_map_panel, cloud_panel, map_panel],
        axis=1,
    )
    label = f"step {step_idx}  |  goal: {goal}  |  success: {success}"
    if horizon_steps > 0:
        label += f"  |  horizon={horizon_steps}"
    cv2.putText(
        canvas,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def episode_already_recorded(video_dir: Path, episode_id: str) -> bool:
    return bool(list(video_dir.glob(f"episode={episode_id}-success=*.mp4")))


def write_video(
    frames: List[np.ndarray],
    video_dir: Path,
    episode_id: str,
    success: float,
    fps: int,
) -> None:
    if not frames:
        print(f"  [warn] ep={episode_id}: no frames, skipping video")
        return
    video_dir.mkdir(parents=True, exist_ok=True)
    name = f"episode={episode_id}-success={success:.2f}"
    images_to_video(frames, str(video_dir), name, fps=fps, verbose=False)


def _agent_state_dict(step: Any) -> Optional[Dict[str, List[float]]]:
    state = (
        step.agent_state
        if hasattr(step, "agent_state")
        else step.get("agent_state")
    )
    if state is None:
        return None
    if hasattr(state, "position"):
        return {
            "position": list(state.position),
            "rotation": list(state.rotation),
        }
    return {
        "position": list(state["position"]),
        "rotation": list(state["rotation"]),
    }


def replay_actions(env: habitat.Env, ep) -> List[Tuple[dict, dict, dict, int]]:
    replay = list(ep.reference_replay)
    if not replay:
        return []

    env.episode_iterator = iter([ep])
    obs = env.reset()
    state0 = _agent_state_dict(replay[0])
    if state0 is None:
        return []
    frames: List[Tuple[dict, dict, dict, int]] = [
        (obs, env.get_metrics(), state0, 0)
    ]

    start_idx = 1 if hasattr(replay[0], "action") else 0
    for idx in range(start_idx, len(replay)):
        obs = env.step(action_name_to_id(replay[idx].action))
        state = _agent_state_dict(replay[idx])
        if state is None:
            continue
        frames.append((obs, env.get_metrics(), state, idx))
        if env.episode_over:
            break
    return frames


def _teleport_to(env: habitat.Env, agent_state: dict) -> Optional[dict]:
    return env.step(
        {
            "action": "REPLAY_TELEPORT",
            "action_args": {
                "position": list(agent_state["position"]),
                "rotation": list(agent_state["rotation"]),
            },
        }
    )


def replay_poses(
    env: habitat.Env, ep, raw_episode: dict
) -> Tuple[List[Tuple[dict, dict, dict, int]], dict]:
    env.episode_iterator = iter([ep])
    env.reset()

    frames: List[Tuple[dict, dict, dict, int]] = []
    for replay_idx, step in enumerate(raw_episode.get("reference_replay", [])):
        state = step.get("agent_state")
        if state is None:
            continue
        obs = _teleport_to(env, state)
        if obs is None:
            print(f"  [warn] ep={ep.episode_id}: agent placement failed")
            continue
        agent_state = {
            "position": list(state["position"]),
            "rotation": list(state["rotation"]),
        }
        frames.append((obs, env.get_metrics(), agent_state, replay_idx))

    if frames and not env.episode_over:
        env.step(0)

    return frames, env.get_metrics()


def process_episode(
    env: habitat.Env,
    ep,
    args: argparse.Namespace,
    video_dir: Path,
    raw_by_id: Optional[Dict[str, dict]],
) -> Optional[Tuple[float, float]]:
    """Run one episode. Returns (success, spl) if processed, else None."""
    eid = str(ep.episode_id)
    if not args.overwrite and episode_already_recorded(video_dir, eid):
        return None

    goal = getattr(ep, "object_category", None) or "unknown"

    if args.replay_mode == "actions":
        frame_data = replay_actions(env, ep)
        final_metrics = frame_data[-1][1] if frame_data else env.get_metrics()
    else:
        raw = (raw_by_id or {}).get(eid)
        if raw is None:
            print(f"  [warn] ep={eid}: missing raw episode, skip")
            return None
        frame_data, final_metrics = replay_poses(env, ep, raw)

    if not frame_data:
        print(f"  [warn] ep={eid}: no frames, skip")
        return None

    success_val = float(final_metrics.get("success", 0.0))
    spl_val = float(final_metrics.get("spl", 0.0))
    success_bin = int(success_val > 0.5)
    instance_to_task = build_instance_to_task_id(env.sim)
    reference_replay = list(ep.reference_replay)
    pathfinder = env.sim.pathfinder
    rendered: List[np.ndarray] = []
    for t, (obs, m, agent_state, replay_idx) in enumerate(frame_data):
        horizon: List[ExpertHorizonWaypoint] = []
        if args.horizon_steps > 0:
            tdm_info = m.get("top_down_map")
            if tdm_info is not None:
                horizon = expert_horizon_from_replay(
                    reference_replay,
                    replay_idx,
                    agent_state["position"],
                    agent_state["rotation"],
                    args.horizon_steps,
                    tdm_info["map"].shape,
                    pathfinder,
                )
        rendered.append(
            render_frame(
                obs,
                m,
                t,
                goal,
                success_bin,
                instance_to_task,
                horizon=horizon or None,
                horizon_steps=args.horizon_steps,
            )
        )
    frames = rendered
    write_video(frames, video_dir, eid, success_val, args.fps)
    print(
        f"  ep={eid:>4} goal={goal:<12} frames={len(frames):>4} "
        f"success={success_bin} spl={spl_val:.3f}"
    )
    return success_val, spl_val


def main() -> None:
    ensure_conda_env()
    args = parse_args()

    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

    video_dir = video_output_dir(
        args.out_root, args.replay_mode, args.allow_sliding
    )
    video_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = build_task_config(
        args.config, args.split, args.allow_sliding, args.success_distance
    )
    env = habitat.Env(config=task_cfg)

    raw_by_id = None
    if args.replay_mode == "poses":
        raw_by_id = load_raw_episodes(args.split, args.scene)

    episodes = list(env.episodes)
    if args.episode_ids:
        want = {str(x) for x in args.episode_ids}
        episodes = [e for e in episodes if str(e.episode_id) in want]
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    print(
        f"[replay] mode={args.replay_mode} split={args.split} "
        f"episodes={len(episodes)} sliding={args.allow_sliding} "
        f"success_distance={args.success_distance}m horizon={args.horizon_steps} "
        f"out={video_dir}"
    )

    t0 = time.time()
    n_processed = 0
    n_success = 0
    spl_sum = 0.0
    try:
        for i, ep in enumerate(episodes):
            result = process_episode(env, ep, args, video_dir, raw_by_id)
            if result is not None:
                success_val, spl_val = result
                n_processed += 1
                n_success += int(success_val > 0.5)
                spl_sum += spl_val
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(episodes)} ({time.time() - t0:.0f}s)")
    finally:
        env.close()

    elapsed = time.time() - t0
    print(f"[replay] done in {elapsed:.1f}s -> {video_dir}")
    if n_processed > 0:
        print(
            f"[replay] summary ({n_processed} episodes processed this run): "
            f"success_rate={100.0 * n_success / n_processed:.2f}% "
            f"({n_success}/{n_processed})  mean_spl={spl_sum / n_processed:.4f}"
        )
    else:
        print("[replay] summary: no episodes processed (all skipped or missing)")


if __name__ == "__main__":
    main()
