"""Reusable accumulator for an in-memory object cloud over MP3D ObjectNav.

Each tracked object is a world-space ``(x, y, z)`` centroid plus an ObjectNav
task id (``0..20``). Centroids are mask-area-weighted across views, so
revisiting an object refines its position rather than overwriting it.

Shared between the offline teleop visualizer (``teleop_object_cloud.py``)
and -- in the future -- an online sensor for IL training. Category constants
and the per-scene ``instance_to_task`` table are imported from
``pirlnav.task.semantic_map`` so there is one source of truth for the MP3D
ObjectNav scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import quaternion  # noqa: F401  (registers np.quaternion dtype)

from pirlnav.task.semantic_map import OBJECTNAV_CATEGORIES, PALETTE


TASK_NAMES: List[str] = [name for name, _ in OBJECTNAV_CATEGORIES]


def make_camera_intrinsics(
    width: int, height: int, hfov_deg: float
) -> Tuple[float, float, float, float]:
    """Pinhole ``(fx, fy, cx, cy)`` for square-pixel camera with given HFOV."""
    f = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    return f, f, width / 2.0, height / 2.0


def depth_to_world_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    sensor_pos: np.ndarray,
    sensor_rot: "np.quaternion",
    fx: float, fy: float, cx: float, cy: float,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    """Back-project masked depth pixels to ``(N, 3)`` world points.

    Habitat camera frame is OpenGL-style: ``+X`` right, ``+Y`` up, ``-Z``
    forward. ``sensor_rot`` is the live sensor world-rotation quaternion.
    """
    depth = np.asarray(depth_m, dtype=np.float64).squeeze()
    valid = mask & (depth > depth_min) & (depth < depth_max)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    v, u = np.nonzero(valid)
    d = depth[v, u]
    x_cam = (u - cx) * d / fx
    y_cam = -(v - cy) * d / fy
    z_cam = -d

    R = quaternion.as_rotation_matrix(sensor_rot)
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=0)
    return (R @ pts_cam).T + np.asarray(sensor_pos, dtype=np.float64)


@dataclass
class _ObjectState:
    """Mask-area-weighted running estimate for one tracked instance."""

    instance_idx: int
    task_id: int
    weighted_centroid_sum: np.ndarray  # (3,) -- sum(centroid_i * area_i)
    total_weight: float = 0.0

    @property
    def centroid(self) -> np.ndarray:
        return self.weighted_centroid_sum / max(self.total_weight, 1e-6)


class ObjectCloud:
    """Incrementally builds a flat object cloud of xyz centroids + task ids.

    On each ``update`` we walk the unique instance ids in the current
    semantic frame, keep only those whose mapped task id is in ``[0, 20]``
    and whose visible mask is at least ``min_mask_pixels`` pixels, then
    back-project the masked depth to world space and update the running
    centroid weighted by visible area.
    """

    def __init__(self, instance_to_task: np.ndarray, min_mask_pixels: int = 100):
        self.instance_to_task = instance_to_task
        self.min_mask_pixels = int(min_mask_pixels)
        self._objects: Dict[int, _ObjectState] = {}

    def update(
        self,
        depth_m: np.ndarray,
        semantic: np.ndarray,
        sensor_pos: np.ndarray,
        sensor_rot: "np.quaternion",
        fx: float, fy: float, cx: float, cy: float,
        depth_min: float,
        depth_max: float,
    ) -> Tuple[List[int], List[int]]:
        """Process one observation. Returns ``(new_ids, updated_ids)``."""
        sem = np.asarray(semantic).squeeze().astype(np.int64)
        new_ids: List[int] = []
        updated_ids: List[int] = []

        for inst_idx in np.unique(sem):
            i = int(inst_idx)
            if i < 0 or i >= len(self.instance_to_task):
                continue
            task_id = int(self.instance_to_task[i])
            if task_id < 0:
                continue

            mask = sem == i
            area = int(mask.sum())
            if area < self.min_mask_pixels:
                continue

            pts = depth_to_world_points(
                depth_m, mask, sensor_pos, sensor_rot,
                fx, fy, cx, cy, depth_min, depth_max,
            )
            if pts.shape[0] == 0:
                continue
            centroid = pts.mean(axis=0)

            if i in self._objects:
                state = self._objects[i]
                state.weighted_centroid_sum += centroid * area
                state.total_weight += area
                updated_ids.append(i)
            else:
                self._objects[i] = _ObjectState(
                    instance_idx=i,
                    task_id=task_id,
                    weighted_centroid_sum=centroid * area,
                    total_weight=float(area),
                )
                new_ids.append(i)

        return new_ids, updated_ids

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Snapshot the cloud as numpy arrays + a parallel labels list."""
        keys = sorted(self._objects.keys())
        n = len(keys)
        if n == 0:
            return {
                "obj_pos": np.zeros((0, 3), dtype=np.float32),
                "task_ids": np.zeros((0,), dtype=np.int64),
                "labels": [],
                "n_objects": 0,
            }
        positions = np.stack(
            [self._objects[k].centroid for k in keys]
        ).astype(np.float32)
        task_ids = np.array(
            [self._objects[k].task_id for k in keys], dtype=np.int64
        )
        return {
            "obj_pos": positions,
            "task_ids": task_ids,
            "labels": [TASK_NAMES[t] for t in task_ids],
            "n_objects": n,
        }

    def to_ego_dict(
        self,
        agent_pos: np.ndarray,
        agent_rot: "np.quaternion",
    ) -> Dict[str, np.ndarray]:
        """Snapshot the cloud in agent-frame coordinates, encoder-ready.

        Storage stays world-frame (so weighted-centroid refinement keeps
        accumulating correctly); this method just transforms the snapshot:

            obj_pos_ego = R_agent.T @ (obj_pos_world - agent_pos_world)
            agent_pos   = 0

        Habitat's agent body frame inherits the OpenGL camera convention
        (+X right, +Y up, -Z forward), so ``ego_z < 0`` means "in front
        of the agent".
        """
        world = self.to_dict()
        n = world["n_objects"]
        agent_pos_ego = np.zeros(3, dtype=np.float32)
        if n == 0:
            return {**world, "agent_pos": agent_pos_ego}

        R = quaternion.as_rotation_matrix(agent_rot)  # world <- agent
        delta = world["obj_pos"] - np.asarray(agent_pos, dtype=np.float32)
        ego = (delta @ R).astype(np.float32)  # equiv to (R.T @ delta.T).T
        return {
            "obj_pos": ego,
            "task_ids": world["task_ids"],
            "labels": world["labels"],
            "n_objects": n,
            "agent_pos": agent_pos_ego,
        }


def render_ego_cloud_topdown(
    packed: np.ndarray,
    side_px: int = 480,
    window_m: float = 12.0,
) -> np.ndarray:
    """Render an agent-frame top-down panel from the packed sensor output.

    ``packed`` is the ``(MAX_OBJECTS, 4) float32`` array emitted by
    :class:`pirlnav.task.sensors.EgoObjectCloudSensor`: each row is
    ``[task_id, ex, ey, ez]`` with padding rows having ``task_id < 0``.
    Forward (``ez < 0``) maps to canvas-up; right (``ex > 0``) to canvas-
    right. Object dots are colored by goal class via ``PALETTE`` (rows
    2..22, matching the semantic-map palette).
    """
    canvas = np.full((side_px, side_px, 3), 30, dtype=np.uint8)
    res = window_m / side_px
    center = side_px // 2

    for k in range(-int(window_m // 2), int(window_m // 2) + 1):
        offset = int(round(k / res + center))
        if 0 <= offset < side_px:
            cv2.line(canvas, (offset, 0), (offset, side_px), (50, 50, 50), 1)
            cv2.line(canvas, (0, offset), (side_px, offset), (50, 50, 50), 1)

    if packed.size:
        valid = packed[:, 0] >= 0
        for tid, ex, _, ez in packed[valid]:
            col = int(round(ex / res + center))
            row = int(round(ez / res + center))
            if not (0 <= col < side_px and 0 <= row < side_px):
                continue
            color = tuple(int(c) for c in PALETTE[int(tid) + 2])
            cv2.circle(canvas, (col, row), 5, color, -1, cv2.LINE_AA)

    cv2.arrowedLine(canvas, (center, center), (center, center - 30),
                    (255, 60, 60), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(canvas, (center, center), 4, (255, 60, 60), -1, cv2.LINE_AA)
    return canvas
