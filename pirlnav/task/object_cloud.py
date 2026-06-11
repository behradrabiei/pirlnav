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

from pirlnav.task.semantic_map import (
    MPCAT40_TO_TASK,
    OBJECTNAV_CATEGORIES,
    PALETTE,
)


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


def build_oracle_object_cloud(sim) -> Dict[str, np.ndarray]:
    """Ground-truth goal-class cloud from ``sim.semantic_annotations()``.

    Same instance filter as ``dump_scene_object_clouds.extract_scene_cloud``
    (mpcat40 -> task id in [0, 20]) but additionally keeps the semantic
    instance id and the AABB bounding radius, which the progressive-reveal
    visibility tests need. No rendering involved. Arrays are empty (N=0) for
    scenes without goal-class annotations.

    NOTE: positions are the annotation AABB centers, which usually match the
    semantic mesh but are not guaranteed to. For training, prefer the
    mesh-derived vertex centroids from ``scripts/dump_ply_object_clouds.py``
    (loaded by ``EgoObjectCloudSensor`` when ``CACHE_ROOT`` is set); this
    builder is the cache-less fallback for ad-hoc probing.
    """
    centers: List[np.ndarray] = []
    task_ids: List[int] = []
    radii: List[float] = []
    instance_ids: List[int] = []
    for obj in sim.semantic_annotations().objects:
        if obj is None or obj.category is None:
            continue
        try:
            inst_id = int(obj.id.split("_")[-1])
            mp = int(obj.category.index("mpcat40"))
        except (ValueError, AttributeError):
            continue
        task = int(MPCAT40_TO_TASK[mp]) if 0 <= mp < len(MPCAT40_TO_TASK) else -1
        if task < 0:
            continue
        centers.append(np.asarray(obj.aabb.center, dtype=np.float32))
        task_ids.append(task)
        radii.append(0.5 * float(np.linalg.norm(obj.aabb.sizes)))
        instance_ids.append(inst_id)
    n = len(centers)
    return {
        "obj_pos": (
            np.stack(centers) if n else np.zeros((0, 3), dtype=np.float32)
        ),
        "task_ids": np.array(task_ids, dtype=np.int64),
        "radii": np.array(radii, dtype=np.float32),
        "instance_ids": np.array(instance_ids, dtype=np.int64),
    }


class OracleRevealCloud:
    """Progressive reveal over a ground-truth scene cloud -- no rendering.

    Holds the full oracle cloud (from :func:`build_oracle_object_cloud`) plus
    a persistent per-episode ``revealed`` mask. Each ``update`` reveals the
    still-hidden objects the camera plausibly sees right now:

      1. range gate -- camera-plane depth within ``[min_dist, max_dist]``
         (mirrors the old DEPTH_SENSOR window),
      2. frustum gate -- inside the camera's HFOV/VFOV cone (camera frame is
         OpenGL-style: +X right, +Y up, -Z forward),
      3. size gate -- projected angular radius >= ``min_angular_size``
         (the analog of the online cloud's MIN_MASK_PIXELS threshold),
      4. occlusion gate -- at least one of 5 rays (AABB center and four
         half-radius offsets along the camera's right/up axes) is not
         blocked. ``cast_ray_fn(origin, direction, max_dist)`` returns the
         first-hit distance or ``None``; MP3D is one merged stage mesh, so
         each ray's test is "first hit no earlier than
         ``dist - min(radius, occlusion_margin)``" (the ray legitimately
         hits the object's own front surface).

    Gates 1-2 are slackened by the object's radius so partially-visible
    objects (poking into the frame edge or the depth window) remain
    candidates, mirroring what a pixel mask would catch. A single center
    ray is too strict in clutter (chairs under tables, frame-edge objects
    get occluded at their center); the 4 offset rays approximate
    partial-mask visibility.

    Once revealed, an object stays revealed until :meth:`reset` (called per
    episode, matching the online accumulator's lifetime).
    """

    def __init__(
        self,
        obj_pos: np.ndarray,
        task_ids: np.ndarray,
        radii: np.ndarray,
        hfov_deg: float,
        aspect_hw: float,
        min_dist: float,
        max_dist: float,
        min_angular_size: float,
        occlusion_margin: float,
        cast_ray_fn,
    ):
        self.obj_pos = np.asarray(obj_pos, dtype=np.float64)
        self.task_ids = np.asarray(task_ids, dtype=np.int64)
        self.radii = np.asarray(radii, dtype=np.float64)
        self._tan_half_h = np.tan(np.deg2rad(hfov_deg) / 2.0)
        self._tan_half_v = self._tan_half_h * aspect_hw
        self._min_dist = float(min_dist)
        self._max_dist = float(max_dist)
        self._min_angular_size = float(min_angular_size)
        self._occlusion_margin = float(occlusion_margin)
        self._cast_ray = cast_ray_fn
        self.revealed = np.zeros(len(self.task_ids), dtype=bool)

    def reset(self) -> None:
        self.revealed[:] = False

    def update(
        self, cam_pos: np.ndarray, cam_rot: "np.quaternion"
    ) -> List[int]:
        """Reveal objects visible from the camera pose; returns new indices."""
        hidden = np.nonzero(~self.revealed)[0]
        if hidden.size == 0:
            return []
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        R = quaternion.as_rotation_matrix(cam_rot)  # world <- camera
        rel = self.obj_pos[hidden] - cam_pos
        pts_cam = rel @ R
        fwd = -pts_cam[:, 2]
        dist = np.linalg.norm(rel, axis=1)
        r = self.radii[hidden]
        ok = (
            (fwd > 0.0)
            & (fwd + r >= self._min_dist)
            & (fwd - r <= self._max_dist)
            & (np.abs(pts_cam[:, 0]) - r <= fwd * self._tan_half_h)
            & (np.abs(pts_cam[:, 1]) - r <= fwd * self._tan_half_v)
            & (r >= dist * self._min_angular_size)
        )
        cam_right, cam_up = R[:, 0], R[:, 1]
        new_ids: List[int] = []
        for j in np.nonzero(ok)[0]:
            idx = int(hidden[j])
            radius = float(self.radii[idx])
            margin = min(radius, self._occlusion_margin)
            center = self.obj_pos[idx]
            targets = (
                center,
                center + 0.5 * radius * cam_right,
                center - 0.5 * radius * cam_right,
                center + 0.5 * radius * cam_up,
                center - 0.5 * radius * cam_up,
            )
            for target in targets:
                to_target = target - cam_pos
                d = float(np.linalg.norm(to_target))
                hit = self._cast_ray(cam_pos, to_target / d, d)
                if hit is None or hit >= d - margin:
                    self.revealed[idx] = True
                    new_ids.append(idx)
                    break
        return new_ids

    def ego_snapshot(
        self, agent_pos: np.ndarray, agent_rot: "np.quaternion"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Revealed objects in agent-frame coordinates, ``_pack_ego``-ready."""
        R = quaternion.as_rotation_matrix(agent_rot)
        delta = self.obj_pos[self.revealed] - np.asarray(agent_pos, dtype=np.float64)
        return (delta @ R).astype(np.float32), self.task_ids[self.revealed]


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
