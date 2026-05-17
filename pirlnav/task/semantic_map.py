"""Reusable egocentric semantic + occupancy mapper for ObjectNav.

Shared between the offline teleop visualizer (``teleop_semantic_map.py``) and
the online ``SemanticMapSensor`` used during IL training. This module is the
single source of truth for the map's class layout, label values, and the
projection / egocentric-rotation math.

Map shape: ``(H, W)`` int8 label map, agent-centered and agent-oriented:

* ``-1`` UNKNOWN (unobserved cell)
* ``0``  FREE
* ``1``  OCCUPIED / unknown class
* ``2..22`` the 21 MP3D ObjectNav goal classes (chair, table, ...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import quaternion  # noqa: F401  (registers np.quaternion dtype)


# 21 MP3D ObjectNav goal categories: (name, mpcat40 index).
OBJECTNAV_CATEGORIES: List[Tuple[str, int]] = [
    ("chair", 3), ("table", 5), ("picture", 6), ("cabinet", 7),
    ("cushion", 8), ("sofa", 10), ("bed", 11), ("chest_of_drawers", 13),
    ("plant", 14), ("sink", 15), ("toilet", 18), ("stool", 19),
    ("towel", 20), ("tv_monitor", 22), ("shower", 23), ("bathtub", 25),
    ("counter", 26), ("fireplace", 27), ("gym_equipment", 33),
    ("seating", 34), ("clothes", 38),
]
NUM_CATEGORIES: int = len(OBJECTNAV_CATEGORIES)  # 21
NUM_CHANNELS: int = NUM_CATEGORIES + 2           # 23 (FREE + OCCUPIED + 21)

NAME_TO_TASK: Dict[str, int] = {
    name: i for i, (name, _) in enumerate(OBJECTNAV_CATEGORIES)
}

UNKNOWN: int = -1
FREE: int = 0
OCCUPIED: int = 1

# Height band (offsets relative to the agent's current y) used to classify
# back-projected points into FREE vs OCCUPIED vs out-of-band.
FLOOR_LOW_OFFSET: float = -0.15
FREE_HIGH_OFFSET: float = 0.20
CEILING_OFFSET: float = 1.50


def _make_mpcat40_to_task() -> np.ndarray:
    table = np.full(64, -1, dtype=np.int32)
    for task_id, (_, mp) in enumerate(OBJECTNAV_CATEGORIES):
        table[mp] = task_id
    return table


MPCAT40_TO_TASK: np.ndarray = _make_mpcat40_to_task()


# Visualization palette: one RGB row per label value written into the map
# (slot 0 = FREE, slot 1 = OCCUPIED, slots 2..22 = the 21 goal classes in
# OBJECTNAV_CATEGORIES order). Hand-picked from Sasha Trubetskoy's "20
# distinct colors" list so adjacent categories stay perceptually separable.
PALETTE: np.ndarray = np.array([
    (200, 200, 200),  # 0  FREE
    ( 60,  60,  60),  # 1  OCCUPIED
    (230,  25,  75),  # chair
    ( 60, 180,  75),  # table
    (255, 225,  25),  # picture
    (  0, 130, 200),  # cabinet
    (245, 130,  48),  # cushion
    (145,  30, 180),  # sofa
    ( 70, 240, 240),  # bed
    (240,  50, 230),  # chest_of_drawers
    (210, 245,  60),  # plant
    (250, 190, 212),  # sink
    (  0, 128, 128),  # toilet
    (220, 190, 255),  # stool
    (170, 110,  40),  # towel
    (255, 250, 200),  # tv_monitor
    (128,   0,   0),  # shower
    (170, 255, 195),  # bathtub
    (128, 128,   0),  # counter
    (255, 215, 180),  # fireplace
    (  0,   0, 128),  # gym_equipment
    (128, 128, 128),  # seating
    (255, 255, 255),  # clothes
], dtype=np.uint8)
assert PALETTE.shape == (NUM_CHANNELS, 3)


def label_map_to_rgb(
    label_map: np.ndarray, unknown_color: Tuple[int, int, int] = (20, 20, 20)
) -> np.ndarray:
    """``(H, W)`` int8 label map -> ``(H, W, 3)`` uint8 RGB.

    UNKNOWN cells (``label == -1``) are painted ``unknown_color`` so that
    "unobserved" is visually distinct from FREE / OCCUPIED. All other label
    values are looked up directly in ``PALETTE``.
    """
    out = np.full((*label_map.shape, 3), unknown_color, dtype=np.uint8)
    valid = label_map >= 0
    out[valid] = PALETTE[label_map[valid]]
    return out


def load_class_embeddings(json_path: Path) -> np.ndarray:
    """Load (NUM_CATEGORIES, D) L2-normalized class embeddings from a tasks
    json (each entry has a ``text_embedding`` field). Rows are indexed by
    task id (order of ``OBJECTNAV_CATEGORIES``).
    """
    with open(json_path) as f:
        tasks = json.load(f)
    sample = np.asarray(
        tasks[OBJECTNAV_CATEGORIES[0][0]]["text_embedding"], dtype=np.float32
    )
    embs = np.zeros((NUM_CATEGORIES, sample.size), dtype=np.float32)
    for task_id, (name, _) in enumerate(OBJECTNAV_CATEGORIES):
        embs[task_id] = np.asarray(tasks[name]["text_embedding"], dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs


def build_instance_to_task_id(sim) -> np.ndarray:
    """Return a table mapping habitat-sim instance id -> ObjectNav task id
    (0..20), or -1 if the instance is not a goal class. Rebuild whenever the
    underlying scene changes; cheap (one pass over semantic annotations).
    """
    pairs: List[Tuple[int, int]] = []
    max_id = 0
    for obj in sim.semantic_annotations().objects:
        if obj is None or obj.category is None:
            continue
        try:
            inst_id = int(obj.id.split("_")[-1])
        except (ValueError, AttributeError):
            continue
        mp = int(obj.category.index("mpcat40"))
        task = int(MPCAT40_TO_TASK[mp]) if 0 <= mp < len(MPCAT40_TO_TASK) else -1
        pairs.append((inst_id, task))
        max_id = max(max_id, inst_id)
    table = np.full(max_id + 1, -1, dtype=np.int32)
    for inst_id, task in pairs:
        table[inst_id] = task
    return table


class SemanticMapper:
    """Builds and queries an egocentric semantic + occupancy map.

    Internally maintains a world-anchored ``(H_g, W_g)`` int8 label map sized
    to cover an entire MP3D house at the chosen resolution. ``update`` projects
    a single depth+semantic frame into world space and writes labels into the
    global map. ``egocentric_view`` rotates / crops a fixed-size window around
    the agent (forward = up).
    """

    def __init__(
        self,
        H: int,
        W: int,
        resolution: float,
        start_pos: np.ndarray,
        world_diameter_m: float = 80.0,
    ) -> None:
        self.H, self.W = int(H), int(W)
        self.resolution = float(resolution)
        side = max(
            2 * max(self.H, self.W),
            int(round(world_diameter_m / self.resolution)),
        )
        self.H_g = self.W_g = side
        self.origin_x = float(start_pos[0])
        self.origin_z = float(start_pos[2])
        self.global_map = np.full((self.H_g, self.W_g), UNKNOWN, dtype=np.int8)

    @classmethod
    def from_cached(
        cls,
        global_map: np.ndarray,
        origin_x: float,
        origin_z: float,
        resolution: float,
        H: int,
        W: int,
    ) -> "SemanticMapper":
        """Build a mapper around an *already-built* world-frame label map.

        Skips the ``start_pos``/``world_diameter_m`` sizing in ``__init__``: the
        global map is taken from disk verbatim, and only ``H, W`` (the size of
        the egocentric crop ``egocentric_view`` should emit) and the world
        anchor ``(origin_x, origin_z, resolution)`` come from the caller.

        Used by ``SemanticMapSensor`` in cached mode and any other consumer
        that loads a precomputed ``.npz`` (see ``teleop_semantic_map.save_map``
        for the on-disk schema).
        """
        m = cls.__new__(cls)
        m.H, m.W = int(H), int(W)
        m.resolution = float(resolution)
        m.global_map = np.ascontiguousarray(global_map, dtype=np.int8)
        m.H_g, m.W_g = int(m.global_map.shape[0]), int(m.global_map.shape[1])
        m.origin_x = float(origin_x)
        m.origin_z = float(origin_z)
        return m

    def _world_to_cell(
        self, x: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        col = np.round(
            (x - self.origin_x) / self.resolution + self.W_g / 2.0
        ).astype(np.int32)
        row = np.round(
            (z - self.origin_z) / self.resolution + self.H_g / 2.0
        ).astype(np.int32)
        return row, col

    def update(
        self,
        depth_m: np.ndarray,
        semantic: np.ndarray,
        sensor_pos: np.ndarray,
        sensor_rot: "np.quaternion",
        agent_y: float,
        hfov_rad: float,
        instance_to_task: np.ndarray,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """Project a depth + semantic frame into the global label map."""
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        if semantic.ndim == 3:
            semantic = semantic[..., 0]

        H_img, W_img = depth_m.shape
        f = (W_img / 2.0) / np.tan(hfov_rad / 2.0)  # square pixels
        cx, cy = W_img / 2.0, H_img / 2.0

        valid = (depth_m > min_depth + 1e-6) & (depth_m < max_depth - 1e-6)
        if not valid.any():
            return
        v, u = np.nonzero(valid)
        d = depth_m[v, u].astype(np.float64)

        # Camera frame (OpenGL: +X right, +Y up, -Z forward) -> world.
        x_cam = (u - cx) * d / f
        y_cam = -(v - cy) * d / f
        z_cam = -d
        R = quaternion.as_rotation_matrix(sensor_rot)
        p_world = R @ np.stack([x_cam, y_cam, z_cam], axis=0) \
            + np.asarray(sensor_pos, dtype=np.float64)[:, None]
        x_w, y_w, z_w = p_world

        y_off = y_w - float(agent_y)
        in_band = (y_off >= FLOOR_LOW_OFFSET) & (y_off <= CEILING_OFFSET)
        row, col = self._world_to_cell(x_w, z_w)
        in_bounds = (row >= 0) & (row < self.H_g) & (col >= 0) & (col < self.W_g)
        keep = in_band & in_bounds
        if not keep.any():
            return
        row, col, y_off = row[keep], col[keep], y_off[keep]
        inst = semantic[v, u][keep].astype(np.int64)

        inst_safe = np.clip(inst, 0, len(instance_to_task) - 1)
        task_id = instance_to_task[inst_safe]

        # Per-point label: goal class -> task_id+2; near floor -> FREE; else OCCUPIED.
        near_floor = y_off < FREE_HIGH_OFFSET
        label = np.where(
            task_id >= 0,
            task_id + 2,
            np.where(near_floor, FREE, OCCUPIED),
        ).astype(np.int8)

        # Priority: a tracked-class cell can flip to FREE or to another tracked
        # class (latest wins), but a later OCCUPIED sweep cannot downgrade it.
        existing = self.global_map[row, col]
        skip = (existing >= 2) & (label == OCCUPIED)
        if not skip.all():
            write = ~skip
            self.global_map[row[write], col[write]] = label[write]

    def egocentric_view(
        self,
        agent_pos: np.ndarray,
        agent_rot: "np.quaternion",
    ) -> np.ndarray:
        """(H, W) agent-centered, agent-oriented label map. Forward = up."""
        H, W = self.H, self.W
        rr, cc = np.meshgrid(
            np.arange(H, dtype=np.float64),
            np.arange(W, dtype=np.float64),
            indexing="ij",
        )
        # Agent-local: +X right, -Z forward => col->X, row->+Z (behind agent).
        x_local = (cc - W / 2.0) * self.resolution
        z_local = (rr - H / 2.0) * self.resolution
        R = quaternion.as_rotation_matrix(agent_rot)
        x_world = agent_pos[0] + R[0, 0] * x_local + R[0, 2] * z_local
        z_world = agent_pos[2] + R[2, 0] * x_local + R[2, 2] * z_local

        row_g, col_g = self._world_to_cell(x_world, z_world)
        valid = (
            (row_g >= 0) & (row_g < self.H_g)
            & (col_g >= 0) & (col_g < self.W_g)
        )
        out = np.full((H, W), UNKNOWN, dtype=np.int8)
        out[valid] = self.global_map[row_g[valid], col_g[valid]]
        return out

    @staticmethod
    def to_onehot(label_map: np.ndarray) -> np.ndarray:
        """(H, W) labels -> (H, W, NUM_CHANNELS) one-hot float32. UNKNOWN
        cells become all-zero rows.
        """
        H, W = label_map.shape
        out = np.zeros((H, W, NUM_CHANNELS), dtype=np.float32)
        valid = label_map >= 0
        rows, cols = np.nonzero(valid)
        out[rows, cols, label_map[valid]] = 1.0
        return out


def smooth_label_map(label_map: np.ndarray, k: int) -> np.ndarray:
    """k x k mode (majority-vote) filter; UNKNOWN votes too. ``k <= 1`` is a
    no-op. Used by both the teleop visualization and the training-time sensor.
    """
    if k <= 1:
        return label_map
    H, W = label_map.shape
    shifted = label_map.astype(np.int32) + 1  # 0 = UNKNOWN, 1..NUM_CHANNELS = labels
    n = NUM_CHANNELS + 1
    votes = np.empty((n, H, W), dtype=np.int16)
    for c in range(n):
        votes[c] = cv2.boxFilter(
            (shifted == c).astype(np.float32), -1, (k, k),
            normalize=False, borderType=cv2.BORDER_CONSTANT,
        ).astype(np.int16)
    return (votes.argmax(axis=0) - 1).astype(np.int8)
