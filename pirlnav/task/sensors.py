import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import quaternion  # noqa: F401  (registers np.quaternion dtype)
import torch
from gym import spaces
from habitat import logger
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from pirlnav.task.object_cloud import ObjectCloud, make_camera_intrinsics
from pirlnav.task.semantic_map import (
    NUM_CATEGORIES,
    NUM_CHANNELS,
    SemanticMapper,
    build_instance_to_task_id,
    smooth_label_map,
)


def get_habitat_sim_action(action):
    if action == "TURN_RIGHT":
        return HabitatSimActions.TURN_RIGHT
    elif action == "TURN_LEFT":
        return HabitatSimActions.TURN_LEFT
    elif action == "MOVE_FORWARD":
        return HabitatSimActions.MOVE_FORWARD
    elif action == "LOOK_UP":
        return HabitatSimActions.LOOK_UP
    elif action == "LOOK_DOWN":
        return HabitatSimActions.LOOK_DOWN
    return HabitatSimActions.STOP


@registry.register_sensor(name="DemonstrationSensor")
class DemonstrationSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "next_actions"
        self.observation_space = spaces.Discrete(1)
        self.timestep = 0
        self.prev_action = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        # Fetch next action as observation
        if task._is_resetting:  # reset
            self.timestep = 1

        if self.timestep < len(episode.reference_replay):
            action_name = episode.reference_replay[self.timestep].action
            action = get_habitat_sim_action(action_name)
        else:
            action = 0

        self.timestep += 1
        return action

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="NextPoseSensor")
class NextPoseSensor(Sensor):
    r"""Per-step expert agent pose for pose-replay IL.

    Mirrors :class:`DemonstrationSensor`'s timestep bookkeeping so the
    emitted pose at step ``t`` aligns with the action emitted by
    ``next_actions`` at the same step.  Returns a 7-D float32 vector
    ``[px, py, pz, qx, qy, qz, qw]`` read from
    ``episode.reference_replay[t].agent_state``.  When the recorded
    ``agent_state`` is missing (legacy data) or the replay is exhausted, a
    zero vector is returned; the IL trainer only dispatches a TELEPORT
    for non-STOP expert actions, so a stale/zero pose is never consumed.
    """

    POSE_DIM = 7

    def __init__(self, **kwargs):
        self.uuid = "next_pose"
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.POSE_DIM,),
            dtype=np.float32,
        )
        self.timestep = 0
        self._zero = np.zeros(self.POSE_DIM, dtype=np.float32)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs,
    ):
        if task._is_resetting:
            self.timestep = 1

        pose = self._zero
        if self.timestep < len(episode.reference_replay):
            state = episode.reference_replay[self.timestep].agent_state
            if state is not None and state.position is not None and state.rotation is not None:
                pose = np.asarray(
                    list(state.position) + list(state.rotation),
                    dtype=np.float32,
                )

        self.timestep += 1
        return pose

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


def _scene_id_to_name(scene_id: str) -> str:
    """Extract a bare scene name from a habitat scene_id path.

    Examples::
        "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb" -> "17DRP5sb8fy"
        "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb" -> "17DRP5sb8fy"
    """
    base = os.path.basename(scene_id)
    return os.path.splitext(base)[0]


@registry.register_sensor(name="CachedDINOv2Sensor")
class CachedDINOv2Sensor(Sensor):
    """Surface precomputed DINOv2 CLS features for the current replay step.

    Under pirlnav's teacher-forced IL loop the env is advanced step-by-step
    along ``episode.reference_replay``. RGB observed at step ``t`` is a
    deterministic function of ``(episode_id, t)``, so a one-time per-episode
    feature file at ``{cache_root}/{scene}/{episode_id}.pt`` (key
    ``"dino_cls"``, shape ``(T+1, 768)``) lets us skip the online DINOv2
    forward entirely.

    Semantics mirror ``DemonstrationSensor`` and ``InflectionWeightSensor``:
    on ``task._is_resetting`` we reset the internal step counter, load the
    matching feature file (through a tiny per-worker LRU), and return
    ``features[0]``. On subsequent steps we return ``features[timestep]``.
    If a file is missing we raise loudly rather than silently fabricating a
    zero tensor.
    """

    def __init__(self, config: Config, **kwargs):
        self._config = config
        self.uuid = "cached_dinov2_feature"
        self._feature_dim = int(config.FEATURE_DIM)
        self._cache_root = str(config.CACHE_ROOT)
        self._lru_size = int(getattr(config, "LRU_SIZE", 32))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._feature_dim,),
            dtype=np.float32,
        )
        self.timestep = 0
        self._features: Optional[np.ndarray] = None
        self._current_ep_id: Optional[str] = None
        self._lru: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _cache_path(self, scene_name: str, episode_id: str) -> str:
        return os.path.join(
            self._cache_root, scene_name, f"{episode_id}.pt"
        )

    def _load_features(self, scene_name: str, episode_id: str) -> np.ndarray:
        key = f"{scene_name}/{episode_id}"
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        path = self._cache_path(scene_name, episode_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "CachedDINOv2Sensor: missing feature file for "
                f"scene={scene_name!r} episode_id={episode_id!r} at {path!r}. "
                "Run scripts/precompute_dinov2_features.py first."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        feats = payload["dino_cls"] if isinstance(payload, dict) else payload
        if not isinstance(feats, torch.Tensor):
            raise TypeError(
                f"CachedDINOv2Sensor: expected torch.Tensor at {path!r}, "
                f"got {type(feats).__name__}"
            )
        feats_np = feats.detach().to(torch.float32).cpu().numpy()
        if feats_np.ndim != 2 or feats_np.shape[1] != self._feature_dim:
            raise ValueError(
                "CachedDINOv2Sensor: feature shape mismatch at "
                f"{path!r}: got {tuple(feats_np.shape)}, expected "
                f"(T+1, {self._feature_dim})"
            )

        self._lru[key] = feats_np
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)
        return feats_np

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs,
    ):
        if task._is_resetting:
            self.timestep = 0
            scene_name = _scene_id_to_name(episode.scene_id)
            self._features = self._load_features(scene_name, episode.episode_id)
            self._current_ep_id = episode.episode_id

        if self._features is None:
            raise RuntimeError(
                "CachedDINOv2Sensor queried before the first task reset."
            )

        idx = min(self.timestep, len(self._features) - 1)
        feat = self._features[idx]
        self.timestep += 1
        return feat.astype(np.float32, copy=False)

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="SemanticMapSensor")
class SemanticMapSensor(Sensor):
    """Egocentric semantic + occupancy map, padded to a fixed ``(MAP_H, MAP_W)``.

    Returns an agent-centered, agent-oriented ``(H, W) int8`` crop on every
    step (forward = up). Two modes, selected by ``CACHE_ROOT``:

    * ``CACHE_ROOT = ""`` (default) -- online mode. Maintains a per-worker
      :class:`SemanticMapper` that accumulates depth + semantic observations
      into a world-anchored label map across the lifetime of the current
      episode. Requires ``DEPTH_SENSOR`` and ``SEMANTIC_SENSOR`` enabled in
      ``SIMULATOR.AGENT_0.SENSORS`` (and ``DEPTH_SENSOR.NORMALIZE_DEPTH =
      False``). Map is reset whenever ``task._is_resetting`` fires; the
      per-scene instance->task-id table is rebuilt only when the scene
      changes.

    * ``CACHE_ROOT = "<path>"`` -- cached / oracle mode. Loads a precomputed
      world-frame label map once per scene from
      ``<CACHE_ROOT>/<scene_name>/<scene_name>.npz`` (the schema written by
      ``teleop_semantic_map.save_map``: ``global_map``, ``origin_x``,
      ``origin_z``, ``resolution``, ``scene_id``). Neither ``DEPTH_SENSOR``
      nor ``SEMANTIC_SENSOR`` agent-level rendering is needed; the policy
      sees the same ``(H, W) int8`` agent-frame crop as in online mode.

    The optional ``SMOOTH_K`` config applies a kxk majority-vote filter to the
    egocentric crop before returning (matches teleop's visualization smoothing
    so what the policy sees is identical to what was being inspected).
    """

    cls_uuid: str = "semantic_map"

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._H = int(config.MAP_H)
        self._W = int(config.MAP_W)
        self._res = float(config.MAP_RESOLUTION)
        self._smooth_k = int(getattr(config, "SMOOTH_K", 0))
        self._world_diameter = float(getattr(config, "WORLD_DIAMETER_M", 80.0))
        self._cache_root = str(getattr(config, "CACHE_ROOT", "") or "")

        self._mapper: Optional[SemanticMapper] = None
        self._instance_to_task: Optional[np.ndarray] = None
        self._cached_scene_id: Optional[str] = None

        self.uuid = self.cls_uuid
        self.observation_space = spaces.Box(
            low=-1,
            high=NUM_CHANNELS - 1,
            shape=(self._H, self._W),
            dtype=np.int8,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _depth_intrinsics(self):
        sim_cfg = self._sim.habitat_config
        depth_cfg = sim_cfg.DEPTH_SENSOR
        return (
            float(np.deg2rad(float(depth_cfg.HFOV))),
            float(depth_cfg.MIN_DEPTH),
            float(depth_cfg.MAX_DEPTH),
        )

    def _load_scene_map(self, scene_id_path: str) -> None:
        """Populate ``self._mapper`` from
        ``<cache_root>/<scene>/<scene>.npz``. Raises with a clear hint if the
        file is missing, malformed, or recorded at a different resolution
        than the sensor expects (which would silently rescale the metric
        window the policy is trained on).
        """
        scene_name = _scene_id_to_name(scene_id_path)
        path = os.path.join(self._cache_root, scene_name, f"{scene_name}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "SemanticMapSensor (cached mode): missing scene map for "
                f"scene={scene_name!r} at {path!r}. Produce it with "
                f"`python teleop_semantic_map.py --scene-id {scene_name} "
                f"--output-dir {os.path.join(self._cache_root, scene_name)}`."
            )
        data = np.load(path, allow_pickle=False)
        global_map = np.asarray(data["global_map"])
        if global_map.ndim != 2:
            raise ValueError(
                f"{path!r}: global_map must be 2-D; got {tuple(global_map.shape)}"
            )
        if global_map.dtype != np.int8:
            global_map = global_map.astype(np.int8)
        cached_res = float(data["resolution"])
        if abs(cached_res - self._res) > 1e-9:
            raise ValueError(
                f"{path!r}: cached map resolution {cached_res} m/cell does "
                f"not match SEMANTIC_MAP_SENSOR.MAP_RESOLUTION {self._res} "
                "m/cell; the trained policy expects a specific metric window."
            )
        self._mapper = SemanticMapper.from_cached(
            global_map=global_map,
            origin_x=float(data["origin_x"]),
            origin_z=float(data["origin_z"]),
            resolution=cached_res,
            H=self._H,
            W=self._W,
        )

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> np.ndarray:
        agent_state = self._sim.get_agent_state()

        if self._cache_root:
            if (
                task._is_resetting
                or self._mapper is None
                or episode.scene_id != self._cached_scene_id
            ):
                self._load_scene_map(episode.scene_id)
                self._cached_scene_id = episode.scene_id
            assert self._mapper is not None  # mypy
            crop = self._mapper.egocentric_view(
                agent_pos=np.asarray(agent_state.position, dtype=np.float64),
                agent_rot=agent_state.rotation,
            )
            if self._smooth_k > 1:
                crop = smooth_label_map(crop, self._smooth_k)
            return crop

        if task._is_resetting or self._mapper is None:
            start_pos = np.asarray(agent_state.position, dtype=np.float64)
            self._mapper = SemanticMapper(
                H=self._H,
                W=self._W,
                resolution=self._res,
                start_pos=start_pos,
                world_diameter_m=self._world_diameter,
            )
            if episode.scene_id != self._cached_scene_id:
                self._instance_to_task = build_instance_to_task_id(self._sim)
                self._cached_scene_id = episode.scene_id

        sensor_state = agent_state.sensor_states["depth"]
        hfov_rad, min_depth, max_depth = self._depth_intrinsics()

        self._mapper.update(
            depth_m=np.asarray(observations["depth"]),
            semantic=np.asarray(observations["semantic"]),
            sensor_pos=np.asarray(sensor_state.position, dtype=np.float64),
            sensor_rot=sensor_state.rotation,
            agent_y=float(agent_state.position[1]),
            hfov_rad=hfov_rad,
            instance_to_task=self._instance_to_task,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        crop = self._mapper.egocentric_view(
            agent_pos=np.asarray(agent_state.position, dtype=np.float64),
            agent_rot=agent_state.rotation,
        )
        if self._smooth_k > 1:
            crop = smooth_label_map(crop, self._smooth_k)
        return crop

    def get_observation(self, **kwargs: Any) -> np.ndarray:
        return self._get_observation(**kwargs)


@registry.register_sensor(name="EgoObjectCloudSensor")
class EgoObjectCloudSensor(Sensor):
    """Egocentric object cloud, padded to a fixed ``MAX_OBJECTS``.

    Returns an agent-frame ``(MAX_OBJECTS, 4)`` float32 packed array every
    step. Each row is ``[task_id, ex, ey, ez]`` for valid objects and
    ``[-1, 0, 0, 0]`` for padding. When more than ``MAX_OBJECTS`` are tracked
    we keep the closest by ego distance (the underlying world-frame storage
    stays intact in either mode).

    Two modes, selected by ``CACHE_ROOT``:

    * ``CACHE_ROOT = ""`` (default) -- online mode. Maintains a per-worker
      :class:`ObjectCloud` of mask-area-weighted depth-projected centroids
      across the current episode. Requires ``DEPTH_SENSOR`` and
      ``SEMANTIC_SENSOR`` enabled in ``SIMULATOR.AGENT_0.SENSORS`` and
      ``DEPTH_SENSOR.NORMALIZE_DEPTH = False``.

    * ``CACHE_ROOT = "<path>"`` -- cached mode. Loads a precomputed
      world-frame cloud once per scene from
      ``<CACHE_ROOT>/<scene_name>/<scene_name>.npz`` (the layout written by
      ``dump_scene_object_clouds.py`` and ``teleop_object_cloud.py``) and
      transforms it into the agent frame every step. Neither ``DEPTH_SENSOR``
      nor ``SEMANTIC_SENSOR`` agent-level rendering is needed; the policy
      sees the same observation as in online mode.
    """

    cls_uuid: str = "ego_object_cloud"

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_objects = int(config.MAX_OBJECTS)
        self._min_mask_pixels = int(getattr(config, "MIN_MASK_PIXELS", 100))
        self._cache_root = str(getattr(config, "CACHE_ROOT", "") or "")

        if not self._cache_root:
            depth_cfg = sim.habitat_config.DEPTH_SENSOR
            self._fx, self._fy, self._cx, self._cy = make_camera_intrinsics(
                int(depth_cfg.WIDTH), int(depth_cfg.HEIGHT), float(depth_cfg.HFOV)
            )
            self._min_depth = float(depth_cfg.MIN_DEPTH)
            self._max_depth = float(depth_cfg.MAX_DEPTH)

        self._cloud: Optional[ObjectCloud] = None
        self._instance_to_task: Optional[np.ndarray] = None
        self._cached_scene_id: Optional[str] = None

        self._world_obj_pos: Optional[np.ndarray] = None   # (N, 3) float32
        self._world_task_ids: Optional[np.ndarray] = None  # (N,)   int64

        self.uuid = self.cls_uuid
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._max_objects, 4),
            dtype=np.float32,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _load_scene_cloud(self, scene_id_path: str) -> None:
        """Populate ``_world_obj_pos`` / ``_world_task_ids`` from the per-scene
        npz at ``<cache_root>/<scene>/<scene>.npz``. Raises a clear error if
        the file is missing, malformed, or out of the expected range.
        """
        scene_name = _scene_id_to_name(scene_id_path)
        path = os.path.join(
            self._cache_root, scene_name, f"{scene_name}.npz"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "EgoObjectCloudSensor (cached mode): missing scene cloud for "
                f"scene={scene_name!r} at {path!r}. Run "
                "`python dump_scene_object_clouds.py "
                f"--output-dir {self._cache_root}` to populate it."
            )
        data = np.load(path)  # allow_pickle=False; we never read 'labels'
        obj_pos = np.asarray(data["obj_pos"], dtype=np.float32)
        task_ids = np.asarray(data["task_ids"], dtype=np.int64)
        if obj_pos.ndim != 2 or obj_pos.shape[1] != 3:
            raise ValueError(
                f"{path!r}: obj_pos must be (N, 3); got {tuple(obj_pos.shape)}"
            )
        if task_ids.shape != (obj_pos.shape[0],):
            raise ValueError(
                f"{path!r}: task_ids shape {tuple(task_ids.shape)} does not "
                f"match obj_pos N={obj_pos.shape[0]}"
            )
        if obj_pos.size and (task_ids.min() < 0 or task_ids.max() >= NUM_CATEGORIES):
            raise ValueError(
                f"{path!r}: task_ids out of range [0, {NUM_CATEGORIES - 1}]; "
                f"saw [{int(task_ids.min())}, {int(task_ids.max())}]"
            )
        self._world_obj_pos = obj_pos
        self._world_task_ids = task_ids

    def _pack_ego(
        self, pos_ego: np.ndarray, task_ids: np.ndarray
    ) -> np.ndarray:
        """Closest-by-ego-distance prune to ``MAX_OBJECTS`` and pack into the
        ``(MAX_OBJECTS, 4)`` observation; padding rows are ``[-1, 0, 0, 0]``.
        """
        packed = np.zeros((self._max_objects, 4), dtype=np.float32)
        packed[:, 0] = -1.0
        n = int(pos_ego.shape[0])
        if n == 0:
            return packed
        pos = pos_ego.astype(np.float32, copy=False)
        tids = task_ids.astype(np.float32, copy=False)
        if n > self._max_objects:
            d2 = np.einsum("ij,ij->i", pos, pos)
            keep = np.argpartition(d2, self._max_objects)[: self._max_objects]
            pos, tids = pos[keep], tids[keep]
            n = self._max_objects
        packed[:n, 0] = tids
        packed[:n, 1:4] = pos
        return packed

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> np.ndarray:
        agent_state = self._sim.get_agent_state()
        agent_pos = np.asarray(agent_state.position, dtype=np.float32)
        R = quaternion.as_rotation_matrix(agent_state.rotation).astype(np.float32)

        if self._cache_root:
            if (
                task._is_resetting
                or self._world_obj_pos is None
                or episode.scene_id != self._cached_scene_id
            ):
                self._load_scene_cloud(episode.scene_id)
                self._cached_scene_id = episode.scene_id
            assert self._world_obj_pos is not None  # mypy
            pos_ego = (self._world_obj_pos - agent_pos) @ R
            return self._pack_ego(pos_ego, self._world_task_ids)

        if task._is_resetting or self._cloud is None:
            if episode.scene_id != self._cached_scene_id:
                self._instance_to_task = build_instance_to_task_id(self._sim)
                self._cached_scene_id = episode.scene_id
            self._cloud = ObjectCloud(
                self._instance_to_task,
                min_mask_pixels=self._min_mask_pixels,
            )

        sensor_state = agent_state.sensor_states["depth"]
        self._cloud.update(
            depth_m=np.asarray(observations["depth"]),
            semantic=np.asarray(observations["semantic"]),
            sensor_pos=np.asarray(sensor_state.position, dtype=np.float64),
            sensor_rot=sensor_state.rotation,
            fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy,
            depth_min=self._min_depth, depth_max=self._max_depth,
        )
        ego = self._cloud.to_ego_dict(
            agent_pos=np.asarray(agent_state.position, dtype=np.float64),
            agent_rot=agent_state.rotation,
        )
        return self._pack_ego(ego["obj_pos"], ego["task_ids"])

    def get_observation(self, **kwargs: Any) -> np.ndarray:
        return self._get_observation(**kwargs)


@registry.register_sensor(name="GoalCompassSensor")
class GoalCompassSensor(Sensor):
    r"""Oracle 12-bin goal-direction compass feature.

    Mirrors the algorithm in ``global_test.py::compute_compass_feature``:
    for each goal instance in ``episode.goals``, computes a ground-plane
    bearing from the agent's forward direction, and adds a rectified,
    distance-weighted cosine contribution to each of ``NUM_BINS`` bins spaced
    uniformly around the agent.

        score_i += max(0, cos(bin_i - bearing)) / (1 + distance)

    Bin 0 points along agent-forward; bin index increases counterclockwise
    (agent's left, viewed from above). All math is done on the world x-z
    ground plane using Habitat's ``[0, 0, -1]`` forward convention.

    Requires access to the simulator (for agent pose) and the episode (for
    goal positions), so it is only available where ``episode.goals`` is
    populated (true for ObjectNav-v2 train/val/eval replays).
    """

    cls_uuid: str = "goal_compass"

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._n_bins = int(getattr(config, "NUM_BINS", 12))
        self._bin_angles = (
            np.arange(self._n_bins, dtype=np.float32)
            * (2.0 * np.pi / self._n_bins)
        )
        self._forward_local = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        self.uuid = self.cls_uuid
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self._n_bins,),
            dtype=np.float32,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        agent_state = self._sim.get_agent_state()
        agent_pos = np.asarray(agent_state.position, dtype=np.float64)
        agent_rot = agent_state.rotation

        R = quaternion.as_rotation_matrix(agent_rot)
        fwd_world = R @ self._forward_local
        fwd_xz = np.array([fwd_world[0], fwd_world[2]], dtype=np.float64)
        fwd_norm = np.linalg.norm(fwd_xz)
        if fwd_norm < 1e-9:
            # Looking straight up/down -> no meaningful ground heading.
            return np.zeros(self._n_bins, dtype=np.float32)
        fwd_xz /= fwd_norm

        goals = getattr(episode, "goals", None) or []
        compass = np.zeros(self._n_bins, dtype=np.float32)
        for goal in goals:
            goal_pos = np.asarray(goal.position, dtype=np.float64)
            delta_xz = np.array(
                [goal_pos[0] - agent_pos[0], goal_pos[2] - agent_pos[2]],
                dtype=np.float64,
            )
            d = float(np.linalg.norm(delta_xz))
            if d < 1e-6:
                continue
            goal_dir = delta_xz / d

            # Signed angle from agent-forward to goal-direction in the x-z
            # plane. Sign-flipped from a raw 2D cross so +bearing = agent's
            # left (CCW about +y viewed from above), matching global_test.py.
            cross = fwd_xz[1] * goal_dir[0] - fwd_xz[0] * goal_dir[1]
            dot = fwd_xz[0] * goal_dir[0] + fwd_xz[1] * goal_dir[1]
            bearing = float(np.arctan2(cross, dot))

            sims = np.clip(np.cos(self._bin_angles - bearing), 0.0, None)
            compass += (sims / (1.0 + d)).astype(np.float32)

        return compass


@registry.register_sensor(name="InflectionWeightSensor")
class InflectionWeightSensor(Sensor):
    def __init__(self, config: Config, **kwargs):
        self.uuid = "inflection_weight"
        self.observation_space = spaces.Discrete(1)
        self._config = config
        self.timestep = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        if task._is_resetting:  # reset
            self.timestep = 0

        inflection_weight = 1.0
        if self.timestep == 0:
            inflection_weight = 1.0
        elif self.timestep >= len(episode.reference_replay):
            inflection_weight = 1.0
        elif (
            episode.reference_replay[self.timestep - 1].action
            != episode.reference_replay[self.timestep].action
        ):
            inflection_weight = self._config.INFLECTION_COEF
        self.timestep += 1
        return inflection_weight

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)
