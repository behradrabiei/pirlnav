import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch
from gym import spaces
from habitat import logger
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor
from habitat.sims.habitat_simulator.actions import HabitatSimActions


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
