#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pose-replay teleport action used by :class:`ILEnvDDPTrainer`.

Distinct from habitat-lab's stock :class:`TeleportAction` (in
``habitat/tasks/nav/nav.py``) in two ways:

* No ``sim.is_navigable`` gate.  Recorded demo poses come from the sim's
  own ``agent_state`` after each teleop step, so they are on-navmesh by
  construction; a navigability point-query can still spuriously reject a
  pose that sits on a floating-point-edge navmesh boundary, in which case
  habitat-lab's action silently leaves the agent in place and the
  trajectory desyncs from the recorded demonstration.

* Loud failure handling.  ``set_agent_state`` is expected to succeed for
  every recorded pose; a failure during IL rollout collection corrupts
  the alignment between the policy's observations and the BC labels for
  every subsequent step in the trajectory, so we log a warning rather
  than swallow it.

Rotation is consumed in ``[x, y, z, w]`` order, matching both the JSON
layout stored in ``episode.reference_replay[t].agent_state.rotation`` and
the convention expected by :meth:`HabitatSim.set_agent_state`.
"""

from typing import Any, Sequence

import numpy as np
from gym import spaces
from habitat import logger
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry


@registry.register_task_action
class ReplayTeleportAction(SimulatorTaskAction):
    name: str = "REPLAY_TELEPORT"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.name

    def step(
        self,
        *args: Any,
        position: Sequence[float],
        rotation: Sequence[float],
        **kwargs: Any,
    ):
        position = list(position)
        rotation = list(rotation)

        obs = self._sim.get_observations_at(
            position=position,
            rotation=rotation,
            keep_agent_at_new_pose=True,
        )
        if obs is None:
            logger.warning(
                "ReplayTeleportAction: set_agent_state failed for "
                "position=%s rotation=%s; falling back to current pose. "
                "This will desync the BC labels for the remainder of the "
                "episode.",
                position,
                rotation,
            )
            obs = self._sim.get_observations_at()
        return obs

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )
