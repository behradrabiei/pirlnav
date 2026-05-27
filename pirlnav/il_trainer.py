#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from gym import spaces
from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame
from pirlnav.utils.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter, get_writer
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
    linear_decay,
)
from torch import nn as nn
from torch.optim.lr_scheduler import LambdaLR

from pirlnav.algos.agent import DDPILAgent
from pirlnav.common.rollout_storage import RolloutStorage


def _drop_replay_teleport(action_space):
    """Return a copy of ``action_space`` with ``REPLAY_TELEPORT`` removed.

    The IL trainer appends ``REPLAY_TELEPORT`` to ``TASK.POSSIBLE_ACTIONS``
    only as an internal teleport hook for rollout collection; the policy
    never outputs it.  Hiding it from the action space the policy is built
    against keeps ``is_continuous_action_space`` returning False, keeps
    ``CategoricalNet`` / ``prev_action_embedding`` sized to the 6 discrete
    nav actions, and ensures checkpoints stay shape-compatible across
    pose-replay training, eval, and downstream RL fine-tuning.
    """
    from habitat.core.spaces import ActionSpace

    spaces_dict = getattr(action_space, "spaces", None)
    if spaces_dict is None or "REPLAY_TELEPORT" not in spaces_dict:
        return action_space
    return ActionSpace(
        {k: v for k, v in spaces_dict.items() if k != "REPLAY_TELEPORT"}
    )


@baseline_registry.register_trainer(name="pirlnav-il")
class ILEnvDDPTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)

    def _setup_actor_critic_agent(self, il_cfg: Config) -> None:
        r"""Sets up actor critic and agent for IL.

        Args:
            il_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        observation_space = self.envs.observation_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space

        policy = baseline_registry.get_policy(self.config.IL.POLICY.name)
        # Use the (REPLAY_TELEPORT-filtered) policy_action_space so the
        # discrete-action policy stays sized to the 6 nav actions in
        # pose-replay mode.  REPLAY_TELEPORT is dispatched internally by
        # _compute_actions_and_step_envs and never output by the policy.
        policy_action_space = _drop_replay_teleport(
            getattr(self, "policy_action_space", self.envs.action_spaces[0])
        )
        self.actor_critic = policy.from_config(
            self.config, observation_space, policy_action_space
        )
        self.actor_critic.to(self.device)

        self.agent = DDPILAgent(
            actor_critic=self.actor_critic,
            num_envs=self.envs.num_envs,
            num_mini_batch=il_cfg.num_mini_batch,
            lr=il_cfg.lr,
            encoder_lr=il_cfg.encoder_lr,
            eps=il_cfg.eps,
            max_grad_norm=il_cfg.max_grad_norm,
            wd=il_cfg.wd,
            entropy_coef=il_cfg.entropy_coef,
            compass_aux_coef=getattr(il_cfg, "compass_aux_coef", 0.0),
        )

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        # Add replay sensors.  In pose-replay mode we also surface the
        # recorded next-step agent_state as an observation and register
        # the REPLAY_TELEPORT task action used by
        # _compute_actions_and_step_envs to advance the env without
        # stepping sim physics.  All extensions are idempotent so resume
        # from a checkpoint (whose config already carries these entries)
        # is a no-op rather than producing duplicates.
        self._replay_mode = self.config.IL.BehaviorCloning.REPLAY_MODE
        assert self._replay_mode in ("poses", "actions"), (
            f"IL.BehaviorCloning.REPLAY_MODE must be 'poses' or 'actions',"
            f" got {self._replay_mode!r}"
        )
        self.config.defrost()
        sensors_list = self.config.TASK_CONFIG.TASK.SENSORS
        extra_sensors = ["DEMONSTRATION_SENSOR", "INFLECTION_WEIGHT_SENSOR"]
        if self._replay_mode == "poses":
            extra_sensors.append("NEXT_POSE_SENSOR")
        for name in extra_sensors:
            if name not in sensors_list:
                sensors_list.append(name)
        if (
            self._replay_mode == "poses"
            and "REPLAY_TELEPORT"
            not in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        ):
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.append(
                "REPLAY_TELEPORT"
            )
        self.config.freeze()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        action_space = _drop_replay_teleport(self.envs.action_spaces[0])
        self.policy_action_space = action_space

        if is_continuous_action_space(action_space):
            # Assume ALL actions are NOT discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = None
            discrete_actions = True

        il_cfg = self.config.IL.BehaviorCloning
        policy_cfg = self.config.POLICY
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(il_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if il_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            il_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            policy_cfg.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=il_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=il_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # fetch actions from replay buffer
        step_batch = self.rollouts.buffers[
            self.rollouts.current_rollout_step_idxs[buffer_index],
            env_slice,
        ]
        next_actions = step_batch["observations"]["next_actions"]
        actions = next_actions.long().unsqueeze(-1)

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")

        # In pose-replay mode we also need the next-step recorded agent
        # pose; pull it once per buffer and dispatch a REPLAY_TELEPORT
        # task action per env (except on the terminal STOP, which we
        # forward unchanged so habitat-lab's StopAction can flip
        # is_stop_called -> _episode_over and let the worker auto-reset).
        next_poses_cpu = None
        if self._replay_mode == "poses":
            next_poses_cpu = (
                step_batch["observations"]["next_pose"].to(device="cpu").numpy()
            )

        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for buffer_pos, (index_env, act) in enumerate(
            zip(range(env_slice.start, env_slice.stop), actions.unbind(0))
        ):
            if self._replay_mode == "poses" and act.item() != 0:
                # NB: outer "action" wrapping matches the convention used by
                # VectorEnv.async_step_at for int/str payloads (see
                # habitat-lab/habitat/core/vector_env.py); the worker does
                # ``env.step(**data)`` so the inner dict is what reaches
                # Env.step as its ``action`` argument.
                pose = next_poses_cpu[buffer_pos]
                step_action = {
                    "action": {
                        "action": "REPLAY_TELEPORT",
                        "action_args": {
                            "position": pose[:3].tolist(),
                            "rotation": pose[3:].tolist(),
                        },
                    }
                }
            elif act.shape[0] > 1:
                step_action = action_array_to_dict(
                    self.policy_action_space, act
                )
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            actions=actions,
            buffer_index=buffer_index,
        )

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        t_update_model = time.time()

        self.agent.train()

        (
            action_loss,
            rnn_hidden_states,
            dist_entropy,
            _,
            compass_loss,
        ) = self.agent.update(self.rollouts)

        self.rollouts.after_update(rnn_hidden_states)
        self.pth_time += time.time() - t_update_model

        return (
            action_loss,
            dist_entropy,
            compass_loss,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO
        il_cfg = self.config.IL.BehaviorCloning

        with (
            get_writer(self.config, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if il_cfg.use_linear_clip_decay:
                    self.agent.clip_param = il_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(il_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == il_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    action_loss,
                    dist_entropy,
                    compass_loss,
                ) = self._update_agent()

                if il_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        action_loss=action_loss,
                        entropy=dist_entropy,
                        compass_loss=compass_loss,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                    "  ".join(
                        "{}: {:.3f}".format(k, v) for k, v in losses.items()
                    ),
                )
            )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        # Upstream forced ObjectNav-v1 here, which cannot parse v2-only fields
        # like `reference_replay`.  The v2 loader also populates per-episode
        # `goals` from `goals_by_category`, and DemonstrationSensor is not
        # attached in eval mode, so v2 works uniformly for both train/eval.
        # Preserve whatever was configured so val datasets with reference_replay
        # still load.
        config.freeze()

        # Seed python / numpy / torch RNGs so random augmentations applied by
        # the visual transform (e.g. RandomShiftsAug, ColorJitter when
        # POLICY.RGB_ENCODER.use_augmentations_test_time=True) are reproducible
        # across process launches.  Habitat seeds the simulator separately via
        # TASK_CONFIG.SEED; this covers the policy-side RNG that was otherwise
        # unseeded in single-GPU (non-distributed) eval.
        eval_seed = int(config.TASK_CONFIG.SEED)
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)

        if (
            len(self.config.VIDEO_OPTION) > 0
            and self.config.VIDEO_RENDER_TOP_DOWN
        ):
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        # Defensive: if the merged eval config still carries
        # REPLAY_TELEPORT in POSSIBLE_ACTIONS (e.g. someone forced
        # USE_CKPT_CONFIG with a custom merge), strip it from the
        # policy_action_space so the policy we construct here matches the
        # shape of the saved state_dict (which was trained with the action
        # filtered out).
        action_space = _drop_replay_teleport(self.envs.action_spaces[0])
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume NONE of the actions are discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = (1,)
                discrete_actions = True

        il_cfg = config.IL.BehaviorCloning
        policy_cfg = config.POLICY
        self._setup_actor_critic_agent(il_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            self.agent.load_state_dict({
                k.replace("model.", "actor_critic."): v
                for k, v in ckpt_dict["state_dict"].items()
            })
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            policy_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        logger.info("Sampling actions deterministically...")
        self.actor_critic.eval()
        # Track pure policy-forward time (excluding sim/env time) so we can
        # report an average per-step inference latency at the end of eval.
        infer_time_total = 0.0
        infer_steps = 0
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t_act = time.time()
                (
                    actions,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_time_total += time.time() - _t_act
                infer_steps += self.envs.num_envs

                # Snapshot any auxiliary tensors the policy produced on this
                # forward (e.g. compass_pred for the compass-aux variants) so
                # we can render them in the eval video without re-running the
                # net.  Empty dict for policies that don't populate _last_aux.
                aux_snapshot: Dict[str, torch.Tensor] = {
                    k: v.detach()
                    for k, v in self.actor_critic.get_last_aux().items()
                }

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if actions[0].shape[0] > 1:
                step_data = [
                    action_array_to_dict(self.policy_action_space, a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        # Optionally save videos only for failed episodes.
                        # `success` is a 0/1 float coming from habitat's Success
                        # measure; treat >0.5 as success.
                        is_success = bool(
                            episode_stats.get("success", 0.0) > 0.5
                        )
                        failed_only = getattr(
                            self.config.EVAL, "VIDEO_FAILED_ONLY", False
                        )
                        if (not failed_only) or (not is_success):
                            generate_video(
                                video_option=self.config.VIDEO_OPTION,
                                video_dir=self.config.VIDEO_DIR,
                                images=rgb_frames[i],
                                episode_id=current_episodes[i].episode_id,
                                checkpoint_idx=checkpoint_index,
                                metrics=self._extract_scalars_from_info(
                                    infos[i]
                                ),
                                fps=self.config.VIDEO_FPS,
                                tb_writer=writer,
                                keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,
                            )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    per_env_obs = {k: v[i] for k, v in batch.items()}
                    # Inject the policy's compass prediction as a pseudo-
                    # observation so observations_to_image can render it as
                    # an extra panel.  NOTE: compass_pred corresponds to the
                    # pre-step batch that drove act(), while the rest of the
                    # frame is built from the post-step batch.  The resulting
                    # one-step lag is visually negligible (per-step pose
                    # changes are 30deg or 0.25m) and avoids an extra forward
                    # pass per frame.
                    compass_pred_batch = aux_snapshot.get("compass_pred")
                    if compass_pred_batch is not None and i < compass_pred_batch.shape[0]:
                        per_env_obs["compass_pred"] = compass_pred_batch[i]
                    frame = observations_to_image(per_env_obs, infos[i])
                    if self.config.VIDEO_RENDER_ALL_INFO:
                        frame = overlay_frame(frame, infos[i])

                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        aggregated_stats["avg_inference_time_ms"] = (
            (infer_time_total / max(infer_steps, 1)) * 1000.0
        )
        aggregated_stats["num_eval_episodes"] = float(num_episodes)
        aggregated_stats["num_policy_steps"] = float(infer_steps)

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
