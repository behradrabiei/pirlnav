from typing import Dict

import torch
import torch.nn as nn
from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net

from pirlnav.policy.dinov2_encoder import DINOv2VisualEncoder
from pirlnav.policy.object_cloud_encoder import ObjectCloudEncoder
from pirlnav.policy.policy import ILPolicy
from pirlnav.policy.semantic_map_encoder import SemanticMapEncoder
from pirlnav.policy.transforms import get_transform
from pirlnav.policy.visual_encoder import VisualEncoder
from pirlnav.task.semantic_map import (
    HM3D6_TO_OBJECT_CLOUD_TASK,
    NUM_CATEGORIES,
)
from pirlnav.utils.utils import load_encoder


class ObjectNavILMAENet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
        self,
        observation_space: Space,
        policy_config: Config,
        num_actions: int,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__()
        self.policy_config = policy_config
        rnn_input_size = 0

        rgb_config = policy_config.RGB_ENCODER
        self._is_dinov2 = rgb_config.backbone == "dinov2_base"

        if self._is_dinov2:
            size_hw = (
                rgb_config.dinov2_resize_h,
                rgb_config.dinov2_resize_w,
            )
            aug_name = "dinov2"
            use_aug = (
                (rgb_config.use_augmentations and run_type == "train")
                or (
                    rgb_config.use_augmentations_test_time
                    and run_type == "eval"
                )
            )
            if use_aug:
                aug_name = f"dinov2+{rgb_config.augmentations_name}"
            self.visual_transform = get_transform(aug_name, size_hw=size_hw)
            self.visual_transform.randomize_environments = (
                rgb_config.randomize_augmentations_over_envs
            )

            self.visual_encoder = DINOv2VisualEncoder(
                model_name=rgb_config.dinov2_model_name,
                resize_hw=size_hw,
                output_dim=rgb_config.dinov2_output_dim,
            )
        else:
            name = "resize"
            if rgb_config.use_augmentations and run_type == "train":
                name = rgb_config.augmentations_name
            if rgb_config.use_augmentations_test_time and run_type == "eval":
                name = rgb_config.augmentations_name
            self.visual_transform = get_transform(name, size=rgb_config.image_size)
            self.visual_transform.randomize_environments = (
                rgb_config.randomize_augmentations_over_envs
            )

            self.visual_encoder = VisualEncoder(
                image_size=rgb_config.image_size,
                backbone=rgb_config.backbone,
                input_channels=3,
                resnet_baseplanes=rgb_config.resnet_baseplanes,
                resnet_ngroups=rgb_config.resnet_baseplanes // 2,
                avgpooled_image=rgb_config.avgpooled_image,
                drop_path_rate=rgb_config.drop_path_rate,
            )

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_encoder.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )

        rnn_input_size += policy_config.RGB_ENCODER.hidden_size
        logger.info(
            "RGB encoder is {}".format(policy_config.RGB_ENCODER.backbone)
        )

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(
                input_compass_dim, self.compass_embedding_dim
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            logger.info(
                "Object categories: {}".format(self._n_object_categories)
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        # Compass auxiliary loss flag (read once; encoder + oracle path
        # below both depend on it).  When True, the point-transformer's
        # cls_head outputs 12-D for supervision against GoalCompassSensor,
        # the oracle goal_compass is *not* wired into the GRU, and the
        # encoder is goal-conditioned (see ObjectCloudEncoder).
        oc_cfg_pre = getattr(policy_config, "OBJECT_CLOUD_ENCODER", None)
        self._compass_aux_loss = bool(
            getattr(oc_cfg_pre, "compass_aux_loss", False)
            if oc_cfg_pre is not None
            else False
        )

        # Optional 12-bin goal-direction compass (oracle feature from
        # GoalCompassSensor).  Auto-activates iff the sensor is listed in
        # TASK.SENSORS so the observation-space carries a "goal_compass" key
        # *and* compass_aux_loss is off (in aux-loss mode the oracle is a
        # training label only, never fed into the GRU).
        self._goal_compass_uuid = "goal_compass"
        if (
            self._goal_compass_uuid in observation_space.spaces
            and not self._compass_aux_loss
        ):
            gc_input_dim = observation_space.spaces[
                self._goal_compass_uuid
            ].shape[0]
            gc_cfg = getattr(policy_config, "GOAL_COMPASS_ENCODER", None)
            gc_embed_dim = (
                int(gc_cfg.embedding_size) if gc_cfg is not None else 32
            )
            self.goal_compass_embedding_dim = gc_embed_dim
            self.goal_compass_embedding = nn.Linear(gc_input_dim, gc_embed_dim)
            rnn_input_size += gc_embed_dim
            logger.info(
                "\n\nSetting up Goal Compass sensor ({} bins -> {} dim)".format(
                    gc_input_dim, gc_embed_dim
                )
            )

        # Optional egocentric semantic+occupancy map (from SemanticMapSensor).
        # Auto-activates iff the sensor is listed in TASK.SENSORS so the
        # observation-space carries a "semantic_map" key.  Drop-in replacement
        # for the goal_compass slot in the GRU input concat; the two can
        # coexist if both sensors are listed, but in practice you list one.
        self._semantic_map_uuid = "semantic_map"
        if self._semantic_map_uuid in observation_space.spaces:
            map_space = observation_space.spaces[self._semantic_map_uuid]
            map_h, map_w = int(map_space.shape[0]), int(map_space.shape[1])
            assert map_h == map_w, (
                "SemanticMapEncoder expects a square map; got "
                f"({map_h}, {map_w})"
            )
            map_cfg = getattr(policy_config, "MAP_ENCODER", None)
            map_embed_dim = (
                int(map_cfg.embedding_size) if map_cfg is not None else 32
            )
            map_backbone = (
                str(map_cfg.backbone)
                if map_cfg is not None and hasattr(map_cfg, "backbone")
                else "resnet18"
            )
            self.map_encoder = SemanticMapEncoder(
                image_size=map_h,
                output_dim=map_embed_dim,
                backbone=map_backbone,
            )
            rnn_input_size += map_embed_dim
            logger.info(
                "\n\nSetting up Semantic Map sensor "
                "({}x{} -> {} dim, backbone={})".format(
                    map_h, map_w, map_embed_dim, map_backbone
                )
            )

        # Optional egocentric object cloud (from EgoObjectCloudSensor).
        # Auto-activates iff the sensor is listed in TASK.SENSORS so the
        # observation-space carries an "ego_object_cloud" key.  Drop-in
        # alternative to the semantic-map slot in the GRU input concat.
        self._object_cloud_uuid = "ego_object_cloud"
        if self._object_cloud_uuid in observation_space.spaces:
            oc_space = observation_space.spaces[self._object_cloud_uuid]
            assert (
                len(oc_space.shape) == 2 and oc_space.shape[1] == 4
            ), (
                "ObjectCloudEncoder expects a (MAX_OBJECTS, 4) packed "
                f"sensor; got shape {tuple(oc_space.shape)}"
            )
            max_obj = int(oc_space.shape[0])
            oc_cfg = getattr(policy_config, "OBJECT_CLOUD_ENCODER", None)
            assert oc_cfg is not None, (
                "POLICY.OBJECT_CLOUD_ENCODER must be configured when "
                "EGO_OBJECT_CLOUD_SENSOR is enabled."
            )
            oc_embed_dim = int(oc_cfg.embedding_size)
            # In compass-aux mode the encoder emits a 12-D compass prediction
            # which is (a) supervised against GoalCompassSensor inside
            # ILAgent.update and (b) projected to oc_embed_dim by compass_proj
            # before entering the GRU concat.  The GRU input slot is the
            # same oc_embed_dim either way, so rnn_input_size is unchanged.
            inner_out_dim = 12 if self._compass_aux_loss else oc_embed_dim
            self.object_cloud_encoder = ObjectCloudEncoder(
                num_classes=int(oc_cfg.num_classes),
                d_model=int(oc_cfg.d_model),
                out_dim=inner_out_dim,
                num_layers=int(oc_cfg.num_layers),
                rpe_mode=str(oc_cfg.rpe_mode),
                ffn_expansion=int(oc_cfg.ffn_expansion),
                use_goal_conditioning=self._compass_aux_loss,
            )
            if self._compass_aux_loss:
                self.compass_proj = nn.Linear(12, oc_embed_dim)
            if (
                self._compass_aux_loss
                and getattr(self, "_n_object_categories", None)
                == len(HM3D6_TO_OBJECT_CLOUD_TASK)
                and int(oc_cfg.num_classes) == NUM_CATEGORIES
            ):
                self.register_buffer(
                    "_oc_goal_remap",
                    torch.tensor(HM3D6_TO_OBJECT_CLOUD_TASK, dtype=torch.long),
                    persistent=False,
                )
            else:
                self._oc_goal_remap = None
            rnn_input_size += oc_embed_dim
            logger.info(
                "\n\nSetting up Ego Object Cloud sensor "
                "(MAX_OBJECTS={} -> {} dim{})".format(
                    max_obj,
                    oc_embed_dim,
                    ", compass-aux 12-D head + goal conditioning"
                    if self._compass_aux_loss
                    else "",
                )
            )

        if policy_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.rnn_input_size = rnn_input_size

        # load pretrained weights
        if not self._is_dinov2 and rgb_config.pretrained_encoder is not None:
            msg = load_encoder(
                self.visual_encoder, rgb_config.pretrained_encoder
            )
            logger.info(
                "Using weights from {}: {}".format(
                    rgb_config.pretrained_encoder, msg
                )
            )

        # freeze backbone (DINOv2 is already frozen inside the encoder)
        if not self._is_dinov2 and rgb_config.freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False

        logger.info(
            "State enc: {} - {} - {} - {}".format(
                rnn_input_size, hidden_size, rnn_type, num_recurrent_layers
            )
        )

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self._hidden_size = hidden_size
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        N = rnn_hidden_states.size(1)

        # Per-forward auxiliary tensor stash (read by ILPolicy.get_last_aux()
        # and consumed by ILAgent.update when compass_aux_coef > 0).  Cleared
        # at every entry so stale values from prior calls cannot leak.
        self._last_aux: Dict[str, torch.Tensor] = {}

        x = []

        cached_uuid = "cached_dinov2_feature"
        use_cached = (
            self._is_dinov2
            and cached_uuid in observations
            and observations[cached_uuid] is not None
        )

        if use_cached:
            rgb_feat = observations[cached_uuid]
            if rgb_feat.dim() == 3:
                rgb_feat = rgb_feat.contiguous().view(-1, rgb_feat.size(-1))
            target_dtype = next(self.visual_fc.parameters()).dtype
            rgb = self.visual_fc(rgb_feat.to(dtype=target_dtype))
            x.append(rgb)
        elif self.visual_encoder is not None:
            rgb_obs = observations["rgb"]
            if len(rgb_obs.size()) == 5:
                observations["rgb"] = rgb_obs.contiguous().view(
                    -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
                )
            rgb = observations["rgb"]

            rgb = self.visual_transform(rgb, N)
            rgb = self.visual_encoder(rgb)
            rgb = self.visual_fc(rgb)
            x.append(rgb)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(
                    -1, obs_compass.size(2)
                )
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(
                compass_observations.float().squeeze(dim=1)
            )
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(
                    -1, object_goal.size(2)
                )
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if (
            self._goal_compass_uuid in observations
            and hasattr(self, "goal_compass_embedding")
        ):
            obs_gc = observations[self._goal_compass_uuid]
            if obs_gc.dim() == 3:
                obs_gc = obs_gc.contiguous().view(-1, obs_gc.size(2))
            x.append(self.goal_compass_embedding(obs_gc.float()))

        if (
            self._semantic_map_uuid in observations
            and hasattr(self, "map_encoder")
        ):
            obs_map = observations[self._semantic_map_uuid]
            if obs_map.dim() == 4:  # (T, N, H, W) -> (T*N, H, W)
                obs_map = obs_map.contiguous().view(
                    -1, obs_map.size(2), obs_map.size(3)
                )
            x.append(self.map_encoder(obs_map))

        if (
            self._object_cloud_uuid in observations
            and hasattr(self, "object_cloud_encoder")
        ):
            obs_oc = observations[self._object_cloud_uuid]
            if obs_oc.dim() == 4:  # (T, N, MAX, 4) -> (T*N, MAX, 4)
                obs_oc = obs_oc.contiguous().view(
                    -1, obs_oc.size(2), obs_oc.size(3)
                )
            if self._compass_aux_loss:
                # reshape(-1) flattens (N, 1), (T*N, 1), or (T, N, 1) into
                # a flat (B,) tensor matching obs_oc's leading dim.
                goal_class = (
                    observations[ObjectGoalSensor.cls_uuid].long().reshape(-1)
                )
                if self._oc_goal_remap is not None:
                    goal_class = self._oc_goal_remap[goal_class]
                compass_pred = self.object_cloud_encoder(
                    obs_oc, goal_class=goal_class
                )
                self._last_aux["compass_pred"] = compass_pred
                x.append(self.compass_proj(compass_pred))
            else:
                x.append(self.object_cloud_encoder(obs_oc))

        if self.policy_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)

        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states.contiguous(), masks
        )

        return x, rnn_hidden_states


@baseline_registry.register_policy
class ObjectNavILMAEPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        policy_config: Config,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__(
            ObjectNavILMAENet(
                observation_space=observation_space,
                policy_config=policy_config,
                num_actions=action_space.n,
                run_type=run_type,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
            ),
            action_space.n,
            no_critic=policy_config.CRITIC.no_critic,
            mlp_critic=policy_config.CRITIC.mlp_critic,
            critic_hidden_dim=policy_config.CRITIC.hidden_dim,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.POLICY,
            run_type=config.RUN_TYPE,
            hidden_size=config.POLICY.STATE_ENCODER.hidden_size,
            rnn_type=config.POLICY.STATE_ENCODER.rnn_type,
            num_recurrent_layers=config.POLICY.STATE_ENCODER.num_recurrent_layers,
        )

    @property
    def num_recurrent_layers(self):
        return self.net.num_recurrent_layers

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        # DINOv2 is permanently frozen; leave it alone.
        if getattr(self.net, "_is_dinov2", False):
            return
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(True)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)
