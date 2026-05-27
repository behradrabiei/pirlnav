import warnings
from typing import List, Optional, Union

from habitat.config.default import _C as _HABITAT_CONFIG
from habitat.config.default import Config as CN
from habitat_baselines.config.default import _C as _BASE_CONFIG

CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------

# fmt:off
_TASK_CONFIG = _HABITAT_CONFIG.clone()
_TASK_CONFIG.defrost()

_TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = 10000

_TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
_TASK_CONFIG.SIMULATOR.TURN_ANGLE = 30
_TASK_CONFIG.SIMULATOR.TURN_ANGLE = 30
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 128
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 128
_TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]

_TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]

# Optional sensor that surfaces precomputed frozen-DINOv2 CLS features for the
# current step of the expert reference_replay.  Only activated by adding
# "CACHED_DINOV2_SENSOR" to TASK.SENSORS in a task yaml; the policy then skips
# its online DINOv2 forward and feeds this directly into visual_fc.
_TASK_CONFIG.TASK.CACHED_DINOV2_SENSOR = CN()
_TASK_CONFIG.TASK.CACHED_DINOV2_SENSOR.TYPE = "CachedDINOv2Sensor"
_TASK_CONFIG.TASK.CACHED_DINOV2_SENSOR.CACHE_ROOT = "data/dinov2_cache"
_TASK_CONFIG.TASK.CACHED_DINOV2_SENSOR.FEATURE_DIM = 768
_TASK_CONFIG.TASK.CACHED_DINOV2_SENSOR.LRU_SIZE = 32
_TASK_CONFIG.TASK.CACHED_DINOV2_SENSOR_UUID = "cached_dinov2_feature"

# Optional 12-bin oracle goal-direction compass sensor.  Only active when
# "GOAL_COMPASS_SENSOR" is listed in TASK.SENSORS; the policy auto-detects it
# in the observation space and adds a Linear(12, 32) embedding alongside the
# GPS / compass / objectgoal streams.
_TASK_CONFIG.TASK.GOAL_COMPASS_SENSOR = CN()
_TASK_CONFIG.TASK.GOAL_COMPASS_SENSOR.TYPE = "GoalCompassSensor"
_TASK_CONFIG.TASK.GOAL_COMPASS_SENSOR.NUM_BINS = 12
_TASK_CONFIG.TASK.GOAL_COMPASS_SENSOR_UUID = "goal_compass"

# Optional online egocentric semantic + occupancy map sensor.  Only active
# when "SEMANTIC_MAP_SENSOR" is listed in TASK.SENSORS *and* DEPTH_SENSOR /
# SEMANTIC_SENSOR are listed in SIMULATOR.AGENT_0.SENSORS (with
# DEPTH_SENSOR.NORMALIZE_DEPTH = False).  Returns an (MAP_H, MAP_W) int8
# label map at every step; the policy converts to one-hot inside forward.
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR = CN()
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.TYPE = "SemanticMapSensor"
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.MAP_H = 256
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.MAP_W = 256
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.MAP_RESOLUTION = 0.025
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.SMOOTH_K = 4
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.WORLD_DIAMETER_M = 80.0
# Empty -> online accumulation from depth + semantic (the default behaviour).
# Non-empty -> load a precomputed world-frame label map from
# "<CACHE_ROOT>/<scene>/<scene>.npz" (the layout written by
# teleop_semantic_map.save_map) and skip the per-step depth/semantic
# projection; the policy still receives an agent-centered egocentric crop.
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.CACHE_ROOT = ""
_TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR_UUID = "semantic_map"

# Optional online egocentric object cloud sensor (variable-N tracked-object
# centroids packed to a fixed (MAX_OBJECTS, 4) float32 array).  Active iff
# "EGO_OBJECT_CLOUD_SENSOR" is listed in TASK.SENSORS *and* DEPTH_SENSOR /
# SEMANTIC_SENSOR are listed in SIMULATOR.AGENT_0.SENSORS (with
# DEPTH_SENSOR.NORMALIZE_DEPTH = False).  Each row is [task_id, ex, ey, ez]
# in agent-frame coordinates; padding rows have task_id = -1.  Consumed by
# ObjectCloudEncoder inside ObjectNavILMAENet.
_TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR = CN()
_TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.TYPE = "EgoObjectCloudSensor"
_TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS = 80
_TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MIN_MASK_PIXELS = 100
# Empty -> online accumulation (current behaviour).  Non-empty -> load
# precomputed world-frame cloud from "<CACHE_ROOT>/<scene>/<scene>.npz"
# (the layout written by dump_scene_object_clouds.py / teleop_object_cloud.py).
_TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.CACHE_ROOT = ""
_TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR_UUID = "ego_object_cloud"

_TASK_CONFIG.TASK.SIMPLE_REWARD = CN()
_TASK_CONFIG.TASK.SIMPLE_REWARD.TYPE = "SimpleReward"
_TASK_CONFIG.TASK.SIMPLE_REWARD.SUCCESS_REWARD = 2.5
_TASK_CONFIG.TASK.SIMPLE_REWARD.ANGLE_SUCCESS_REWARD = 2.5
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_DTG_REWARD = True
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_ATG_REWARD = True
_TASK_CONFIG.TASK.SIMPLE_REWARD.ATG_REWARD_DISTANCE = 1.0
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_ATG_FIX = True
_TASK_CONFIG.TASK.SIMPLE_REWARD.SLACK_PENALTY = -0.01

_TASK_CONFIG.TASK.SPARSE_REWARD = CN()
_TASK_CONFIG.TASK.SPARSE_REWARD.TYPE = "SparseReward"
_TASK_CONFIG.TASK.SPARSE_REWARD.SUCCESS_REWARD = 2.5

_TASK_CONFIG.TASK.TYPE = "Nav-v0"
_TASK_CONFIG.TASK.SUCCESS_DISTANCE = 0.1
_TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE = 0.1

_TASK_CONFIG.TASK.SUCCESS_MEASURE = "success"
_TASK_CONFIG.TASK.SUCCESS_REWARD = 2.5

_TASK_CONFIG.DATASET.TYPE = "ObjectNav-v2"
_TASK_CONFIG.DATASET.MAX_EPISODE_STEPS = 500


def get_task_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = _TASK_CONFIG.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

_CONFIG = _BASE_CONFIG.clone()
_CONFIG.defrost()

_CONFIG.VERBOSE = True

_CONFIG.BASE_TASK_CONFIG_PATH = "configs/tasks/objectnav_hm3d.yaml"

_CONFIG.TRAINER_NAME = "pirlnav-ddppo"
_CONFIG.ENV_NAME = "SimpleRLEnv"
_CONFIG.SENSORS = ["RGB_SENSOR"]

_CONFIG.VIDEO_OPTION = []
_CONFIG.VIDEO_DIR = "data/video"
_CONFIG.TENSORBOARD_DIR = "data/tensorboard"
_CONFIG.EVAL_CKPT_PATH_DIR = "data/checkpoints"
_CONFIG.CHECKPOINT_FOLDER = "data/checkpoints"
_CONFIG.LOG_FILE = "data/train.log"

_CONFIG.NUM_ENVIRONMENTS = 10
_CONFIG.LOG_INTERVAL = 10
_CONFIG.NUM_CHECKPOINTS = 100
_CONFIG.NUM_UPDATES = 20000
_CONFIG.TOTAL_NUM_STEPS = -1.0

_CONFIG.FORCE_TORCH_SINGLE_THREADED = False

_CONFIG.RUN_TYPE = None

_CONFIG.EVAL.SPLIT = "val"
_CONFIG.EVAL.USE_CKPT_CONFIG = True
_CONFIG.EVAL.EVAL_FREQ = 5
# When True, only episodes where the agent failed (success < 0.5) get their
# rollout video written to disk / tensorboard.  Default False preserves the
# original "record every episode" behaviour.
_CONFIG.EVAL.VIDEO_FAILED_ONLY = False

# Suffix each saved eval video filename with these metric names + values, so
# you can see at-a-glance which episodes succeeded.  Override in YAML if
# needed.  (CLI override is awkward because yacs requires python-literal
# strings, so we set a sensible default here and leave it alone.)
_CONFIG.EVAL_KEYS_TO_INCLUDE_IN_NAME = ["success", "spl"]

##############################################
# IL config
##############################################

_CONFIG.IL = CN()
_CONFIG.IL.POLICY = CN()
_CONFIG.IL.POLICY.name = "ObjectNavILMAEPolicy"
_CONFIG.IL.POLICY.USE_IW = True
_CONFIG.IL.POLICY.distrib_backend = "NCCL"
_CONFIG.IL.BehaviorCloning = CN()
_CONFIG.IL.BehaviorCloning.lr = 0.001
_CONFIG.IL.BehaviorCloning.encoder_lr = 0.001
_CONFIG.IL.BehaviorCloning.entropy_coef = 0.0
_CONFIG.IL.BehaviorCloning.eps = 1.0e-5
_CONFIG.IL.BehaviorCloning.wd = 0.0
_CONFIG.IL.BehaviorCloning.clip_param = 0.2
_CONFIG.IL.BehaviorCloning.num_mini_batch = 2
_CONFIG.IL.BehaviorCloning.max_grad_norm = 0.2
_CONFIG.IL.BehaviorCloning.num_steps = 64
_CONFIG.IL.BehaviorCloning.use_linear_clip_decay = False
_CONFIG.IL.BehaviorCloning.use_linear_lr_decay = True
_CONFIG.IL.BehaviorCloning.reward_window_size = 50
_CONFIG.IL.BehaviorCloning.sync_frac = 0.6
_CONFIG.IL.BehaviorCloning.use_double_buffered_sampler = False
_CONFIG.IL.BehaviorCloning.hidden_size = 2048

##############################################
# Policy config
##############################################

_CONFIG.POLICY = CN()
_CONFIG.POLICY.RGB_ENCODER = CN()
_CONFIG.POLICY.RGB_ENCODER.image_size = 256
_CONFIG.POLICY.RGB_ENCODER.backbone = "resnet50"
_CONFIG.POLICY.RGB_ENCODER.resnet_baseplanes = 32
_CONFIG.POLICY.RGB_ENCODER.vit_use_fc_norm = False
_CONFIG.POLICY.RGB_ENCODER.vit_global_pool = False
_CONFIG.POLICY.RGB_ENCODER.vit_use_cls = False
_CONFIG.POLICY.RGB_ENCODER.vit_mask_ratio = None
_CONFIG.POLICY.RGB_ENCODER.hidden_size = 512
_CONFIG.POLICY.RGB_ENCODER.use_augmentations = True
_CONFIG.POLICY.RGB_ENCODER.use_augmentations_test_time = True
_CONFIG.POLICY.RGB_ENCODER.randomize_augmentations_over_envs = False
_CONFIG.POLICY.RGB_ENCODER.pretrained_encoder = None
_CONFIG.POLICY.RGB_ENCODER.freeze_backbone = False
_CONFIG.POLICY.RGB_ENCODER.avgpooled_image = False
_CONFIG.POLICY.RGB_ENCODER.augmentations_name = "jitter+shift"
_CONFIG.POLICY.RGB_ENCODER.drop_path_rate = 0.0
_CONFIG.POLICY.RGB_ENCODER.normalize_visual_inputs = False
# DINOv2 variant (activated by setting backbone = "dinov2_base").
_CONFIG.POLICY.RGB_ENCODER.dinov2_model_name = "facebook/dinov2-base"
_CONFIG.POLICY.RGB_ENCODER.dinov2_resize_h = 476
_CONFIG.POLICY.RGB_ENCODER.dinov2_resize_w = 630
_CONFIG.POLICY.RGB_ENCODER.dinov2_output_dim = 768

_CONFIG.POLICY.GOAL_COMPASS_ENCODER = CN()
_CONFIG.POLICY.GOAL_COMPASS_ENCODER.embedding_size = 32

# Encoder for the egocentric semantic map (only used when SEMANTIC_MAP_SENSOR
# is in TASK.SENSORS).  Wraps VisualEncoder(<backbone>) with input_channels
# = NUM_CHANNELS + 1 = 24, then projects flat backbone output to
# embedding_size with a single Linear (no trailing ReLU, matching the
# goal_compass_embedding convention).
_CONFIG.POLICY.MAP_ENCODER = CN()
_CONFIG.POLICY.MAP_ENCODER.embedding_size = 32
_CONFIG.POLICY.MAP_ENCODER.backbone = "resnet18"

# Encoder for the egocentric object cloud (only used when
# EGO_OBJECT_CLOUD_SENSOR is in TASK.SENSORS).  PTv1-style point transformer
# that summarises the (MAX_OBJECTS, 4) packed cloud into a fixed
# embedding_size CLS vector concatenated into the GRU input alongside the
# cached-DINOv2 RGB feature.
_CONFIG.POLICY.OBJECT_CLOUD_ENCODER = CN()
_CONFIG.POLICY.OBJECT_CLOUD_ENCODER.embedding_size = 32
_CONFIG.POLICY.OBJECT_CLOUD_ENCODER.num_classes = 21
_CONFIG.POLICY.OBJECT_CLOUD_ENCODER.d_model = 64
_CONFIG.POLICY.OBJECT_CLOUD_ENCODER.num_layers = 2
_CONFIG.POLICY.OBJECT_CLOUD_ENCODER.rpe_mode = "log_distance"
_CONFIG.POLICY.OBJECT_CLOUD_ENCODER.ffn_expansion = 2

_CONFIG.POLICY.STATE_ENCODER = CN()
_CONFIG.POLICY.STATE_ENCODER.hidden_size = 2048
_CONFIG.POLICY.STATE_ENCODER.rnn_type = "GRU"
_CONFIG.POLICY.STATE_ENCODER.num_recurrent_layers = 2

_CONFIG.POLICY.SEQ2SEQ = CN()
_CONFIG.POLICY.SEQ2SEQ.use_prev_action = True

_CONFIG.POLICY.CRITIC = CN()
_CONFIG.POLICY.CRITIC.no_critic = False
_CONFIG.POLICY.CRITIC.mlp_critic = False
_CONFIG.POLICY.CRITIC.hidden_dim = 512

##############################################
# Default RL config
##############################################

_CONFIG.RL.POLICY.name = "ObjectNavILMAEPolicy"
_CONFIG.RL.PPO.num_mini_batch = 2
_CONFIG.RL.PPO.use_linear_lr_decay = True
_CONFIG.RL.PPO.hidden_size = 2048

##############################################
# Policy Finetuning config
##############################################

_CONFIG.RL.Finetune = CN()
_CONFIG.RL.Finetune.finetune = True
_CONFIG.RL.Finetune.freeze_encoders = True

_CONFIG.RL.Finetune.lr = 1.5e-5

_CONFIG.RL.Finetune.start_actor_warmup_at = 50
_CONFIG.RL.Finetune.start_actor_update_at = 100
_CONFIG.RL.Finetune.start_critic_warmup_at = 20
_CONFIG.RL.Finetune.start_critic_update_at = 80

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = _CONFIG.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    if config.NUM_PROCESSES != -1:
        warnings.warn(
            "NUM_PROCESSES is deprecated and will be removed in a future version."
            "  Use NUM_ENVIRONMENTS instead."
            "  Overwriting NUM_ENVIRONMENTS with NUM_PROCESSES for backwards compatibility."
        )

        config.NUM_ENVIRONMENTS = config.NUM_PROCESSES

    config.freeze()
    return config
