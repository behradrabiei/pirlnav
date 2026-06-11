#!/usr/bin/env bash
# Single-GPU launcher for pirlnav RL fine-tuning with online DINOv2 @ 252x252 plus
# the online egocentric object cloud on the one-scene MP3D subset.
#
# Prereq: a finished IL checkpoint from
# run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud.sh (default: ckpt.10.pth).
# RL loads IL trainable heads (visual_fc, object_cloud_encoder, GRU, action head);
# DINOv2 backbone is pulled from HuggingFace at runtime; critic is reinitialized.
#
# Inputs to the policy during RL:
#   - Live RGB -> frozen DINOv2-base @ 252x252
#   - OBJECTGOAL / COMPASS / GPS
#   - EGO_OBJECT_CLOUD (online from depth+semantic; no NPZ cache)
#
# Usage:
#   bash scripts/run_rl_mp3d_1scene_dinov2_252_online_object_cloud.sh          # smoke
#   bash scripts/run_rl_mp3d_1scene_dinov2_252_online_object_cloud.sh --full  # 20k updates
#
# Env-var overrides:
#   IL_CKPT=...  TAG=...  SUCCESS_DISTANCE=1.0  ALLOW_SLIDING=True
#   MAX_OBJECTS=80  MIN_MASK_PIXELS=100  NUM_CHECKPOINTS=10
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "pirlnav" ]]; then
  source /workspace/conda/etc/profile.d/conda.sh
  conda activate pirlnav
fi

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONUNBUFFERED=1

MODE="${1:-smoke}"
if [[ "${MODE}" == "--full" ]]; then
  NUM_UPDATES="${NUM_UPDATES:-20000}"
  NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-4}"
else
  NUM_UPDATES="${NUM_UPDATES:-200}"
  NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-2}"
fi

CONFIG="configs/experiments/rl_objectnav_mp3d_dinov2_252_online_object_cloud.yaml"
TAG="${TAG:-mp3d_1scene_6cat_dinov2_rl_252_online_object_cloud}"
TENSORBOARD_DIR="tb/objectnav_rl/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_rl_252_online_object_cloud/objectnav_rl/${TAG}/"
IL_CKPT="${IL_CKPT:-data/new_checkpoints_dinov2_cached_252_online_object_cloud/objectnav_il/mp3d_1scene_6cat_dinov2_cached_252_online_object_cloud/ckpt.10.pth}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-1.0}"
ALLOW_SLIDING="${ALLOW_SLIDING:-True}"
MAX_OBJECTS="${MAX_OBJECTS:-80}"
MIN_MASK_PIXELS="${MIN_MASK_PIXELS:-100}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_rl_mp3d_1scene_dinov2_252_online_object_cloud] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_rl_mp3d_1scene_dinov2_252_online_object_cloud] tb=${TENSORBOARD_DIR}"
echo "[run_rl_mp3d_1scene_dinov2_252_online_object_cloud] ckpt=${CHECKPOINT_DIR}"
echo "[run_rl_mp3d_1scene_dinov2_252_online_object_cloud] il_init=${IL_CKPT}"
echo "[run_rl_mp3d_1scene_dinov2_252_online_object_cloud] cloud max_objects=${MAX_OBJECTS} min_mask_pixels=${MIN_MASK_PIXELS}"
echo "[run_rl_mp3d_1scene_dinov2_252_online_object_cloud] success_distance=${SUCCESS_DISTANCE} allow_sliding=${ALLOW_SLIDING}"

set -x
python -u -m run \
  --exp-config "${CONFIG}" \
  --run-type train \
  TENSORBOARD_DIR "${TENSORBOARD_DIR}" \
  CHECKPOINT_FOLDER "${CHECKPOINT_DIR}" \
  NUM_UPDATES "${NUM_UPDATES}" \
  NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}" \
  NUM_CHECKPOINTS "${NUM_CHECKPOINTS}" \
  RL.DDPPO.pretrained_weights "${IL_CKPT}" \
  TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS "${MAX_OBJECTS}" \
  TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MIN_MASK_PIXELS "${MIN_MASK_PIXELS}" \
  TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE "${SUCCESS_DISTANCE}" \
  TASK_CONFIG.TASK.SUCCESS_DISTANCE "${SUCCESS_DISTANCE}" \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING "${ALLOW_SLIDING}"
