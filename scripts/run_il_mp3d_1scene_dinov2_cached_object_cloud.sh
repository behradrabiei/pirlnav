#!/usr/bin/env bash
# Single-GPU launcher for pirlnav IL with cached-DINOv2 features + the online
# egocentric object cloud sensor on the one-scene MP3D subset. Replaces the
# semantic-map branch with a 32-d embedding from a small PTv1 point
# transformer (ObjectCloudEncoder) over the agent-frame (MAX_OBJECTS, 4)
# packed object cloud.
#
# Prerequisites:
#   - data/dinov2_cache/<scene>/*.pt populated by
#     scripts/precompute_dinov2_features.py (same cache as the regular
#     cached-DINOv2 variant; nothing extra to precompute here -- the cloud is
#     accumulated online from the sim's depth+semantic sensors).
#
# Usage:
#   bash scripts/run_il_mp3d_1scene_dinov2_cached_object_cloud.sh                # smoke (200 updates, 2 envs)
#   bash scripts/run_il_mp3d_1scene_dinov2_cached_object_cloud.sh --full         # 20k updates, 4 envs
#   NUM_UPDATES=1000 NUM_ENVIRONMENTS=2 bash scripts/run_il_mp3d_1scene_dinov2_cached_object_cloud.sh
#
# Env-var overrides (default = teleop geometry):
#   MAX_OBJECTS=80  MIN_MASK_PIXELS=100
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

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

CONFIG="configs/experiments/il_objectnav_mp3d_dinov2_cached_object_cloud.yaml"
TAG="${TAG:-mp3d_1scene_6cat_dinov2_cached_object_cloud}"
TENSORBOARD_DIR="tb/objectnav_il/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_cached_object_cloud/objectnav_il/${TAG}/"
INFLECTION_COEF="${INFLECTION_COEF:-3.234951275740812}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
CACHE_ROOT="${CACHE_ROOT:-data/dinov2_cache}"

MAX_OBJECTS="${MAX_OBJECTS:-80}"
MIN_MASK_PIXELS="${MIN_MASK_PIXELS:-100}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_il_mp3d_1scene_dinov2_cached_object_cloud] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_il_mp3d_1scene_dinov2_cached_object_cloud] tb=${TENSORBOARD_DIR}"
echo "[run_il_mp3d_1scene_dinov2_cached_object_cloud] ckpt=${CHECKPOINT_DIR}"
echo "[run_il_mp3d_1scene_dinov2_cached_object_cloud] dinov2_cache=${CACHE_ROOT}"
echo "[run_il_mp3d_1scene_dinov2_cached_object_cloud] cloud max_objects=${MAX_OBJECTS} min_mask_pixels=${MIN_MASK_PIXELS}"

set -x
python -u -m run \
  --exp-config "${CONFIG}" \
  --run-type train \
  TENSORBOARD_DIR "${TENSORBOARD_DIR}" \
  CHECKPOINT_FOLDER "${CHECKPOINT_DIR}" \
  NUM_UPDATES "${NUM_UPDATES}" \
  NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}" \
  NUM_CHECKPOINTS "${NUM_CHECKPOINTS}" \
  TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF "${INFLECTION_COEF}" \
  TASK_CONFIG.TASK.CACHED_DINOV2_SENSOR.CACHE_ROOT "${CACHE_ROOT}" \
  TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS "${MAX_OBJECTS}" \
  TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MIN_MASK_PIXELS "${MIN_MASK_PIXELS}"
