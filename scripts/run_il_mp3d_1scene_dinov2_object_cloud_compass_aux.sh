#!/usr/bin/env bash
# Single-GPU launcher for pirlnav IL with cached-DINOv2 features + the *online*
# egocentric object cloud sensor (accumulated from live depth + semantic), with
# the goal-compass auxiliary regression loss enabled on the point-transformer's
# 12-D head.  Drop-in alternative to
# run_il_mp3d_1scene_dinov2_cached_object_cloud.sh: same inputs (agent pose,
# objectgoal, cached DINOv2, online object cloud) -- only the encoder is
# extended with a compass-prediction head supervised against the privileged
# GoalCompassSensor.  The oracle value is *only* used as a training label;
# it is never fed into the policy at train or eval time.
#
# Prerequisites:
#   - data/dinov2_cache/<scene>/*.pt populated by
#     scripts/precompute_dinov2_features.py (same cache as the regular
#     cached-DINOv2 variant; nothing extra to precompute here -- the cloud is
#     accumulated online from the sim's depth+semantic sensors).
#
# Usage:
#   bash scripts/run_il_mp3d_1scene_dinov2_object_cloud_compass_aux.sh           # smoke (200 updates, 2 envs)
#   bash scripts/run_il_mp3d_1scene_dinov2_object_cloud_compass_aux.sh --full    # 20k updates, 4 envs
#   NUM_UPDATES=1000 NUM_ENVIRONMENTS=2 bash scripts/run_il_mp3d_1scene_dinov2_object_cloud_compass_aux.sh
#
# Env-var overrides (default = teleop geometry):
#   MAX_OBJECTS=80  MIN_MASK_PIXELS=100
#   CACHE_ROOT=data/dinov2_cache
#   COMPASS_AUX_COEF=1.0   # weight on the auxiliary MSE loss
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

CONFIG="configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_compass_aux.yaml"
TAG="${TAG:-mp3d_1scene_6cat_dinov2_object_cloud_compass_aux}"
TENSORBOARD_DIR="tb/objectnav_il/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_object_cloud_compass_aux/objectnav_il/${TAG}/"
INFLECTION_COEF="${INFLECTION_COEF:-3.234951275740812}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
CACHE_ROOT="${CACHE_ROOT:-data/dinov2_cache}"
COMPASS_AUX_COEF="${COMPASS_AUX_COEF:-1.0}"

MAX_OBJECTS="${MAX_OBJECTS:-80}"
MIN_MASK_PIXELS="${MIN_MASK_PIXELS:-100}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_il_mp3d_1scene_dinov2_object_cloud_compass_aux] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_il_mp3d_1scene_dinov2_object_cloud_compass_aux] tb=${TENSORBOARD_DIR}"
echo "[run_il_mp3d_1scene_dinov2_object_cloud_compass_aux] ckpt=${CHECKPOINT_DIR}"
echo "[run_il_mp3d_1scene_dinov2_object_cloud_compass_aux] dinov2_cache=${CACHE_ROOT}"
echo "[run_il_mp3d_1scene_dinov2_object_cloud_compass_aux] cloud max_objects=${MAX_OBJECTS} min_mask_pixels=${MIN_MASK_PIXELS}"
echo "[run_il_mp3d_1scene_dinov2_object_cloud_compass_aux] compass_aux_coef=${COMPASS_AUX_COEF}"

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
  TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MIN_MASK_PIXELS "${MIN_MASK_PIXELS}" \
  IL.BehaviorCloning.compass_aux_coef "${COMPASS_AUX_COEF}"
