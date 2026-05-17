#!/usr/bin/env bash
# Single-GPU launcher for pirlnav IL with cached-DINOv2 features + the online
# semantic map sensor on the one-scene MP3D subset. Replaces the goal_compass
# branch with a 32-d embedding from a small ResNet18 over the agent's
# accumulated egocentric semantic+occupancy map.
#
# Prerequisites:
#   - data/dinov2_cache/<scene>/*.pt populated by
#     scripts/precompute_dinov2_features.py (same cache as the regular
#     cached-DINOv2 variant; nothing extra to precompute here -- the map is
#     computed online from the sim's depth+semantic sensors).
#
# Usage:
#   bash scripts/run_il_mp3d_1scene_dinov2_cached_semmap.sh                # smoke (200 updates, 2 envs)
#   bash scripts/run_il_mp3d_1scene_dinov2_cached_semmap.sh --full         # 20k updates, 4 envs
#   NUM_UPDATES=1000 NUM_ENVIRONMENTS=2 bash scripts/run_il_mp3d_1scene_dinov2_cached_semmap.sh
#
# Env-var overrides (default = teleop geometry):
#   MAP_H=256  MAP_W=256  MAP_RES=0.025  SMOOTH_K=4
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

CONFIG="configs/experiments/il_objectnav_mp3d_dinov2_cached_semmap.yaml"
TAG="${TAG:-mp3d_1scene_6cat_dinov2_cached_semmap}"
TENSORBOARD_DIR="tb/objectnav_il/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_cached_semmap/objectnav_il/${TAG}/"
INFLECTION_COEF="${INFLECTION_COEF:-3.234951275740812}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
CACHE_ROOT="${CACHE_ROOT:-data/dinov2_cache}"

MAP_H="${MAP_H:-256}"
MAP_W="${MAP_W:-256}"
MAP_RES="${MAP_RES:-0.025}"
SMOOTH_K="${SMOOTH_K:-4}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_il_mp3d_1scene_dinov2_cached_semmap] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_il_mp3d_1scene_dinov2_cached_semmap] tb=${TENSORBOARD_DIR}"
echo "[run_il_mp3d_1scene_dinov2_cached_semmap] ckpt=${CHECKPOINT_DIR}"
echo "[run_il_mp3d_1scene_dinov2_cached_semmap] dinov2_cache=${CACHE_ROOT}"
echo "[run_il_mp3d_1scene_dinov2_cached_semmap] map=${MAP_H}x${MAP_W} res=${MAP_RES} smooth_k=${SMOOTH_K}"

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
  TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.MAP_H "${MAP_H}" \
  TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.MAP_W "${MAP_W}" \
  TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.MAP_RESOLUTION "${MAP_RES}" \
  TASK_CONFIG.TASK.SEMANTIC_MAP_SENSOR.SMOOTH_K "${SMOOTH_K}"
