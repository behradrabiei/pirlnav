#!/usr/bin/env bash
# Single-GPU launcher for pirlnav IL with cached-DINOv2 CLS features precomputed
# at 252x252 in pose-replay mode, plus the *online* egocentric object cloud
# sensor (accumulated from live depth + semantic).  No compass auxiliary loss
# and no oracle NPZ cloud cache.
#
# Drop-in successor to run_il_mp3d_1scene_dinov2_cached_object_cloud.sh: same
# policy inputs (cached DINOv2 + online object cloud) but uses the pose-replay
# 252 cache and IL.BehaviorCloning.REPLAY_MODE=poses so rollout collection
# teleports to recorded agent_state each step.
#
# Inputs to the policy:
#   - CACHED_DINOV2 (768-d CLS, frozen, precomputed at 252x252, pose-replay)
#   - OBJECTGOAL / COMPASS / GPS
#   - EGO_OBJECT_CLOUD (online from depth+semantic; no NPZ cloud cache)
#   - INFLECTION_WEIGHT and DEMONSTRATION sensors (auto-added by the trainer)
#   - NEXT_POSE sensor (auto-added when REPLAY_MODE == "poses")
#
# Prereq:  populate the 252x252 pose-replay cache (same as
# run_il_mp3d_1scene_dinov2_cached_252.sh).  Precompute appends "_poses" to the
# cache root, so the on-disk layout is:
#   data/dinov2_cache_poses_252_poses/<scene>/<episode_id>.pt
#
#   python -m scripts.precompute_dinov2_features \
#       --cache-root data/dinov2_cache_poses_252 \
#       --resize-h 252 --resize-w 252 --replay-mode poses
#
# No extra object-cloud precompute -- the cloud is accumulated online from the
# sim's depth+semantic sensors.
#
# Usage:
#   bash scripts/run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud.sh                # smoke (200 updates, 2 envs)
#   bash scripts/run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud.sh --full         # 20k updates, 4 envs
#   NUM_UPDATES=1000 NUM_ENVIRONMENTS=2 \
#       bash scripts/run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud.sh
#
# Env-var overrides (default = teleop geometry):
#   CACHE_ROOT=data/dinov2_cache_poses_252_poses   # cached .pt feature root
#   REPLAY_MODE=poses              # poses (default) or actions
#   MAX_OBJECTS=80  MIN_MASK_PIXELS=100
#   INFLECTION_COEF=...            # weight for inflection-up-weighted CE loss
#   NUM_CHECKPOINTS=10
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
TAG="${TAG:-mp3d_1scene_6cat_dinov2_cached_252_online_object_cloud}"
TENSORBOARD_DIR="tb/objectnav_il/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_cached_252_online_object_cloud/objectnav_il/${TAG}/"
INFLECTION_COEF="${INFLECTION_COEF:-3.234951275740812}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
CACHE_ROOT="${CACHE_ROOT:-data/dinov2_cache_poses_252_poses}"
REPLAY_MODE="${REPLAY_MODE:-poses}"

MAX_OBJECTS="${MAX_OBJECTS:-80}"
MIN_MASK_PIXELS="${MIN_MASK_PIXELS:-100}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud] tb=${TENSORBOARD_DIR}"
echo "[run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud] ckpt=${CHECKPOINT_DIR}"
echo "[run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud] cache=${CACHE_ROOT}"
echo "[run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud] replay_mode=${REPLAY_MODE}"
echo "[run_il_mp3d_1scene_dinov2_cached_252_online_object_cloud] cloud max_objects=${MAX_OBJECTS} min_mask_pixels=${MIN_MASK_PIXELS}"

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
  IL.BehaviorCloning.REPLAY_MODE "${REPLAY_MODE}"
