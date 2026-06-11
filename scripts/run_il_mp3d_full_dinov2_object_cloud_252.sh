#!/usr/bin/env bash
# Single-node launcher for pirlnav IL with the online-DINOv2 (@ 252x252) +
# online object-cloud variant on the *full* MP3D THDA 70k dataset filtered to
# the canonical 21 ObjectNav classes (56 scenes, 60085 episodes).
#
# Combines scripts/run_il_mp3d_full_dinov2.sh (dino-only @ 252) with the
# object-cloud knobs from scripts/run_il_mp3d_full_dinov2_object_cloud.sh, but
# with MAX_OBJECTS=100 and the lighter 252x252 backbone so the run targets one
# ghx4 node (4 GPUs) at NUM_ENVIRONMENTS=40.
#
# Run inside the dependency-only Apptainer image with these bind mounts:
#   - data/scene_datasets/mp3d/                                           (MP3D scenes + semantic annotations)
#   - data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat/  (filtered demos)
# See DELTAAI_CONTAINER.md for the canonical bind-mount layout and
# scripts/slurm_train_pirlnav_deltaai_full_dinov2_object_cloud.sh for the
# 4-GPU SLURM version.
#
# Usage:
#   bash scripts/run_il_mp3d_full_dinov2_object_cloud_252.sh                # smoke (20 updates, 4 envs)
#   bash scripts/run_il_mp3d_full_dinov2_object_cloud_252.sh --full         # 54000 updates, 40 envs (~500M steps @ 4 ranks)
#   NUM_UPDATES=30 NUM_ENVIRONMENTS=40 \
#     bash scripts/run_il_mp3d_full_dinov2_object_cloud_252.sh              # 40-env VRAM/throughput probe
#
# Env-var overrides:
#   CONFIG, NUM_UPDATES, NUM_ENVIRONMENTS, NUM_CHECKPOINTS, MAX_OBJECTS,
#   REPLAY_MODE, INFLECTION_COEF, TAG
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "pirlnav" ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate pirlnav
fi

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONUNBUFFERED=1

MODE="${1:-smoke}"
if [[ "${MODE}" == "--full" ]]; then
  NUM_UPDATES="${NUM_UPDATES:-54000}"
  NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-40}"
else
  NUM_UPDATES="${NUM_UPDATES:-20}"
  NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-4}"
fi

CONFIG="${CONFIG:-configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full_252.yaml}"
TAG="${TAG:-mp3d_full_dinov2_object_cloud_252}"
TENSORBOARD_DIR="tb/objectnav_il/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_object_cloud_full_252/objectnav_il/${TAG}/"
INFLECTION_COEF="${INFLECTION_COEF:-3.513870128085}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-60}"
MAX_OBJECTS="${MAX_OBJECTS:-100}"
REPLAY_MODE="${REPLAY_MODE:-poses}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_il_mp3d_full_dinov2_object_cloud_252] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_il_mp3d_full_dinov2_object_cloud_252] tb=${TENSORBOARD_DIR}"
echo "[run_il_mp3d_full_dinov2_object_cloud_252] ckpt=${CHECKPOINT_DIR}"
echo "[run_il_mp3d_full_dinov2_object_cloud_252] cloud max_objects=${MAX_OBJECTS} replay_mode=${REPLAY_MODE}"
echo "[run_il_mp3d_full_dinov2_object_cloud_252] inflection_coef=${INFLECTION_COEF}"

set -x
python -u -m run \
  --exp-config "${CONFIG}" \
  --run-type train \
  TENSORBOARD_DIR "${TENSORBOARD_DIR}" \
  CHECKPOINT_FOLDER "${CHECKPOINT_DIR}" \
  NUM_UPDATES "${NUM_UPDATES}" \
  NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}" \
  NUM_CHECKPOINTS "${NUM_CHECKPOINTS}" \
  IL.BehaviorCloning.REPLAY_MODE "${REPLAY_MODE}" \
  TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF "${INFLECTION_COEF}" \
  TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS "${MAX_OBJECTS}"
