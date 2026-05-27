#!/usr/bin/env bash
# Single-node launcher for pirlnav IL with the online DINOv2 + online
# object cloud variant on the *full* MP3D THDA 70k dataset, filtered down
# to the canonical 21 ObjectNav classes via
# scripts/filter_mp3d_thda_to_21cat.py (56 scenes, 60085 episodes).
# Mirrors the existing 1-scene launchers in shape; the new bits are the
# config switch and the env-var knobs you need for the GH200
# NUM_ENVIRONMENTS probe.
#
# Run inside the dependency-only Apptainer image with the three full-MP3D
# bind mounts in place; see DELTAAI_CONTAINER.md for the bind layout and
# scripts/slurm_train_pirlnav_deltaai_full.sh for the multi-node version.
#
# Prerequisites (all bind-mounted into /workspace/pirlnav at runtime):
#   - data/scene_datasets/mp3d/                                          (90 MP3D scene assets)
#   - data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat/  (filtered demos)
#
# Usage:
#   bash scripts/run_il_mp3d_full_dinov2_object_cloud.sh                # smoke (200 updates, 4 envs)
#   bash scripts/run_il_mp3d_full_dinov2_object_cloud.sh --full         # 125k updates, 8 envs (paper-faithful at 8 ranks)
#   NUM_UPDATES=20 NUM_ENVIRONMENTS=16 \
#     bash scripts/run_il_mp3d_full_dinov2_object_cloud.sh              # interactive VRAM probe
#
# Env-var overrides:
#   NUM_UPDATES, NUM_ENVIRONMENTS, NUM_CHECKPOINTS, MAX_OBJECTS,
#   INFLECTION_COEF, TAG
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
  NUM_UPDATES="${NUM_UPDATES:-125000}"
  NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-8}"
else
  NUM_UPDATES="${NUM_UPDATES:-200}"
  NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-4}"
fi

CONFIG="configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full.yaml"
TAG="${TAG:-mp3d_full_dinov2_object_cloud}"
TENSORBOARD_DIR="tb/objectnav_il/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_object_cloud_full/objectnav_il/${TAG}/"
# Recomputed for the 21-class filtered THDA 70k bundle (56 scenes,
# 60085 episodes, 14.1M total / 4.0M inflection steps -> coef = total/inflection).
INFLECTION_COEF="${INFLECTION_COEF:-3.513870128085}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-50}"
MAX_OBJECTS="${MAX_OBJECTS:-300}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_il_mp3d_full_dinov2_object_cloud] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_il_mp3d_full_dinov2_object_cloud] tb=${TENSORBOARD_DIR}"
echo "[run_il_mp3d_full_dinov2_object_cloud] ckpt=${CHECKPOINT_DIR}"
echo "[run_il_mp3d_full_dinov2_object_cloud] cloud max_objects=${MAX_OBJECTS}"
echo "[run_il_mp3d_full_dinov2_object_cloud] inflection_coef=${INFLECTION_COEF}"

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
  TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS "${MAX_OBJECTS}"
