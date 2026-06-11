#!/usr/bin/env bash
# Single-node launcher for pirlnav IL with the online-DINOv2 *dino-only*
# variant @ 252x252 on the *full* MP3D THDA 70k dataset filtered to the
# canonical 21 ObjectNav classes (56 scenes, 60085 episodes).
#
# Sibling of scripts/run_il_mp3d_full_dinov2_object_cloud.sh; this one
# drops the online cloud + point transformer branch so the policy is
# strictly DINOv2 CLS + OBJECTGOAL/COMPASS/GPS.
#
# Run inside the dependency-only Apptainer image with these bind mounts:
#   - data/scene_datasets/mp3d/                                           (90 MP3D scenes; only the 56 referenced scenes load)
#   - data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat/  (filtered demos)
# See DELTAAI_CONTAINER.md for the canonical bind-mount layout.
#
# Usage:
#   bash scripts/run_il_mp3d_full_dinov2.sh                    # smoke (20 updates, 4 envs)
#   bash scripts/run_il_mp3d_full_dinov2.sh --full             # 125k updates, 8 envs
#   NUM_UPDATES=30 NUM_ENVIRONMENTS=16 \
#     bash scripts/run_il_mp3d_full_dinov2.sh                  # NUM_ENVIRONMENTS probe
#
# Env-var overrides:
#   NUM_UPDATES, NUM_ENVIRONMENTS, NUM_CHECKPOINTS, INFLECTION_COEF, TAG
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
  NUM_UPDATES="${NUM_UPDATES:-125000}"
  NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-8}"
else
  NUM_UPDATES="${NUM_UPDATES:-20}"
  NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-4}"
fi

CONFIG="configs/experiments/il_objectnav_mp3d_dinov2_full.yaml"
TAG="${TAG:-mp3d_full_dinov2_252}"
TENSORBOARD_DIR="tb/objectnav_il/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_full/objectnav_il/${TAG}/"
INFLECTION_COEF="${INFLECTION_COEF:-3.513870128085}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-50}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_il_mp3d_full_dinov2] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_il_mp3d_full_dinov2] tb=${TENSORBOARD_DIR}"
echo "[run_il_mp3d_full_dinov2] ckpt=${CHECKPOINT_DIR}"
echo "[run_il_mp3d_full_dinov2] inflection_coef=${INFLECTION_COEF}"

set -x
python -u -m run \
  --exp-config "${CONFIG}" \
  --run-type train \
  TENSORBOARD_DIR "${TENSORBOARD_DIR}" \
  CHECKPOINT_FOLDER "${CHECKPOINT_DIR}" \
  NUM_UPDATES "${NUM_UPDATES}" \
  NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}" \
  NUM_CHECKPOINTS "${NUM_CHECKPOINTS}" \
  TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF "${INFLECTION_COEF}"
