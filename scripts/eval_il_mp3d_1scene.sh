#!/usr/bin/env bash
# Single-GPU, non-SLURM eval on the one-scene MP3D subset.
#
# Usage:
#   bash scripts/eval_il_mp3d_1scene.sh <ckpt_path_or_dir>
#
# Env overrides:
#   EVAL_SPLIT=val      (default: val)   which split folder to iterate
#   NUM_ENVIRONMENTS=2  (default: 2)
#   TAG=<name>          (default: derived from ckpt path)
#   TEST_EPISODE_COUNT=-1  (default: -1, i.e. all episodes in the split)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ckpt_path_or_dir>" >&2
  exit 1
fi

EVAL_CKPT="$1"

cd "$(dirname "$0")/.."

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "pirlnav" ]]; then
  source /workspace/conda/etc/profile.d/conda.sh
  conda activate pirlnav
fi

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONUNBUFFERED=1

CONFIG="configs/experiments/il_objectnav_mp3d.yaml"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-2}"
TEST_EPISODE_COUNT="${TEST_EPISODE_COUNT:--1}"
TAG="${TAG:-$(basename "${EVAL_CKPT}" .pth)_on_${EVAL_SPLIT}}"
TENSORBOARD_DIR="tb/objectnav_il_eval/${TAG}/"
mkdir -p "${TENSORBOARD_DIR}"

echo "[eval_il_mp3d_1scene] ckpt=${EVAL_CKPT}"
echo "[eval_il_mp3d_1scene] split=${EVAL_SPLIT} envs=${NUM_ENVIRONMENTS}"
echo "[eval_il_mp3d_1scene] tb=${TENSORBOARD_DIR}"

set -x
python -u -m run \
  --exp-config "${CONFIG}" \
  --run-type eval \
  TENSORBOARD_DIR "${TENSORBOARD_DIR}" \
  EVAL_CKPT_PATH_DIR "${EVAL_CKPT}" \
  NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}" \
  TEST_EPISODE_COUNT "${TEST_EPISODE_COUNT}" \
  EVAL.USE_CKPT_CONFIG False \
  EVAL.SPLIT "${EVAL_SPLIT}"
