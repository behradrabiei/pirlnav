#!/usr/bin/env bash
# Single-GPU launcher for pirlnav RL fine-tuning with the *dino-only* policy
# variant on the one-scene MP3D subset, using online DINOv2 @ 252x252.
#
# Prereq: a finished IL checkpoint from run_il_mp3d_1scene_dinov2_cached_252.sh
# (default: ckpt.10.pth).  RL loads the IL trainable heads (visual_fc, GRU,
# action head); DINOv2 backbone is pulled from HuggingFace at runtime; critic
# is reinitialized.
#
# Inputs to the policy during RL:
#   - Live RGB -> frozen DINOv2-base @ 252x252
#   - OBJECTGOAL  /  COMPASS  /  GPS
#
# Usage:
#   bash scripts/run_rl_mp3d_1scene_dinov2_252.sh                # smoke (200 updates, 2 envs)
#   bash scripts/run_rl_mp3d_1scene_dinov2_252.sh --full         # 20k updates, 4 envs
#   NUM_UPDATES=10 NUM_ENVIRONMENTS=2 \
#       bash scripts/run_rl_mp3d_1scene_dinov2_252.sh
#
# Env-var overrides:
#   IL_CKPT=...                    # IL initialization checkpoint
#   TAG=...                        # output naming tag
#   SUCCESS_DISTANCE=1.0           # match demo recording + eval
#   ALLOW_SLIDING=True             # match demo recording + eval
#   NUM_CHECKPOINTS=10
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

CONFIG="configs/experiments/rl_objectnav_mp3d_dinov2_252.yaml"
TAG="${TAG:-mp3d_1scene_6cat_dinov2_rl_252}"
TENSORBOARD_DIR="tb/objectnav_rl/${TAG}/"
CHECKPOINT_DIR="data/new_checkpoints_dinov2_rl_252/objectnav_rl/${TAG}/"
IL_CKPT="${IL_CKPT:-data/new_checkpoints_dinov2_cached_252/objectnav_il/mp3d_1scene_6cat_dinov2_cached_252/ckpt.10.pth}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-1.0}"
ALLOW_SLIDING="${ALLOW_SLIDING:-True}"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "[run_rl_mp3d_1scene_dinov2_252] mode=${MODE} updates=${NUM_UPDATES} envs=${NUM_ENVIRONMENTS} ckpts=${NUM_CHECKPOINTS}"
echo "[run_rl_mp3d_1scene_dinov2_252] tb=${TENSORBOARD_DIR}"
echo "[run_rl_mp3d_1scene_dinov2_252] ckpt=${CHECKPOINT_DIR}"
echo "[run_rl_mp3d_1scene_dinov2_252] il_init=${IL_CKPT}"
echo "[run_rl_mp3d_1scene_dinov2_252] success_distance=${SUCCESS_DISTANCE} allow_sliding=${ALLOW_SLIDING}"

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
  TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE "${SUCCESS_DISTANCE}" \
  TASK_CONFIG.TASK.SUCCESS_DISTANCE "${SUCCESS_DISTANCE}" \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING "${ALLOW_SLIDING}"
