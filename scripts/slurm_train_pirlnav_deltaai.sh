#!/usr/bin/env bash
#SBATCH --job-name=pirlnav-il
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#
# NCSA Delta AI multi-GPU PIRLNav training template.
#
# Notes:
#   - Replace YOUR_ACCOUNT with the project name shown by `accounts`.
#   - ghx4 nodes have 4 GH200 superchips (1 H100 + 72 ARM cores each, 480 GB
#     CPU RAM, 384 GB GPU memory). With 4 tasks per node we leave a small
#     amount of CPU headroom (64 < 72 cores per task).
#   - Use --partition=ghx4-interactive (max 2 hr, max 4 nodes) for short
#     debugging.
#   - Stage MP3D, ObjectNav episodes, and visual encoders into /projects or
#     /work and pass them via EXTRA_BINDS, e.g.
#       EXTRA_BINDS="/projects/your_proj/mp3d:/workspace/pirlnav/data/scene_datasets/mp3d,/work/$USER/pirlnav_outputs:/workspace/pirlnav/data/new_checkpoints"

set -euo pipefail

# Set these before sbatch or edit them here.
SIF="${SIF:-/path/to/pirlnav-deltaai.sif}"
REPO="${REPO:-/path/to/pirlnav}"
EXTRA_BINDS="${EXTRA_BINDS:-}"

CONFIG="${CONFIG:-configs/experiments/il_objectnav_mp3d.yaml}"
TAG="${TAG:-deltaai_mp3d_1scene_ddp}"
NUM_UPDATES="${NUM_UPDATES:-20000}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-4}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
INFLECTION_COEF="${INFLECTION_COEF:-3.234951275740812}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-tb/objectnav_il/${TAG}/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/new_checkpoints/objectnav_il/${TAG}/}"
MAIN_PORT="${MAIN_PORT:-8738}"

mkdir -p "${REPO}/slurm_logs" "${REPO}/${TENSORBOARD_DIR}" "${REPO}/${CHECKPOINT_DIR}"

if [[ ! -f "${SIF}" ]]; then
  echo "SIF does not exist: ${SIF}" >&2
  exit 1
fi

if [[ ! -d "${REPO}" ]]; then
  echo "Repo does not exist: ${REPO}" >&2
  exit 1
fi

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER=singularity
else
  echo "Neither apptainer nor singularity is available on PATH." >&2
  exit 1
fi

BIND_ARGS=(--bind "${REPO}:/workspace/pirlnav")
if [[ -n "${EXTRA_BINDS}" ]]; then
  IFS=',' read -r -a EXTRA_BIND_ARRAY <<< "${EXTRA_BINDS}"
  for bind_spec in "${EXTRA_BIND_ARRAY[@]}"; do
    BIND_ARGS+=(--bind "${bind_spec}")
  done
fi

# Habitat's distributed helper uses MAIN_ADDR/MAIN_PORT plus SLURM rank vars.
export MAIN_ADDR="${MAIN_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')}"
export MAIN_PORT
export PYTHONUNBUFFERED=1
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export NCCL_ASYNC_ERROR_HANDLING=1

echo "[pirlnav-deltaai] sif=${SIF}"
echo "[pirlnav-deltaai] repo=${REPO}"
echo "[pirlnav-deltaai] config=${CONFIG}"
echo "[pirlnav-deltaai] tag=${TAG}"
echo "[pirlnav-deltaai] main=${MAIN_ADDR}:${MAIN_PORT}"
echo "[pirlnav-deltaai] tasks=${SLURM_NTASKS:-unknown} gpus_per_node=${SLURM_GPUS_ON_NODE:-unknown}"

srun "${CONTAINER}" exec --nv \
  "${BIND_ARGS[@]}" \
  --pwd /workspace/pirlnav \
  "${SIF}" \
  python -u -m run \
    --exp-config "${CONFIG}" \
    --run-type train \
    TENSORBOARD_DIR "${TENSORBOARD_DIR}" \
    CHECKPOINT_FOLDER "${CHECKPOINT_DIR}" \
    NUM_UPDATES "${NUM_UPDATES}" \
    NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}" \
    NUM_CHECKPOINTS "${NUM_CHECKPOINTS}" \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF "${INFLECTION_COEF}"
