#!/usr/bin/env bash
#SBATCH --job-name=pirlnav-il-full
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --account=bgon-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#
# NCSA Delta AI 2-node x 4-GPU = 8-rank PIRLNav IL run on the *full* MP3D
# THDA 70k dataset, filtered to the canonical 21 ObjectNav classes via
# scripts/filter_mp3d_thda_to_21cat.py (60085 episodes / 56 scenes;
# online DINOv2 + online object cloud, larger point transformer,
# MAX_OBJECTS=300). Sibling of slurm_train_pirlnav_deltaai.sh with
# multi-node SBATCH headers and the three --bind flags pre-wired.
#
# Notes:
#   - Adjust --account if you submit under a different project than
#     bgon-dtai-gh.
#   - Default binds map the on-disk layout under /projects/bgon/brabiei/MP3D
#     to the repo's expected data/ paths inside the container; override
#     EXTRA_BINDS to add or substitute paths.
#   - At 8 ranks * NUM_ENVIRONMENTS=8 envs * num_steps=64 frames * 125k
#     updates this targets ~512M total env steps, on par with the paper's
#     500M-step BC budget. If the interactive VRAM probe lets you push
#     NUM_ENVIRONMENTS higher, drop NUM_UPDATES proportionally.
#   - Use --partition=ghx4-interactive (max 2 hr, max 4 nodes) for the
#     short interactive probe described in DELTAAI_CONTAINER.md.

set -euo pipefail

SIF="${SIF:-/path/to/pirlnav-deltaai.sif}"
REPO="${REPO:-/path/to/pirlnav}"

# Default bind mounts for the full-MP3D run. Comma-separated; each entry is
# "<host_path>:<container_path>". Override or extend via the EXTRA_BINDS
# environment variable on submit.
DEFAULT_FULL_BINDS=$(cat <<'BINDS' | tr '\n' ',' | sed 's/,$//'
/projects/bgon/brabiei/MP3D/scenes/mp3d:/workspace/pirlnav/data/scene_datasets/mp3d
/projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat:/workspace/pirlnav/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat
/projects/bgon/brabiei/MP3D/eval_episodes:/workspace/pirlnav/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_v1
BINDS
)
EXTRA_BINDS="${EXTRA_BINDS:-${DEFAULT_FULL_BINDS}}"

CONFIG="${CONFIG:-configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full.yaml}"
TAG="${TAG:-deltaai_mp3d_full_dinov2_object_cloud_ddp_8gpu}"
NUM_UPDATES="${NUM_UPDATES:-125000}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-8}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-50}"
# Recomputed for the 21-class filtered THDA 70k bundle via
# scripts/compute_inflection_coef.py (60085 episodes, 14.1M total steps,
# 28.459% inflection rate).
INFLECTION_COEF="${INFLECTION_COEF:-3.513870128085}"
MAX_OBJECTS="${MAX_OBJECTS:-300}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-tb/objectnav_il/${TAG}/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/new_checkpoints_dinov2_object_cloud_full/objectnav_il/${TAG}/}"
MAIN_PORT="${MAIN_PORT:-8738}"

# Create placeholder directories under the repo so Apptainer's bind mount
# has a target. No host-side symlinks anywhere.
mkdir -p "${REPO}/slurm_logs" \
         "${REPO}/${TENSORBOARD_DIR}" \
         "${REPO}/${CHECKPOINT_DIR}" \
         "${REPO}/data/scene_datasets/mp3d" \
         "${REPO}/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat" \
         "${REPO}/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_v1"

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

echo "[pirlnav-deltaai-full] sif=${SIF}"
echo "[pirlnav-deltaai-full] repo=${REPO}"
echo "[pirlnav-deltaai-full] config=${CONFIG}"
echo "[pirlnav-deltaai-full] tag=${TAG}"
echo "[pirlnav-deltaai-full] main=${MAIN_ADDR}:${MAIN_PORT}"
echo "[pirlnav-deltaai-full] tasks=${SLURM_NTASKS:-unknown} gpus_per_node=${SLURM_GPUS_ON_NODE:-unknown}"
echo "[pirlnav-deltaai-full] num_updates=${NUM_UPDATES} num_envs=${NUM_ENVIRONMENTS} max_objects=${MAX_OBJECTS}"

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
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF "${INFLECTION_COEF}" \
    TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS "${MAX_OBJECTS}"
