#!/usr/bin/env bash
#SBATCH --job-name=pirlnav-il-dino-oc-full
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --account=bgon-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#
# NCSA Delta AI 1-node x 4-GPU PIRLNav IL run for the online-DINOv2 (@ 252x252)
# + object-cloud variants on the *full* MP3D THDA 70k -> 21cat bundle
# (56 scenes, 60085 episodes). Sibling of
# slurm_train_pirlnav_deltaai_full_dinov2.sh; this one adds the object-cloud
# branch (MAX_OBJECTS=100, original-size point transformer). Works for both
# the online-cloud config (default; adds DEPTH + SEMANTIC sensors) and the
# oracle-reveal config (pass CONFIG=...oracle_reveal...; RGB-only, reads the
# vertex caches in data/object_clouds_ply/). Stages MP3D + SIF to /tmp on the
# compute node for fast scene loads and auto-resumes across 48h submissions.
#
# Throughput note: the dino-only sibling measured ~1100-1140 fps aggregate at
# N=40/rank. Adding semantic rendering + per-step object-cloud accumulation
# lowers fps, so the 500M-step budget spreads over more 48h submissions.
# env-steps/update is set by num_steps/sync_frac/ranks (NOT by the cloud), so
# NUM_UPDATES below stays valid. If the interactive 40-env probe OOMs, lower
# NUM_ENVIRONMENTS and bump NUM_UPDATES to keep ~500M steps:
#   NUM_UPDATES ~= 500e6 / (NUM_ENVIRONMENTS * 231.6)   # 40->54000, 32->67500
#
# Submit (fresh):
#   sbatch scripts/slurm_train_pirlnav_deltaai_full_dinov2_object_cloud.sh
#
# Resubmit (auto-resume; same TAG, same CHECKPOINT_FOLDER):
#   sbatch scripts/slurm_train_pirlnav_deltaai_full_dinov2_object_cloud.sh
#
# Override anything via env var on submit, e.g.:
#   NUM_ENVIRONMENTS=32 NUM_UPDATES=67500 MAX_OBJECTS=100 \
#     sbatch scripts/slurm_train_pirlnav_deltaai_full_dinov2_object_cloud.sh

set -euo pipefail

SIF="${SIF:-/projects/bgon/brabiei/images/pirlnav-deltaai.sif}"
REPO="${REPO:-/u/brabiei/projects/pirlnav}"
DATA_SRC="${DATA_SRC:-/projects/bgon/brabiei/MP3D}"

CONFIG="${CONFIG:-configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full_252.yaml}"
TAG="${TAG:-mp3d_full_dinov2_object_cloud_252_4gpu_500M}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-40}"
# Training stops on UPDATES (percent_done = num_updates_done / NUM_UPDATES).
# Empirically the dino-only sibling lands ~9267 env-steps/update at N=40/rank
# (not 40*64*4=10240: DDPPO sync_frac=0.6 preempts straggler rollouts early).
#   ~500M env-steps / 9267 ~= 53953 -> 54000 (headroom; ~500.5M).
# The object-cloud branch does not change env-steps/update (only fps), so this
# value carries over. It persists across resumes via PIRLNAV_OVERRIDE_NUM_UPDATES.
# Keep this >= current num_updates_done or a bare resume would exit immediately.
NUM_UPDATES="${NUM_UPDATES:-54000}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-60}"
MAX_SCENE_REPEAT_STEPS="${MAX_SCENE_REPEAT_STEPS:-25000}"
INFLECTION_COEF="${INFLECTION_COEF:-3.513870128085}"
MAX_OBJECTS="${MAX_OBJECTS:-100}"
REPLAY_MODE="${REPLAY_MODE:-poses}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-tb/objectnav_il/${TAG}/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/new_checkpoints_dinov2_object_cloud_full_252/objectnav_il/${TAG}/}"
MAIN_PORT="${MAIN_PORT:-8738}"

# /tmp staging targets on the compute node. /tmp is node-local NVMe (~3.5 TB
# on a ghx4 node) and is wiped at job end, so we re-stage on every job; the
# parallel cp of MP3D (~22 GB) + SIF (~9.8 GB) takes ~55 s on a cold node and
# is a no-op on subsequent reuse within the same job.
LOCAL_SIF="/tmp/$(basename "${SIF}")"
LOCAL_DATA="/tmp/MP3D"

if [[ ! -f "${SIF}" ]]; then
  echo "SIF does not exist: ${SIF}" >&2
  exit 1
fi
if [[ ! -d "${REPO}" ]]; then
  echo "Repo does not exist: ${REPO}" >&2
  exit 1
fi
if [[ ! -d "${DATA_SRC}/scenes" ]]; then
  echo "DATA_SRC layout unexpected (no scenes/ under ${DATA_SRC})" >&2
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

mkdir -p "${REPO}/slurm_logs" \
         "${REPO}/${TENSORBOARD_DIR}" \
         "${REPO}/${CHECKPOINT_DIR}" \
         "${REPO}/data/scene_datasets/mp3d" \
         "${REPO}/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat" \
         "${REPO}/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_v1"

echo "[pirlnav-dino-oc-full] staging to /tmp ($(date +%T))"
if [[ -f "${LOCAL_SIF}" ]]; then
  echo "[pirlnav-dino-oc-full]   SIF already on /tmp, skipping"
else
  time cp "${SIF}" "${LOCAL_SIF}"
fi
if [[ -d "${LOCAL_DATA}/scenes" && -d "${LOCAL_DATA}/demo_episodes" && -d "${LOCAL_DATA}/eval_episodes" ]]; then
  echo "[pirlnav-dino-oc-full]   MP3D already on /tmp, skipping"
else
  mkdir -p "${LOCAL_DATA}"
  time (
    cp -r "${DATA_SRC}/scenes"        "${LOCAL_DATA}/" &
    cp -r "${DATA_SRC}/demo_episodes" "${LOCAL_DATA}/" &
    cp -r "${DATA_SRC}/eval_episodes" "${LOCAL_DATA}/" &
    wait
  )
fi
du -sh "${LOCAL_DATA}"/* "${LOCAL_SIF}"

BIND_ARGS=(--bind "${REPO}:/workspace/pirlnav"
           --bind "${LOCAL_DATA}/scenes/mp3d:/workspace/pirlnav/data/scene_datasets/mp3d"
           --bind "${LOCAL_DATA}/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat:/workspace/pirlnav/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat"
           --bind "${LOCAL_DATA}/eval_episodes:/workspace/pirlnav/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_v1")

export MAIN_ADDR="${MAIN_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')}"
export MAIN_PORT
export PYTHONUNBUFFERED=1
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export NCCL_ASYNC_ERROR_HANDLING=1
# Make NUM_UPDATES authoritative across resumes: the trainer reloads the
# saved config on resume (which pins the original budget), so we surface the
# desired total here and the trainer applies it after loading resume state.
# Ignored on a fresh run. APPTAINERENV_* ensures apptainer forwards it into
# the container regardless of host env-passthrough settings.
export PIRLNAV_OVERRIDE_NUM_UPDATES="${NUM_UPDATES}"
export APPTAINERENV_PIRLNAV_OVERRIDE_NUM_UPDATES="${NUM_UPDATES}"
export SINGULARITYENV_PIRLNAV_OVERRIDE_NUM_UPDATES="${NUM_UPDATES}"

RESUME_FILE="${REPO}/${CHECKPOINT_DIR}.habitat-resume-state.pth"
if [[ -f "${RESUME_FILE}" ]]; then
  echo "[pirlnav-dino-oc-full] resume-state present -> trainer will auto-resume from ${RESUME_FILE}"
else
  echo "[pirlnav-dino-oc-full] no resume-state -> fresh run"
fi

echo "[pirlnav-dino-oc-full] sif=${LOCAL_SIF}"
echo "[pirlnav-dino-oc-full] data=${LOCAL_DATA}"
echo "[pirlnav-dino-oc-full] config=${CONFIG} tag=${TAG}"
echo "[pirlnav-dino-oc-full] num_envs=${NUM_ENVIRONMENTS} num_updates=${NUM_UPDATES} num_checkpoints=${NUM_CHECKPOINTS}"
echo "[pirlnav-dino-oc-full] max_objects=${MAX_OBJECTS} replay_mode=${REPLAY_MODE}"
echo "[pirlnav-dino-oc-full] max_scene_repeat_steps=${MAX_SCENE_REPEAT_STEPS} inflection_coef=${INFLECTION_COEF}"
echo "[pirlnav-dino-oc-full] main=${MAIN_ADDR}:${MAIN_PORT}"
echo "[pirlnav-dino-oc-full] tasks=${SLURM_NTASKS:-unknown} gpus_per_node=${SLURM_GPUS_ON_NODE:-unknown}"

srun "${CONTAINER}" exec --nv \
  "${BIND_ARGS[@]}" \
  --pwd /workspace/pirlnav \
  "${LOCAL_SIF}" \
  python -u -m run \
    --exp-config "${CONFIG}" \
    --run-type train \
    TENSORBOARD_DIR "${TENSORBOARD_DIR}" \
    CHECKPOINT_FOLDER "${CHECKPOINT_DIR}" \
    NUM_UPDATES "${NUM_UPDATES}" \
    NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}" \
    NUM_CHECKPOINTS "${NUM_CHECKPOINTS}" \
    RL.DDPPO.force_distributed True \
    IL.BehaviorCloning.REPLAY_MODE "${REPLAY_MODE}" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF "${INFLECTION_COEF}" \
    TASK_CONFIG.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS "${MAX_OBJECTS}" \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS "${MAX_SCENE_REPEAT_STEPS}"
