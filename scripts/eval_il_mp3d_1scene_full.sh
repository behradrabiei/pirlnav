#!/usr/bin/env bash
# Full single-GPU, non-SLURM eval of one IL checkpoint on the MP3D 1-scene
# 6-category subset.  Runs both train and val splits back-to-back and prints a
# side-by-side summary (success / spl / inference_ms).
#
# All knobs live in configs/eval_overfit.env.  Just edit that file and run:
#   bash scripts/eval_il_mp3d_1scene_full.sh
#
# Optional: point to a different env file via the first positional arg, e.g.
#   bash scripts/eval_il_mp3d_1scene_full.sh configs/my_eval.env

set -euo pipefail

cd "$(dirname "$0")/.."

ENV_FILE="${1:-configs/eval_overfit.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[eval_full] env file not found: ${ENV_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${EVAL_CKPT:?EVAL_CKPT not set in ${ENV_FILE}}"
: "${EVAL_SPLITS:?EVAL_SPLITS not set in ${ENV_FILE}}"
: "${OUT_ROOT:?OUT_ROOT not set in ${ENV_FILE}}"
TEST_EPISODE_COUNT="${TEST_EPISODE_COUNT:--1}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-2}"
VIDEO_ENABLED="${VIDEO_ENABLED:-true}"
VIDEO_FAILED_ONLY="${VIDEO_FAILED_ONLY:-false}"
VIDEO_FPS="${VIDEO_FPS:-10}"
VIDEO_RENDER_TOP_DOWN="${VIDEO_RENDER_TOP_DOWN:-true}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-}"
ALLOW_SLIDING="${ALLOW_SLIDING:-}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "pirlnav" ]]; then
  # shellcheck disable=SC1091
  source /workspace/conda/etc/profile.d/conda.sh
  conda activate pirlnav
fi

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/experiments/il_objectnav_mp3d.yaml}"

# YACS wants VIDEO_OPTION as a literal list token.
if [[ "${VIDEO_ENABLED}" == "true" ]]; then
  VIDEO_OPTION_VAL='["disk"]'
else
  VIDEO_OPTION_VAL='[]'
fi

# YACS booleans need python-cased True/False.
bool_to_py() { [[ "$1" == "true" ]] && echo True || echo False; }
VIDEO_FAILED_ONLY_PY=$(bool_to_py "${VIDEO_FAILED_ONLY}")
VIDEO_RENDER_TOP_DOWN_PY=$(bool_to_py "${VIDEO_RENDER_TOP_DOWN}")

mkdir -p "${OUT_ROOT}"
SUMMARY_FILE="${OUT_ROOT}/summary.tsv"
: > "${SUMMARY_FILE}"
printf "split\tsuccess\tspl\tsoftspl\tdist_to_goal\tinfer_ms_per_step\tn_episodes\n" \
  >> "${SUMMARY_FILE}"

for SPLIT in ${EVAL_SPLITS}; do
  SPLIT_OUT="${OUT_ROOT}/${SPLIT}"
  VIDEO_DIR="${SPLIT_OUT}/videos"
  TB_DIR="${SPLIT_OUT}/tb"
  LOG_FILE="${SPLIT_OUT}/eval.log"
  mkdir -p "${VIDEO_DIR}" "${TB_DIR}"

  echo
  echo "================================================================"
  echo "[eval_full] split=${SPLIT}"
  echo "[eval_full] ckpt=${EVAL_CKPT}"
  echo "[eval_full] videos -> ${VIDEO_DIR}"
  echo "[eval_full] tb     -> ${TB_DIR}"
  echo "[eval_full] video_enabled=${VIDEO_ENABLED}  failed_only=${VIDEO_FAILED_ONLY}  top_down=${VIDEO_RENDER_TOP_DOWN}"
  echo "================================================================"

  # Build task-config overrides only if the user actually set them.  We must
  # pass these through as TASK_CONFIG.* since they live in the nested habitat
  # task config, and PIRLNav forwards everything.
  TASK_OVERRIDES=()
  if [[ -n "${SUCCESS_DISTANCE}" ]]; then
    TASK_OVERRIDES+=(
      TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE "${SUCCESS_DISTANCE}"
      TASK_CONFIG.TASK.SUCCESS_DISTANCE "${SUCCESS_DISTANCE}"
    )
  fi
  if [[ -n "${ALLOW_SLIDING}" ]]; then
    TASK_OVERRIDES+=(
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING "${ALLOW_SLIDING}"
    )
  fi

  python -u -m run \
    --exp-config "${CONFIG}" \
    --run-type eval \
    TENSORBOARD_DIR "${TB_DIR}" \
    EVAL_CKPT_PATH_DIR "${EVAL_CKPT}" \
    NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}" \
    TEST_EPISODE_COUNT "${TEST_EPISODE_COUNT}" \
    VIDEO_OPTION "${VIDEO_OPTION_VAL}" \
    VIDEO_DIR "${VIDEO_DIR}" \
    VIDEO_FPS "${VIDEO_FPS}" \
    VIDEO_RENDER_TOP_DOWN "${VIDEO_RENDER_TOP_DOWN_PY}" \
    EVAL.USE_CKPT_CONFIG False \
    EVAL.SPLIT "${SPLIT}" \
    EVAL.VIDEO_FAILED_ONLY "${VIDEO_FAILED_ONLY_PY}" \
    "${TASK_OVERRIDES[@]}" \
    ${EXTRA_OPTS} \
    2>&1 | tee "${LOG_FILE}"

  # Parse the trainer's "Average episode <metric>: <value>" log lines.
  grab() {
    local key="$1"
    grep -E "Average episode ${key}:" "${LOG_FILE}" | tail -n1 \
      | awk -F': ' '{print $NF}'
  }
  SUCC=$(grab success)
  SPL=$(grab spl)
  SSPL=$(grab softspl)
  DTG=$(grab distance_to_goal)
  INFER=$(grab avg_inference_time_ms)
  NEP=$(grab num_eval_episodes)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${SPLIT}" "${SUCC}" "${SPL}" "${SSPL}" "${DTG}" "${INFER}" "${NEP}" \
    >> "${SUMMARY_FILE}"
done

echo
echo "================================================================"
echo "[eval_full] summary (${SUMMARY_FILE})"
echo "================================================================"
column -t -s$'\t' "${SUMMARY_FILE}"
echo
