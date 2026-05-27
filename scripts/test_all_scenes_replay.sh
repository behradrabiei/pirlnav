#!/usr/bin/env bash
# Sweep scripts/test_scene_replay.py over EVERY MP3D scene in the 21-class
# THDA bundle, one scene per fresh Python subprocess. Subprocess isolation
# means a SIGSEGV / SIGABRT from habitat-sim on scene N only kills that one
# probe -- the driver records it and moves on.
#
# Run from inside the dependency-only Apptainer image, with the standard
# bind layout in place (see DELTAAI_CONTAINER.md):
#   - data/scene_datasets/mp3d                                          (90 .glb scenes)
#   - data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat/  (filtered demos)
#
# Usage (already inside the container, single-GPU salloc):
#   bash scripts/test_all_scenes_replay.sh                       # sweeps all scenes
#   bash scripts/test_all_scenes_replay.sh --scenes 17DRP5sb8fy 1pXnuDYAj8r
#   MAX_ACTIONS=100 bash scripts/test_all_scenes_replay.sh        # cap actions per ep
#
# Output:
#   - One line per scene: "PASS  17DRP5sb8fy  steps=...  d2g=..."
#                       or "FAIL  17DRP5sb8fy  reason=...  exit=..."
#   - A grouped summary at the end: passed / failed / no-replay / setup-fail
#     plus the explicit list of failed scene ids (so you can rerun just those).
#   - Per-scene full stdout/stderr saved to scene_loadability_logs/<scene>.log.
set -uo pipefail

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
# Match the training launcher: stay offline so habitat / pirlnav imports do
# not try to fetch DINOv2 weights from HF (we are not loading the policy
# anyway, but pirlnav.__init__ imports visual_policy which constructs the
# encoder lazily; safe to lock everything down).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CONTENT_DIR="data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat/train/content"
LOG_DIR="scene_loadability_logs"
mkdir -p "${LOG_DIR}"

# --- Build scene list ---------------------------------------------------------
declare -a SCENES=()
if [[ "${1:-}" == "--scenes" ]]; then
  shift
  SCENES=("$@")
else
  if [[ ! -d "${CONTENT_DIR}" ]]; then
    echo "[test_all_scenes_replay] missing content dir: ${CONTENT_DIR}" >&2
    echo "  is the THDA-21cat bind mounted? see DELTAAI_CONTAINER.md" >&2
    exit 1
  fi
  while IFS= read -r f; do
    base=$(basename "${f}" .json.gz)
    SCENES+=("${base}")
  done < <(find "${CONTENT_DIR}" -maxdepth 1 -name '*.json.gz' | sort)
fi

if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "[test_all_scenes_replay] no scenes to test" >&2
  exit 1
fi

echo "[test_all_scenes_replay] sweeping ${#SCENES[@]} scenes; logs -> ${LOG_DIR}/"
echo

# --- Run each scene in a fresh subprocess ------------------------------------
MAX_ACTIONS="${MAX_ACTIONS:-500}"
declare -a PASSED=()
declare -a FAILED=()
declare -a CRASHED=()
declare -a NOREPLAY=()
declare -a SETUPFAIL=()

t_sweep_start=$(date +%s)
for i in "${!SCENES[@]}"; do
  scene="${SCENES[$i]}"
  log="${LOG_DIR}/${scene}.log"
  printf "[%2d/%2d] %s ... " "$((i + 1))" "${#SCENES[@]}" "${scene}"
  set +e
  python -u scripts/test_scene_replay.py \
    --scene "${scene}" \
    --max-actions "${MAX_ACTIONS}" >"${log}" 2>&1
  rc=$?
  set -e

  case "${rc}" in
    0)
      PASSED+=("${scene}")
      summary=$(grep -E "^\[test_scene_replay\] ${scene}: PASS" "${log}" | tail -n 1)
      echo "PASS  ${summary#*PASS }"
      ;;
    2)
      NOREPLAY+=("${scene}")
      echo "SKIP  no episode with reference_replay (exit=2)"
      ;;
    3)
      SETUPFAIL+=("${scene}")
      echo "SETUP-FAIL  see ${log} (exit=3)"
      ;;
    4)
      FAILED+=("${scene}")
      echo "PYFAIL  python exception, see ${log} (exit=4)"
      ;;
    *)
      # Bash surfaces "killed by signal N" as exit 128+N.
      if (( rc >= 128 )); then
        sig=$((rc - 128))
        CRASHED+=("${scene}:SIG${sig}")
        echo "CRASH  signal ${sig} (exit=${rc}); see ${log}"
      else
        FAILED+=("${scene}")
        echo "FAIL  exit=${rc}; see ${log}"
      fi
      ;;
  esac
done
t_sweep_end=$(date +%s)

# --- Summary -----------------------------------------------------------------
echo
echo "============================================================"
echo "scene-loadability sweep summary  (${#SCENES[@]} scenes, $((t_sweep_end - t_sweep_start))s total)"
echo "  PASS         : ${#PASSED[@]}"
echo "  PYFAIL       : ${#FAILED[@]}"
echo "  CRASH(signal): ${#CRASHED[@]}"
echo "  NO-REPLAY    : ${#NOREPLAY[@]}"
echo "  SETUP-FAIL   : ${#SETUPFAIL[@]}"
echo "============================================================"

print_list() {
  local label="$1"; shift
  local arr=("$@")
  if (( ${#arr[@]} > 0 )); then
    echo "${label}:"
    for s in "${arr[@]}"; do echo "  ${s}"; done
  fi
}
print_list "Failed scenes (Python exception)" "${FAILED[@]:-}"
print_list "Crashed scenes (native signal)"   "${CRASHED[@]:-}"
print_list "No-replay scenes"                  "${NOREPLAY[@]:-}"
print_list "Setup-fail scenes"                 "${SETUPFAIL[@]:-}"

if (( ${#FAILED[@]} > 0 || ${#CRASHED[@]} > 0 )); then
  exit 1
fi
exit 0
