#!/usr/bin/env bash
# Build the dependency-only PIRLNav Apptainer image for NCSA Delta AI.
#
# Defaults:
#   - Builds containers/pirlnav-deltaai.def into pirlnav-deltaai.sif.
#   - Uses `apptainer build --fakeroot --ignore-fakeroot-command`. Delta AI
#     login nodes have user namespaces enabled but most users are not in
#     /etc/subuid, so Apptainer falls back to bind-mounting the host's
#     fakeroot binary into the container; that binary's glibc is newer than
#     the Ubuntu 22.04 base image and fails to load. --ignore-fakeroot-command
#     stops that bind-mount and lets %post run as fake root via user
#     namespaces alone, which is what we want.
#   - Redirects APPTAINER_CACHEDIR / APPTAINER_TMPDIR off of $HOME so the
#     build does not blow out the small home quota; prefers $WORK if set,
#     else falls back to $HOME.
#
# Override examples:
#   IMAGE=/projects/bgon/$USER/images/pirlnav-deltaai.sif bash containers/build_pirlnav_deltaai_sif.sh
#   FAKEROOT_FLAGS="--fakeroot" bash containers/build_pirlnav_deltaai_sif.sh           # try without --ignore-fakeroot-command
#   FAKEROOT_FLAGS="" bash containers/build_pirlnav_deltaai_sif.sh                     # disable fakeroot entirely (only works inside an interactive job)
#   FAKEROOT_FLAGS="--fakeroot --ignore-fakeroot-command --no-https" bash ...          # custom multi-flag override
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-pirlnav-deltaai.sif}"
DEF="${DEF:-containers/pirlnav-deltaai.def}"
BUILDER="${BUILDER:-}"
# Space-separated list of flags passed to `apptainer build`. Default works
# for unprivileged users on Delta AI login nodes.
FAKEROOT_FLAGS="${FAKEROOT_FLAGS:---fakeroot --ignore-fakeroot-command}"

if [[ -z "${BUILDER}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    BUILDER=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    BUILDER=singularity
  else
    echo "Neither apptainer nor singularity is available on PATH." >&2
    exit 1
  fi
fi

CACHE_BASE="${WORK:-$HOME}"
: "${APPTAINER_CACHEDIR:=${CACHE_BASE}/.apptainer}"
: "${APPTAINER_TMPDIR:=${CACHE_BASE}/apptainer-tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
export APPTAINER_CACHEDIR APPTAINER_TMPDIR

# Split FAKEROOT_FLAGS into an array so multiple flags pass through correctly.
read -r -a FAKEROOT_FLAG_ARR <<< "${FAKEROOT_FLAGS}"

echo "[build] BUILDER=${BUILDER}"
echo "[build] IMAGE=${IMAGE}"
echo "[build] DEF=${DEF}"
echo "[build] APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
echo "[build] APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"
echo "[build] FAKEROOT_FLAGS=${FAKEROOT_FLAGS:-<unset>}"

set -x
"${BUILDER}" build "${FAKEROOT_FLAG_ARR[@]}" "${IMAGE}" "${DEF}"
