#!/usr/bin/env bash
# Build the dependency-only PIRLNav Singularity/Apptainer image locally.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-pirlnav.sif}"
DEF="${DEF:-containers/pirlnav.def}"
BUILDER="${BUILDER:-}"

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

set -x
"${BUILDER}" build "${IMAGE}" "${DEF}"
