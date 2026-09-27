#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${EGOSMPLX_CONTROLLER_PYTHON:-python3}" -m egosmplx_pipeline "${1:-check}" --config "${EGOSMPLX_CONFIG_FILE:-configs/pipeline.example.json}" "${@:2}"
