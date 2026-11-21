#!/bin/bash
# Optional speed-up of the pipeline: AOTInductor builds of the detail-estimator networks that are re-run at every level
# (about 3-4 minutes once; GPU- and torch-version specific). Without them the pipeline uses the eager networks (same settings).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SF3D_CACHE/inductor}"
"$SF3D_PYTHON" "$SF3D_ROOT/setup/build_estimator_aot.py"
