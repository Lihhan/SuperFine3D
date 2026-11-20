#!/bin/bash
# Run a Python script in the training/tooling environment: the TRELLIS.2 fork (training/trellis2_variants, see
# setup/prepare_trellis2_variants.sh) and the stage modules come before upstream TRELLIS.2.
# usage: bash setup/python.sh [-u] SCRIPT.py ARGS...   (or -m MODULE)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
R="$SF3D_ROOT"
export PYTHONPATH="$R/training/refiner:$R/superfine3d/refiner/code:$R/third_party/site-packages:$R/training/trellis2_variants:$SF3D_TRELLIS2_REPO:$SF3D_TRELLIS2_REPO/o-voxel"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}" TRELLIS_DATA_WORKERS="${TRELLIS_DATA_WORKERS:-4}"
ulimit -n 65535 2>/dev/null || true
exec "$SF3D_PYTHON" "$@"
