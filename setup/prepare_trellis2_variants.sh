#!/bin/bash
# Materialize the TRELLIS.2 fork used for training: upstream TRELLIS.2 (commit 75fbf01: trellis2/, data_toolkit/, train.py)
# with the files committed in training/trellis2_variants laid over it (existing files are never overwritten).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
D="$SF3D_ROOT/training/trellis2_variants"
[ -d "$SF3D_TRELLIS2_REPO/trellis2" ] || { echo "SF3D_TRELLIS2_REPO does not contain trellis2/: $SF3D_TRELLIS2_REPO"; exit 1; }
git -C "$SF3D_TRELLIS2_REPO" rev-parse --short HEAD 2>/dev/null | grep -q '^75fbf01' || echo "warning: TRELLIS.2 is not at commit 75fbf01"
"$SF3D_PYTHON" - "$SF3D_TRELLIS2_REPO" "$D" <<'PY'
import os, shutil, sys
src, dst = sys.argv[1:3]; n = 0
for top in ('trellis2', 'data_toolkit', 'train.py'):
    s = os.path.join(src, top)
    items = [(s, os.path.join(dst, top))] if os.path.isfile(s) else [
        (os.path.join(r, f), os.path.join(dst, top, os.path.relpath(os.path.join(r, f), s))) for r, _, fs in os.walk(s) for f in fs]
    for a, b in items:
        if '__pycache__' in a or os.path.exists(b): continue
        os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copy2(a, b); n += 1
print('copied', n, 'upstream files')
PY
echo "TRELLIS2_VARIANTS_READY $D"
