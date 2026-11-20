"""Warm Pixal3D 512 process for bench_t1 (OPT_PXWARM=1). Started before the timed window, it imports what fig5/pixal512_worker.py
and the official inference code import (interpreter start-up and imports are outside the timed window, as for the main process),
writes OUT/pixal_ready, and waits for one line on stdin. Then it runs the unchanged worker (runpy; model loads, MoGe camera,
sampling and decoding are all inside the timed window), records OUT/peak.json and marks OUT/pixal_done (OUT/pixal_failed on error).
With PIXAL_LEAN=1 the worker writes OUT/zup/mesh_512.ply directly (byte-identical to bench/pixal_peak.py's conversion).
usage (cwd fig5, Pixal3D venv): pixal_warm.py INPUT OUT"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys, os, time, json, runpy
INP, O = sys.argv[1], sys.argv[2]; os.makedirs(f'{O}/zup', exist_ok=True)
base = f'{_E["SF3D_PIXAL3D"]}'; sys.path.insert(0, f'{base}/repo')
import torch, numpy as np, trimesh                                                     # noqa: F401  (imports only)
import inference                                                                        # noqa: F401
from pixal3d.pipelines import Pixal3DImageTo3DPipeline                                 # noqa: F401
import moge.model.v2                                                                    # noqa: F401  (imported lazily by the camera step)
import pixal3d.trainers.flow_matching.mixins.image_conditioned_proj                     # noqa: F401  (imported lazily by the image conditioning)
if os.environ.get('PIXAL_NOINIT') == '1': sys.path.insert(0, f'{_E["SF3D_ROOT"]}/superfine3d/detail/bench'); import noinit  # noqa: F401
open(f'{O}/pixal_ready', 'w').close()
sys.stdin.readline(); t = time.time()
sys.argv = ['pixal512_worker.py', INP, O]
try:
    runpy.run_path(f'{_E["SF3D_ROOT"]}/superfine3d/detail/fig5/pixal512_worker.py', run_name='__main__')
except BaseException:
    open(f'{O}/pixal_failed', 'w').close(); raise
json.dump(dict(peak_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None, seconds=time.time() - t), open(f'{O}/peak.json', 'w'))
if os.environ.get('PIXAL_LEAN') != '1':   # same Z-up copy as bench/pixal_peak.py
    m = trimesh.load(f'{O}/mesh_512_native.ply', process=False); v = np.asarray(m.vertices, dtype=np.float32); v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
    trimesh.Trimesh(v, np.asarray(m.faces), process=False).export(f'{O}/zup/mesh_512.ply')
import shutil; shutil.copy(INP, f'{O}/zup/input.png')
os.replace(f'{O}/pixal_ready', f'{O}/pixal_done')
