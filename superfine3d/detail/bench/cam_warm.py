"""Warm camera driver of the Pixal3D chain for bench_t1 (OPT_PXWARM=1). Started before the timed window under the refine
environment (bash setup/python.sh cam_warm.py OUT; same interpreter and numpy/scipy/trimesh/PIL as the fit's environment),
it imports the camera scripts' modules, writes OUT/cam_ready and waits for one line on stdin (the start of the timed window).
Then it waits for OUT/pixal_done and runs fit_input_camera.py and whole/camera_refine.py in this process (runpy, the unchanged
scripts) with exactly pixal_chain.sh's settings, and writes OUT/chain_times.json. usage: cam_warm.py OUT"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys, os, time, json, runpy
O = sys.argv[1]; P = f'{_E["SF3D_ROOT"]}/superfine3d/detail'; sys.path.insert(0, P)
import numpy, scipy.optimize, scipy.ndimage, scipy.spatial, trimesh, torch, PIL.Image, torch.nn.functional  # noqa: F401
import pixcam                                                                  # noqa: F401
from hfdetail import raster, camera                                            # noqa: F401
import nvdiffrast.torch                                                         # noqa: F401
open(f'{O}/cam_ready', 'w').close()
sys.stdin.readline(); t0 = time.time()
while not os.path.exists(f'{O}/pixal_done'):
    if os.path.exists(f'{O}/pixal_failed'): print('PIXAL_WORKER_FAILED', flush=True); sys.exit(3)
    time.sleep(0.02)
t1 = time.time(); NT = int(os.environ.get('PXCHAIN_THREADS', '4'))
os.environ.update(OMP_NUM_THREADS='4', CAMFIT_THREADS=str(min(NT, 4)), CAMREF_THREADS=str(NT), CAMREF_HULL='1', CAMREF_LEAN='1')   # pixal_chain.sh
if os.environ.get('PXCHAIN_FAST') == '1':
    os.environ.update(CAMFIT_FASTNORM='2', CAMREF_THREADS='12', CAMREF_LOCKSTEP='1', CAMREF_FNCACHE='1', CAMREF_MEMO='1', CAMREF_GRIDBATCH='72', CAMREF_FASTNORM='1')
def run(path, argv):
    old = sys.argv; sys.argv = [path] + argv
    try: runpy.run_path(path, run_name='__main__')
    except SystemExit as e:
        if e.code not in (0, None): raise
    finally: sys.argv = old
Z = f'{O}/zup'
run(f'{P}/fit_input_camera.py', [f'{Z}/input.png', f'{Z}/mesh_512.ply', f'{Z}/cam_fit.json']); t2 = time.time()
run(f'{P}/whole/camera_refine.py', [f'{Z}/input.png', f'{Z}/mesh_512.ply', f'{Z}/cam_fit.json', f'{Z}/cam_refined.json']); t3 = time.time()
json.dump(dict(pixal3d_512=t1 - t0, camera_fit=t2 - t1, camera_refine=t3 - t2, total=t3 - t0), open(f'{O}/chain_times.json', 'w'), indent=1)
print('CHAIN_DONE', flush=True)
