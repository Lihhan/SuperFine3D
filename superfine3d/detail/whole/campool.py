"""Exact parallel camera search in a pre-started process pool (same per-start computation as campar.py: each worker runs
fit_input_camera.py / camera_refine.py with *_ONLY=i, *_PARTDIR=dir; the combine pass in the caller applies the scripts'
own selection). Workers import torch / nvdiffrast / scipy when the pool starts, so the pipeline pays no interpreter or
import start-up at camera time.  start(n) early; then run(script, argv, nstart, var, partdir, env) per search."""
import os, sys, runpy, multiprocessing as mp

_POOL = {}


def _init(paths):
    sys.path[:0] = [p for p in paths if p not in sys.path]
    import numpy, torch, scipy.optimize, trimesh, PIL.Image  # noqa: F401
    import nvdiffrast.torch  # noqa: F401
    from hfdetail import raster, camera  # noqa: F401
    import pixcam  # noqa: F401


def _task(args):
    script, argv, env = args
    import torch
    if torch.cuda.is_initialized(): torch.cuda.reset_peak_memory_stats()
    old_env = {k: os.environ.get(k) for k in env}; os.environ.update(env); old_argv = sys.argv; sys.argv = [script] + list(argv)
    try:
        runpy.run_path(script, run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0): raise
    finally:
        sys.argv = old_argv
        for k, v in old_env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    return peak


def start(n, paths=()):
    if n not in _POOL: _POOL[n] = mp.get_context('spawn').Pool(n, initializer=_init, initargs=(list(paths) + list(sys.path),))
    return _POOL[n]


def run(n, script, argv, nstart, var, partdir, env=None):
    """Run starts 0..nstart-1 of `script` in the pool; partial results land in partdir/part{i}.json.
    Returns the peak allocated GPU bytes of each start (workers run concurrently)."""
    os.makedirs(partdir, exist_ok=True); env = dict(env or {})
    jobs = [(script, argv, {**env, var + '_ONLY': str(i), var + '_PARTDIR': partdir}) for i in range(nstart)]
    return start(n).map(_task, jobs, chunksize=1)


def close():
    for p in _POOL.values(): p.close(); p.join()
    _POOL.clear()
