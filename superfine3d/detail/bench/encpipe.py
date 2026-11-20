"""Exact pipelined hfdetail.latent_io.encode_slabs for bench_t1 (OPT_ENCPIPE=1). Same slabs, same selected faces and the same
per-slab encode_direct calls as encode_slabs; only the scheduling changes: every slab's voxelisation is submitted at once to one
parvox pool (POOL workers, NPROC tasks per slab, exactly the tasks of the synchronous call), and each slab's encoder pass starts
as soon as its voxels are ready, overlapping the remaining voxelisations. The voxeliser hook (bench _Conv) returns PRE[0] when set.
Checked against encode_slabs on a dumped 4096 input (bench/test_encpipe.py)."""
import numpy as np, torch
from hfdetail import latent_io, parvox
PRE = []

def encode_slabs(v, f, res, nslab, nproc, pool, margin=4, axis=None):
    if nslab <= 1: return latent_io.encode_direct(v, f, res)
    n_tok = res // 16; vc = v.float().cuda(); fc = f.long().cuda()
    ax = int(latent_io.torch_argmax_extent(vc)) if axis is None else axis
    tri = vc[fc][..., ax]; tlo = tri.min(1).values; thi = tri.max(1).values; del tri
    lo_t = int(((vc[:, ax].min() + .5) * n_tok).floor()); hi_t = int(((vc[:, ax].max() + .5) * n_tok).floor()) + 1
    cuts = np.linspace(lo_t, hi_t, nslab + 1).round().astype(int); jobs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        x0 = (a - margin) / n_tok - .5; x1 = (b + margin) / n_tok - .5
        sel = (tlo >= x0) & (thi <= x1)
        if not sel.any(): continue
        fs = fc[sel]; uv_, inv = torch.unique(fs, return_inverse=True)
        vs, fs_ = vc[uv_].cpu(), inv.reshape(-1, 3).cpu()
        jobs.append((a, b, vs, fs_, parvox.voxelize_async(vs.float(), fs_.long(), res, nproc, pool)))
    C = []; Fe = []
    for a, b, vs, fs_, pend in jobs:
        PRE.append(pend); c, fe = latent_io.encode_direct(vs, fs_, res); assert not PRE, 'pre-voxelised slab not consumed'
        keep = (c[:, 1 + ax] >= a) & (c[:, 1 + ax] < b); C.append(c[keep]); Fe.append(fe[keep])
        torch.cuda.empty_cache()
    return np.concatenate(C), np.concatenate(Fe)
