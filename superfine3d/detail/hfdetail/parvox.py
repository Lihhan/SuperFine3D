"""Parallel, bit-exact flexible-dual-grid voxelisation (o_voxel's CPU voxeliser is single threaded).
The grid is split into slabs along x; slab s keeps the cells with x in [a_s, b_s) and receives every face whose
grid-space bounding box overlaps [a_s-1, b_s+1). A cell's dual vertex, boundary terms and edge-intersection flags depend
only on faces touching that cell, so the merged result equals the single call (verified by hashing: bench/vox_par_check.py).
Vertices/faces are passed through shared memory; each worker selects its own faces.
voxelize(v, f, res, nproc) -> (idx, dual, inter) torch CPU tensors in lexicographic (x, y, z) order (each worker sorts its slab).
(The shape encoder's output does not depend on the voxel order: bench/test_order_invariance.py.)"""
import numpy as np, torch, multiprocessing as mp
from multiprocessing import shared_memory

def _work(args):
    nv, nf, vname, fname, res, a, b = args
    import torch, o_voxel
    sv = shared_memory.SharedMemory(name=vname); sf = shared_memory.SharedMemory(name=fname)
    v = np.ndarray((nv, 3), np.float32, buffer=sv.buf); f = np.ndarray((nf, 3), np.int32, buffer=sf.buf)
    gx = (v[:, 0] + .5) * res; fx = gx[f]; lo = np.floor(fx.min(1)); hi = np.floor(fx.max(1))
    ff = np.ascontiguousarray(f[(hi >= a - 1) & (lo <= b)])
    idx, dual, inter = o_voxel.convert.mesh_to_flexible_dual_grid(vertices=torch.from_numpy(np.array(v)), faces=torch.from_numpy(ff), grid_size=res,
        aabb=[[-.5] * 3, [.5] * 3], face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    k = (idx[:, 0] >= a) & (idx[:, 0] < b)
    i, d, t = idx[k].numpy(), dual[k].numpy(), inter[k].numpy()
    o = np.argsort((i[:, 0].astype(np.int64) * res + i[:, 1]) * res + i[:, 2], kind='stable')   # lexicographic (x,y,z) within the slab
    out = i[o], d[o], t[o]
    del v, f; sv.close(); sf.close(); return out

def _init():
    import torch, o_voxel  # noqa: F401  (import cost paid when the pool starts, not on the first call)

def start(nproc):
    """Start (or reuse) the worker pool; workers import torch/o_voxel up front."""
    if nproc not in _POOL: _POOL[nproc] = mp.get_context('spawn').Pool(nproc, initializer=_init)
    return _POOL[nproc]

_POOL = {}
def voxelize(v, f, res, nproc=8):
    v = np.ascontiguousarray(v.float().numpy() if torch.is_tensor(v) else v, np.float32)
    f = np.ascontiguousarray(f.int().numpy() if torch.is_tensor(f) else f, np.int32)
    cx = np.floor((v[f[::max(1, len(f) // 2000000), 0], 0] + .5) * res)        # slab edges balanced on face x (subsample)
    edges = np.unique(np.quantile(cx, np.linspace(0, 1, nproc + 1)).astype(np.int64)); edges[0] = 0; edges[-1] = res
    sv = shared_memory.SharedMemory(create=True, size=v.nbytes); sf = shared_memory.SharedMemory(create=True, size=f.nbytes)
    try:
        np.ndarray(v.shape, v.dtype, buffer=sv.buf)[:] = v; np.ndarray(f.shape, f.dtype, buffer=sf.buf)[:] = f
        out = start(nproc).map(_work, [(len(v), len(f), sv.name, sf.name, res, int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])])
    finally:
        sv.close(); sv.unlink(); sf.close(); sf.unlink()
    # slabs cover disjoint, increasing x ranges and each is sorted, so the concatenation is in lexicographic (x, y, z) order
    idx = np.concatenate([o[0] for o in out]); dual = np.concatenate([o[1] for o in out]); inter = np.concatenate([o[2] for o in out])
    return torch.from_numpy(idx), torch.from_numpy(dual), torch.from_numpy(inter)

class Pending:
    """voxelize_async handle: the same tasks as voxelize(v, f, res, nproc), submitted without waiting (several meshes can be in
    flight in one larger pool); get() returns exactly what voxelize would."""
    def __init__(s, v, f, res, nproc, pool):
        v = np.ascontiguousarray(v.float().numpy() if torch.is_tensor(v) else v, np.float32)
        f = np.ascontiguousarray(f.int().numpy() if torch.is_tensor(f) else f, np.int32)
        cx = np.floor((v[f[::max(1, len(f) // 2000000), 0], 0] + .5) * res)
        edges = np.unique(np.quantile(cx, np.linspace(0, 1, nproc + 1)).astype(np.int64)); edges[0] = 0; edges[-1] = res
        s.sv = shared_memory.SharedMemory(create=True, size=v.nbytes); s.sf = shared_memory.SharedMemory(create=True, size=f.nbytes)
        np.ndarray(v.shape, v.dtype, buffer=s.sv.buf)[:] = v; np.ndarray(f.shape, f.dtype, buffer=s.sf.buf)[:] = f
        s.ar = start(pool).map_async(_work, [(len(v), len(f), s.sv.name, s.sf.name, res, int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])])
    def get(s):
        try: out = s.ar.get()
        finally: s.sv.close(); s.sv.unlink(); s.sf.close(); s.sf.unlink()
        idx = np.concatenate([o[0] for o in out]); dual = np.concatenate([o[1] for o in out]); inter = np.concatenate([o[2] for o in out])
        return torch.from_numpy(idx), torch.from_numpy(dual), torch.from_numpy(inter)

def voxelize_async(v, f, res, nproc=8, pool=None):
    """Like voxelize, but returns a Pending handle at once; pool = worker count of the pool used (default nproc)."""
    return Pending(v, f, res, nproc, pool or nproc)

def close():
    for p in _POOL.values(): p.close(); p.join()
    _POOL.clear()
