"""Encoding of large meshes into a global high-resolution shape latent.

encode_direct (default): one global encode on the GPU. The stock encoder breaks above 1024^3 because
SparseSpatial2Channel builds coordinate codes as int32 tensor x python int (2048^3 > 2^31 at the first 4096 downsample),
and o_voxel sizes its hash map with an int32 volume; patch_int64() fixes both at runtime (no source change).
encode_tiled (fallback, slow): 1024-resolution crop encodes stitched together, see below.

Tiled encoding:
The stock shape encoder overflows its int32 spatial hashing above 1024^3 (encode at 4096 returns garbage tokens), so a
4096 (or 2048) latent is built from 1024-resolution crop encodes, exactly like the zoom training data
(pairs.encode at 1024 in the crop frame): crops of 64 tokens, aligned with the global token grid
(res/16 tokens per axis), stride 64-2*margin tokens; each crop keeps only its interior tokens (margin tokens from every
cut face are discarded, except at the grid border where no neighbour exists). Faces are not clipped (pairs.crop_mesh
does that exactly but in float64 numpy, ~15 min per 42M-face mesh): only faces lying entirely inside the crop cube are
encoded; faces crossing the cut are dropped, which only affects voxels next to the cut, i.e. inside the discarded margin.
usage: coords, feats = encode_tiled(v, f, res=4096)   (v object frame in [-.5,.5], f faces; CPU or CUDA tensors)"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys, numpy as np, torch
sys.path.insert(0, f'{_E["SF3D_ROOT"]}/superfine3d/encode')

def crop_origins(lo_tok, hi_tok, n_tok, size=64, margin=4):
    """Token-grid origins along one axis so that the interiors [o+margin, o+size-margin) cover [lo_tok, hi_tok]."""
    st = size - 2 * margin; o = max(0, lo_tok - margin); out = []
    while True:
        out.append(min(o, n_tok - size))
        if out[-1] + size - margin > hi_tok: break
        o += st
    return sorted(set(out))

def encode_tiled(v, f, res=4096, margin=4, log=None):
    import pairs
    n_tok = res // 16; zoom = res / 1024; size = 64
    v = v.float(); f = f.long(); vc = v.cuda(); fc = f.cuda()
    lo = ((vc.min(0).values + .5) * n_tok).floor().long().clamp(0, n_tok - 1).tolist(); hi = ((vc.max(0).values + .5) * n_tok).floor().long().clamp(0, n_tok - 1).tolist()
    O = [crop_origins(lo[k], hi[k], n_tok, size, margin) for k in range(3)]
    tri = vc[fc]; tlo = tri.min(1).values; thi = tri.max(1).values; del tri
    keys = {}; feats_all = []; coords_all = []
    for ox in O[0]:
        for oy in O[1]:
            for oz in O[2]:
                o = torch.tensor([ox, oy, oz], device='cuda'); cmin = o.float() / n_tok - .5; cmax = (o.float() + size) / n_tok - .5
                ctr = (cmin + cmax) / 2; h = .5 / zoom * (1 - 1e-5)
                sel = ((tlo >= ctr - h) & (thi <= ctr + h)).all(1)            # faces entirely inside the crop cube
                if not sel.any(): continue
                fs = fc[sel]; uv_, inv = torch.unique(fs, return_inverse=True)
                c, fe, _ = pairs.encode(((vc[uv_] - ctr) * zoom).cpu(), inv.reshape(-1, 3).cpu(), 1024)   # local 64^3 token grid
                c = torch.from_numpy(c).long(); g = c + o.cpu()
                keep = torch.ones(len(c), dtype=torch.bool)
                for k, ok in enumerate((ox, oy, oz)):
                    if ok > 0: keep &= c[:, k] >= margin                      # interior only, unless at the grid border
                    if ok + size < n_tok: keep &= c[:, k] < size - margin
                coords_all.append(g[keep]); feats_all.append(torch.from_numpy(fe)[keep])
                if log is not None: log.append((ox, oy, oz, int(keep.sum())))
    C = torch.cat(coords_all); Fe = torch.cat(feats_all)
    key = (C[:, 0] * n_tok + C[:, 1]) * n_tok + C[:, 2]; uk, first = np.unique(key.numpy(), return_index=True)   # overlap: keep one copy
    C, Fe = C[first], Fe[first]
    return np.concatenate([np.zeros((len(C), 1), np.int32), C.numpy().astype(np.int32)], 1), Fe.numpy()


_PATCHED = False
def patch_int64():
    """Runtime fixes so the shape encoder works at 2048/4096: int64 coordinate codes in SparseSpatial2Channel (coords
    returned as int32 again) and an int64 hash-map volume in o_voxel's flexible dual grid."""
    global _PATCHED
    if _PATCHED: return
    import inspect, textwrap, torch
    import trellis2.modules.sparse.spatial.spatial2channel as s2c
    src = textwrap.dedent(inspect.getsource(s2c.SparseSpatial2Channel.forward))
    a = 'coord = list(x.coords.unbind(dim=-1))'
    assert src.count(a) == 1, 'unexpected SparseSpatial2Channel.forward source'
    src = src.replace(a, 'coord = list(x.coords.long().unbind(dim=-1))')
    import re
    src, n = re.subn(r'(dim=-1\s*\n\s*\))', r'\1.to(x.coords.dtype)', src, count=1)
    assert n == 1, 'unexpected new_coords construction'
    ns = dict(s2c.__dict__); exec(compile(src, '<s2c-int64>', 'exec'), ns); s2c.SparseSpatial2Channel.forward = ns['forward']
    import o_voxel.convert.flexible_dual_grid as fdg
    src = textwrap.dedent(inspect.getsource(fdg._init_hashmap))
    src = src.replace('VOL = (grid_size[0] * grid_size[1] * grid_size[2]).item()', 'VOL = int(grid_size[0].item()) * int(grid_size[1].item()) * int(grid_size[2].item())')
    ns = dict(fdg._init_hashmap.__globals__); exec(compile(src, '<int64-volume>', 'exec'), ns); fdg._init_hashmap = ns['_init_hashmap']
    _PATCHED = True

def encode_direct(v, f, res):
    """Global encode at res (1024/2048/4096). Returns coords (N,4) int32 with batch column, feats (N,32)."""
    import pairs
    patch_int64(); c, fe, _ = pairs.encode(v.float().cpu(), f.long().cpu(), res)
    return np.concatenate([np.zeros((len(c), 1), np.int32), c.astype(np.int32)], 1), fe

def encode_slabs(v, f, res, nslab=2, margin=4, axis=None):
    """Memory-bounded global encode: split along the longest axis into nslab slabs (token-aligned), encode each slab's
    faces (entirely inside the slab widened by `margin` tokens) with encode_direct on the global grid, keep each slab's
    own token range. Equals encode_direct away from the cuts; tokens within `margin` of a cut see one-sided context
    beyond the widened slab only (receptive-field effect, see tests/test_latent_io.py)."""
    if nslab <= 1: return encode_direct(v, f, res)
    n_tok = res // 16; vc = v.float().cuda(); fc = f.long().cuda()
    ax = int(torch_argmax_extent(vc)) if axis is None else axis
    tri = vc[fc][..., ax]; tlo = tri.min(1).values; thi = tri.max(1).values; del tri
    lo_t = int(((vc[:, ax].min() + .5) * n_tok).floor()); hi_t = int(((vc[:, ax].max() + .5) * n_tok).floor()) + 1
    cuts = np.linspace(lo_t, hi_t, nslab + 1).round().astype(int); C = []; Fe = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        x0 = (a - margin) / n_tok - .5; x1 = (b + margin) / n_tok - .5
        sel = (tlo >= x0) & (thi <= x1)
        if not sel.any(): continue
        fs = fc[sel]; uv_, inv = __import__('torch').unique(fs, return_inverse=True)
        c, fe = encode_direct(vc[uv_].cpu(), inv.reshape(-1, 3).cpu(), res)
        keep = (c[:, 1 + ax] >= a) & (c[:, 1 + ax] < b); C.append(c[keep]); Fe.append(fe[keep])
        __import__('torch').cuda.empty_cache()
    return np.concatenate(C), np.concatenate(Fe)

def torch_argmax_extent(v):
    return (v.max(0).values - v.min(0).values).argmax()
