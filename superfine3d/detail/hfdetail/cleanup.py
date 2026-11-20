"""Local repair of decoder topology defects (thin slits / fins). The decoded meshes carry boundary (hole) and non-manifold
edges along sharp creases; rendered they show as rows of dark dots, and the height field amplifies them. Repair: find
vertices on such edges, grow the region by `rings` 1-rings with a weight that decays per ring, and run umbrella
(Laplacian) smoothing there only. Vertices outside the region never move. Edge keys are built without materialising
the (3M,2) edge array, so 200M-face 4096 meshes fit in GPU memory."""
import torch
from .smooth import _umbrella, _DIRECTED

def edge_counts(F, nV):
    """F (M,3) long -> (M,3) number of faces sharing each face edge (edge j = vertices j, j+1)."""
    keys = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        lo = torch.minimum(F[:, a], F[:, b]); hi = torch.maximum(F[:, a], F[:, b]); keys.append(lo * nV + hi)
    key = torch.cat(keys); del keys
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True); del key
    return cnt[inv].view(3, -1).T.contiguous()

def defect_vertices(F, nV, boundary=True, nonmanifold=True):
    """(nV,) bool: vertices on boundary (count 1) and/or non-manifold (count > 2) edges."""
    ec = edge_counts(F, nV); bad = torch.zeros_like(ec, dtype=torch.bool)
    if boundary: bad |= ec == 1
    if nonmanifold: bad |= ec > 2
    m = torch.zeros(nV, dtype=torch.bool, device=F.device)
    for j, k in ((0, 1), (1, 2), (2, 0)):
        e = bad[:, j]; m[F[e, j]] = True; m[F[e, k]] = True
    return m

def grow(w, F, rings=2, decay=.5):
    """Soft region: defect vertices weight 1, each further ring weight * decay (max over neighbours)."""
    w = w.float().clone()
    for r in range(rings):
        nb = torch.zeros_like(w)
        for a, b in _DIRECTED: nb.scatter_reduce_(0, F[:, a], w[F[:, b]] * decay, reduce='amax')
        w = torch.maximum(w, nb)
    return w

def repair(v, F, rings=2, iters=10, lam=.5, decay=.5, chunk=32_000_000, stats=None):
    """v (V,3), F (M,3) long -> repaired vertices (V,3). Only vertices within `rings` of a defect edge move."""
    d = defect_vertices(F, len(v)); w = grow(d, F, rings, decay); v = smooth_region(v, F, w, iters, lam, chunk)
    if stats is not None: stats.update(defect_vertices=float(d.float().mean()), region=float((w > 0).float().mean()))
    return v

def zipper(v, F, eps, rounds=2, stats=None):
    """Close sub-voxel slits: defect vertices (boundary / non-manifold edges) that fall in the same eps-cell are merged
    to their mean (cells offset by eps/2 in a second round catch pairs split by a cell border). Faces that become
    degenerate are dropped. Returns (v, F) with unreferenced vertices kept (indices of other vertices unchanged)."""
    v = v.clone(); F = F.clone(); n0 = len(F)
    for r in range(rounds):
        d = torch.nonzero(defect_vertices(F, len(v))).view(-1)
        if len(d) == 0: break
        q = torch.floor(v[d] / eps + (.5 if r % 2 else 0.)).long(); q -= q.min(0).values
        key = (q[:, 0] * (q[:, 1].max() + 1) + q[:, 1]) * (q[:, 2].max() + 1) + q[:, 2]
        u, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
        rep = torch.full((len(u),), len(v), dtype=torch.long, device=v.device).scatter_reduce_(0, inv, d, reduce='amin')
        mean = torch.zeros(len(u), 3, device=v.device).index_add_(0, inv, v[d]) / cnt[:, None]
        multi = cnt[inv] > 1; remap = torch.arange(len(v), device=v.device); remap[d[multi]] = rep[inv[multi]]
        v[rep[cnt > 1]] = mean[cnt > 1]; F = remap[F]
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    if stats is not None: stats.update(dropped_faces=n0 - len(F))
    return v, F

def smooth_region(v, F, w, iters=10, lam=.5, chunk=32_000_000):
    """Umbrella (Laplacian) smoothing with per-vertex step weight w (V,) in [0,1]; w = 0 vertices never move."""
    dev = v.device; deg = torch.zeros(len(v), device=dev)
    for a, b in _DIRECTED:
        for c0 in range(0, len(F), chunk): deg.index_add_(0, F[c0:c0 + chunk, a], torch.ones(len(F[c0:c0 + chunk]), device=dev))
    deg = deg.clamp_min(1)
    with torch.no_grad():
        for _ in range(iters): v = v + lam * w[:, None] * _umbrella(v, F, deg, chunk)
    return v

def outlier_vertices(X, F, P, fwd, T, Tm, res, deg=60.):
    """(V,) bool: vertices of faces that are visible at `res` and whose camera-facing normal is more than `deg` degrees
    from the target normal T at that pixel (e.g. grazing slit walls where the image says the surface faces the camera)."""
    from . import raster, camera
    tri = raster.tri_ids(raster.rasterize(camera.clip_coords(X, P), F, res)); N, m = raster.normal_map(X, F, tri, fwd)
    bad = m & Tm & ((N * T).sum(-1) < torch.cos(torch.deg2rad(torch.tensor(float(deg)))).item())
    out = torch.zeros(len(X), dtype=torch.bool, device=X.device); out[F[tri[bad]].view(-1)] = True
    return out

def normal_anomaly_vertices(X, F, P, fwd, res, deg=50., win=9):
    """(V,) bool: vertices of faces whose rendered (camera-facing) normal differs by more than `deg` degrees from the
    win x win mean normal around the pixel; isolated slit walls / fins show up this way at any lighting. Render at a
    resolution where such features cover pixels (e.g. 2x the mesh resolution)."""
    import torch.nn.functional as F_
    from . import raster, camera
    tri = raster.tri_ids(raster.rasterize(camera.clip_coords(X, P), F, res)); N, m = raster.normal_map(X, F, tri, fwd)
    full = F_.avg_pool2d(m[None, None].float(), win, 1, win // 2)[0, 0] > .99
    A = F_.avg_pool2d(N.permute(2, 0, 1)[None].float(), win, 1, win // 2)[0].permute(1, 2, 0); A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    bad = m & full & ((N * A).sum(-1) < torch.cos(torch.deg2rad(torch.tensor(float(deg)))).item())
    out = torch.zeros(len(X), dtype=torch.bool, device=X.device); out[F[tri[bad]].view(-1)] = True
    return out

def boundary_edges(F, nV):
    """(B,2) long: edges used by exactly one face (oriented as in that face)."""
    ec = edge_counts(F, nV); out = []
    for j, k in ((0, 1), (1, 2), (2, 0)):
        e = ec[:, j] == 1; out.append(torch.stack([F[e, j], F[e, k]], 1))
    return torch.cat(out)

def tjunctions(v, F, eps, cell=None):
    """T-junction cracks: for each boundary vertex, the nearest boundary edge not incident to it whose segment lies
    within eps. Returns (vertex ids, edges (n,2), distances). Grid hash on edge midpoints (cell >= half the longest
    boundary edge + eps), 27-cell search."""
    E = boundary_edges(F, len(v)); bv = torch.unique(E.view(-1))
    if len(E) == 0: return bv[:0], E, v.new_zeros(0)
    L = (v[E[:, 0]] - v[E[:, 1]]).norm(dim=1); cell = cell or float(L.max() / 2 + eps)
    mid = (v[E[:, 0]] + v[E[:, 1]]) / 2; q = torch.floor(mid / cell).long(); lo = q.min(0).values - 1; q -= lo; D = q.max(0).values + 2
    key = (q[:, 0] * D[1] + q[:, 1]) * D[2] + q[:, 2]; order = key.argsort(); ks = key[order]
    qv = torch.floor(v[bv] / cell).long() - lo; best = torch.full((len(bv),), float('inf'), device=v.device); arg = torch.full((len(bv),), -1, dtype=torch.long, device=v.device)
    P = v[bv]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                kq = ((qv[:, 0] + dx) * D[1] + qv[:, 1] + dy) * D[2] + qv[:, 2] + dz
                a = torch.searchsorted(ks, kq); b = torch.searchsorted(ks, kq, right=True); n = int((b - a).max()) if len(a) else 0
                for i in range(n):
                    ok = a + i < b; idx = order[(a + i).clamp_max(len(ks) - 1)]; e = E[idx]
                    A = v[e[:, 0]]; B = v[e[:, 1]]; t = (((P - A) * (B - A)).sum(1) / ((B - A) ** 2).sum(1).clamp_min(1e-20)).clamp(0, 1)
                    d = (A + t[:, None] * (B - A) - P).norm(dim=1)
                    ok &= (e[:, 0] != bv) & (e[:, 1] != bv) & (t > 0) & (t < 1) & (d < eps) & (d < best)
                    best = torch.where(ok, d, best); arg = torch.where(ok, idx, arg)
    m = arg >= 0
    return bv[m], E[arg[m]], best[m]

def close_tjunctions(v, F, eps, stats=None):
    """Fill T-junction cracks with one triangle (a, b, vertex) per boundary vertex lying near an opposite boundary edge."""
    vid, e, d = tjunctions(v, F, eps); add = torch.stack([e[:, 0], vid, e[:, 1]], 1)
    if stats is not None: stats.update(tjunctions=len(vid), boundary_vertices=int(torch.unique(boundary_edges(F, len(v)).view(-1)).numel()))
    return torch.cat([F, add])

def boundary_components(v, F, E=None, max_iter=1000):
    """Connected components of the boundary-edge graph. Returns (E (B,2), comp id per edge (B,), number of comps).
    Many decoder holes are bounded partly by non-manifold edges, so components are often open chains, not loops."""
    E = boundary_edges(F, len(v)) if E is None else E; lab = torch.arange(len(v), device=v.device)
    for _ in range(max_iter):
        m = torch.minimum(lab[E[:, 0]], lab[E[:, 1]]); new = lab.clone()
        new.scatter_reduce_(0, E[:, 0], m, reduce='amin'); new.scatter_reduce_(0, E[:, 1], m, reduce='amin'); new = new[new]
        if torch.equal(new, lab): break
        lab = new
    _, comp = torch.unique(lab[E[:, 0]], return_inverse=True)
    return E, comp, int(comp.max()) + 1 if len(comp) else 0

def cap_small_holes(v, F, max_edges=20, max_extent=None, stats=None):
    """Cover each small boundary component (<= max_edges edges, bbox extent <= max_extent) with a fan of triangles to
    a new vertex at the component's mean edge midpoint. Returns (v (V+k,3), F (M+n,3)); original vertices unchanged."""
    E, comp, nc = boundary_components(v, F)
    if nc == 0: return v, F
    cnt = torch.bincount(comp, minlength=nc); P = v[E[:, 0]]
    lo = torch.full((nc, 3), float('inf'), device=v.device).scatter_reduce_(0, comp[:, None].expand(-1, 3), P, reduce='amin')
    hi = torch.full((nc, 3), -float('inf'), device=v.device).scatter_reduce_(0, comp[:, None].expand(-1, 3), P, reduce='amax')
    ok = cnt <= max_edges
    if max_extent is not None: ok &= (hi - lo).max(1).values <= max_extent
    newid = torch.full((nc,), -1, dtype=torch.long, device=v.device); newid[ok] = len(v) + torch.arange(int(ok.sum()), device=v.device)
    mid = (v[E[:, 0]] + v[E[:, 1]]) / 2; c = torch.zeros(nc, 3, device=v.device).index_add_(0, comp, mid) / cnt.clamp_min(1)[:, None]
    sel = ok[comp]; fan = torch.stack([E[sel, 0], E[sel, 1], newid[comp[sel]]], 1)
    if stats is not None: stats.update(capped=int(ok.sum()), components=nc, fan_faces=len(fan))
    return torch.cat([v, c[ok]]), torch.cat([F, fan])


def visible_backfaces(v, F, P, res, tiles=4, view=None):
    """(M,) bool: faces seen from camera P (rasterised at res*tiles over the image, tile by tile) whose raw normal points
    away from the camera. In a single-view reconstruction a visible back face is always a defect (the inside of a
    slit or fold). view: (3,) direction toward the camera (default: from P's third row)."""
    from . import raster, camera
    import numpy as np
    P = np.asarray(P, np.float64); V = torch.as_tensor(view if view is not None else -P[2, :3] / np.linalg.norm(P[2, :3]),
                                                          device=v.device).float()
    seen = torch.zeros(len(F), dtype=torch.bool, device=v.device)
    for i in range(tiles):
        for j in range(tiles):
            Pz = P.copy(); Pz[0] = tiles * (P[0] - (j / tiles) * P[2]); Pz[1] = tiles * (P[1] - (i / tiles) * P[2])
            t = raster.tri_ids(raster.rasterize(camera.clip_coords(v, Pz.tolist()), F, res)); seen[t[t >= 0].unique()] = True
    return seen & ((raster.face_normals(v, F) @ V) < 0)


def slit_faces(v, F, P, res, tiles=3, width=5, ratio=2.0):
    """(M,) bool: faces seen at the bottom of narrow pits of the input-view depth map (winding independent).
    The camera depth is rasterised tile by tile at res*tiles; a grey closing with a width x width window fills pits
    narrower than `width` pixels; a pixel is a slit when the filled depth exceeds its own by more than `ratio` x the
    local pixel footprint (deeper than wide)."""
    from . import raster, camera
    import numpy as np, torch.nn.functional as Fn
    P = np.asarray(P, np.float64); out = torch.zeros(len(F), dtype=torch.bool, device=v.device); k = width // 2
    for i in range(tiles):
        for j in range(tiles):
            Pz = P.copy(); Pz[0] = tiles * (P[0] - (j / tiles) * P[2]); Pz[1] = tiles * (P[1] - (i / tiles) * P[2])
            uv, w = camera.project(v, Pz.tolist()); r = raster.rasterize(camera.clip_coords(v, Pz.tolist()), F, res)
            t = raster.tri_ids(r); m = t >= 0
            if not bool(m.any()): continue
            tf = F[t.clamp_min(0)]; b0, b1 = r[..., 0].float(), r[..., 1].float()
            d = (b0 * w[tf[..., 0]] + b1 * w[tf[..., 1]] + (1 - b0 - b1) * w[tf[..., 2]])
            e = (v[tf[..., 1]] - v[tf[..., 0]]).norm(dim=-1); eu = ((uv[tf[..., 1]] - uv[tf[..., 0]]).norm(dim=-1) * res).clamp_min(1e-3)
            foot = (e / eu)[m].median()                                    # world size of one pixel here
            h = torch.where(m, -d, torch.full_like(d, -1e9))[None, None]   # height toward the camera
            closed = -Fn.max_pool2d(-Fn.max_pool2d(h, width, 1, k), width, 1, k)[0, 0]
            pit = m & ((closed - h[0, 0]) > ratio * foot)
            out[t[pit].unique()] = True
    return out
