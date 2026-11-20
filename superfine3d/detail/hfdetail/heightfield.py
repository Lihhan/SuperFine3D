"""Height-field post-optimisation (shape from normals), the core of mesh_hf.py, in two numerically equivalent forms.

Model: a scalar field H on the input-image pixel grid (res/ds cells). Each vertex moves along the viewing direction by
  d = maxd * tanh(H(uv0)/maxd) * vis * unit          (unit = one voxel in the local frame; maxd in voxels)
  X = frame.center + (v0 + d * fwd) / frame.zoom      (object frame; for a crop: center/zoom of the crop, unit 1/1024)
Loss: mean over target-mask pixels covered by the mesh of (1 - n.T) + lap * |Laplacian H|^2 + mag * H^2
(+ optional slope penalty, see HFParams.slope), Adam;
the pixel -> face assignment is re-rasterized every `raster_every` iterations.

fit_reference : line-by-line port of the mesh_hf.py loop (all vertices, all pixels every iteration).
fit_fast      : identical mathematics, restricted to what the loss can see. Between two re-rasterizations only the
                faces hit by target-mask pixels (and their vertices) enter the loss, so it evaluates H, positions and
                face normals for that active set only, and the per-pixel loss only on mask pixels, in the same row-major
                order (identical mean). Full positions are computed only at re-rasterization steps (as the reference
                uses them). Remaining differences come from CUDA atomics in backward passes, whose float summation
                order is not deterministic in either implementation (see tests/test_heightfield.py)."""
from dataclasses import dataclass
import torch, torch.nn.functional as F_
from . import camera, raster

@dataclass
class HFParams:
    ds: int = 1               # height-field cell = ds pixels
    iters: int = 800
    lr: float = 0.1
    lap: float = 0.01
    mag: float = 1e-3
    maxd: float = 3.0          # max displacement, voxels
    raster_every: int = 25
    slope: float = None        # max surface slope added by the field (world units, |dd|/|dx|); None = off (mesh_hf.py)
    slope_w: float = 10.       # weight of the slope penalty  slope_w * mean(relu(|grad H| - smax)^2)
    lap_huber: float = None    # edge-preserving: Huber on the Laplacian (quadratic below lap_huber voxels, linear above); None = L2
    reg_w: object = None       # optional (G,G) tensor in [0,1]: per-cell weight of the Laplacian term (e.g. low on image edges)
    depth_T: object = None     # optional (g,g) tensor: target mid-scale displacement band (voxels) on a 1/4-res grid, NaN = no target
    depth_w: float = 1.0       # weight of the depth term, relative to the normal loss at the first iteration (auto-balanced)
    depth_f: int = 32          # depth band = D - lowpass_f(D) on the 1/4-res grid (f cells)
    levels: int = 1            # coarse-to-fine: solve on grids ds*2^(levels-1) ... ds, iters split evenly, each level starts
                               # from the previous one upsampled; 1 = single level (mesh_hf.py)
    orient_init: bool = False  # faces that start front-facing keep that orientation in the loss (a fold-over is penalised
                               # instead of being mirrored back toward the camera); False = sign-agnostic (mesh_hf.py)
    fold_tau: float = None     # fold repair after the fit: smooth the displacement around faces whose normal turned by more
                               # than acos(fold_tau) from the input mesh (0 = flipped), until none is left; None = off
    fold_iters: int = 64
    lp_w: float = None         # opt-in: weight of a penalty on the coarse part of the field, mean(lowpass_lp_k(H)^2): the fitted
                               # displacement stays detail-only (no panel-scale drift from non-integrable targets); None = off
    lp_k: int = 64             # lowpass scale of that penalty in image pixels (cells = lp_k / ds)

@dataclass
class Frame:
    """Local mesh frame: object X = center + v / zoom; displacement unit (one voxel) in local units."""
    center: tuple = (0., 0., 0.)
    zoom: float = 1.
    unit: float = 1 / 1024
    def to_object(self, v):
        return torch.tensor(self.center, device=v.device).float() + v / self.zoom

_LAP = None
FOLD_STATS = {}
def _lapk(device):
    global _LAP
    if _LAP is None or _LAP.device != device: _LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device).float()[None, None]
    return _LAP

def _setup(v0, frame, P, fwd, res, p, H0=None):
    X0 = frame.to_object(v0); uv0, _ = camera.project(X0, P)
    fwd_c = torch.as_tensor(fwd, device=v0.device).float(); G = res // p.ds
    if H0 is None: Hf = torch.zeros(1, 1, G, G, device=v0.device, requires_grad=True)
    else: Hf = F_.interpolate(H0.detach(), size=(G, G), mode='bilinear', align_corners=False).clone().requires_grad_(True)
    opt = torch.optim.Adam([Hf], lr=p.lr)
    return X0, uv0, fwd_c, Hf, opt, (uv0 * 2 - 1)[None, :, None, :]

def _disp(Hf, gq, vis, p, frame):
    h = F_.grid_sample(Hf, gq, mode='bilinear', align_corners=False, padding_mode='border').view(-1)
    return p.maxd * torch.tanh(h / p.maxd) * vis * frame.unit

def _final(v0, Hf, gq, vis, p, frame, fwd_c):
    with torch.no_grad():
        h = F_.grid_sample(Hf, gq, mode='bilinear', align_corners=False, padding_mode='border').view(-1)
        d = p.maxd * torch.tanh(h / p.maxd) * vis
        return v0 + d[:, None] * frame.unit * fwd_c

def _sign(n, V_, s0):
    """Loss-side normal orientation: toward the camera, except faces that started front-facing (s0) keep their sign."""
    cur = torch.where((n @ V_) < 0, -1., 1.)
    return n * (cur if s0 is None else torch.where(s0, 1., cur))[..., None]

def fold_guard(v0, v, F, fwd, tau, iters=16, rings=0, smooth=1):
    """Repair fold-overs added by the fit: where a face turned by more than acos(tau) from the input mesh (tau=0: flipped),
    Laplacian-smooth the displacement d = (v - v0).fwd over those faces' vertices grown by `rings` (both ends of a wall then
    move together, so it stops inverting); repeat until none is left. Returns (v, fraction of faces still beyond tau,
    fraction of vertices touched)."""
    dev = v0.device; f = torch.as_tensor(fwd, device=dev).float(); d = (v - v0) @ f; V = len(v0)
    n0 = raster.face_normals(v0, F); ok = torch.linalg.cross(v0[F[:, 1]] - v0[F[:, 0]], v0[F[:, 2]] - v0[F[:, 0]], dim=1).norm(dim=1) > 0
    e = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = torch.cat([e, e.flip(1)])
    deg = torch.zeros(V, device=dev).index_add_(0, e[:, 0], torch.ones(len(e), device=dev)).clamp_min(1)
    touched = torch.zeros(V, dtype=torch.bool, device=dev)
    for _ in range(iters):
        c = (raster.face_normals(v0 + d[:, None] * f, F) * n0).sum(1); bad = ok & (c < tau)
        if not bool(bad.any()): break
        m = torch.zeros(V, dtype=torch.bool, device=dev); m[F[bad].reshape(-1)] = True
        for _ in range(rings): m = m | (torch.zeros(V, device=dev).index_add_(0, e[:, 0], m[e[:, 1]].float()) > 0)
        for _ in range(smooth):
            avg = torch.zeros(V, device=dev).index_add_(0, e[:, 0], d[e[:, 1]]) / deg; d = torch.where(m, avg, d)
        touched |= m
    c = (raster.face_normals(v0 + d[:, None] * f, F) * n0).sum(1)
    return v0 + d[:, None] * f, float((ok & (c < tau)).float().mean()), float(touched.float().mean())

def _rho(x, delta):
    if delta is None: return x * x
    a = x.abs(); return torch.where(a < delta, x * x, delta * (2 * a - delta))

def _reg(Hf, p, smax=None):
    lp = _rho(F_.conv2d(Hf, _lapk(Hf.device)), p.lap_huber)
    if p.reg_w is not None:
        w = F_.interpolate(torch.as_tensor(p.reg_w, device=Hf.device).float()[None, None], size=lp.shape[-2:], mode='area'); lp = lp * w
    r = p.lap * lp.mean() + p.mag * (Hf ** 2).mean()
    if p.lp_w:                                                       # opt-in coarse-displacement penalty
        k = max(1, int(round(p.lp_k / p.ds)))
        lo = F_.interpolate(F_.avg_pool2d(Hf, k, k, ceil_mode=True), size=Hf.shape[-2:], mode='bilinear', align_corners=False)
        r = r + p.lp_w * (lo ** 2).mean()
    if smax is not None:                                              # finite differences of H between neighbouring cells
        gx = Hf[..., :, 1:] - Hf[..., :, :-1]; gy = Hf[..., 1:, :] - Hf[..., :-1, :]
        r = r + p.slope_w * ((F_.relu(gx.abs() - smax) ** 2).mean() + (F_.relu(gy.abs() - smax) ** 2).mean())
    return r

def depth_band(D, f):
    """Mid-scale band of a (1,1,g,g) field: D - upsample(avg_pool_f(D))."""
    lo = F_.interpolate(F_.avg_pool2d(D, f, f, ceil_mode=True), size=D.shape[-2:], mode='bilinear', align_corners=False)
    return D - lo

def _depth_term(Hf, p):
    """Mean squared difference between the band of the field's displacement and the depth target (voxels^2)."""
    T = torch.as_tensor(p.depth_T, device=Hf.device).float()[None, None]; ok = torch.isfinite(T)
    D = F_.interpolate(p.maxd * torch.tanh(Hf / p.maxd), size=T.shape[-2:], mode='area' if Hf.shape[-1] > T.shape[-1] else 'bilinear',
                       **({} if Hf.shape[-1] > T.shape[-1] else dict(align_corners=False)))
    return ((depth_band(D, p.depth_f) - torch.nan_to_num(T)) ** 2)[ok].mean()

def cells_per_voxel(X0, vis, frame, P, fwd, res, p):
    """Median image-grid cells spanned by a one-voxel step perpendicular to the view, over visible vertices."""
    f = torch.as_tensor(fwd, device=X0.device).float(); a = torch.tensor([1., 0., 0.], device=X0.device)
    if abs(float(f @ a)) > .9: a = torch.tensor([0., 1., 0.], device=X0.device)
    t = torch.linalg.cross(f, a); t = t / t.norm(); X = X0[vis > .5] if (vis > .5).any() else X0
    du = (camera.project(X + t * frame.unit / frame.zoom, P)[0] - camera.project(X, P)[0]).norm(dim=1) * res / p.ds
    return float(du.median())

def slope_limit(X0, vis, frame, P, fwd, res, p):
    """Per-cell bound on |grad H| (voxels per cell) so that the added slope stays below p.slope: one voxel of H
    across one cell is a world slope of cells_per_voxel."""
    return None if p.slope is None else p.slope / cells_per_voxel(X0, vis, frame, P, fwd, res, p)

def _levels(core, v0, F, frame, P, fwd, vis, T, Tm, res, p, log):
    if p.levels <= 1: return core(v0, F, frame, P, fwd, vis, T, Tm, res, p, log)[0]
    from dataclasses import replace
    H = None; n = p.levels
    for i, l in enumerate(reversed(range(n))):
        q = replace(p, ds=p.ds * 2 ** l, iters=p.iters // n + (p.iters % n if l == 0 else 0), levels=1)
        out, H = core(v0, F, frame, P, fwd, vis, T, Tm, res, q, log, H)
    return out

def fit_reference(v0, F, frame, P, fwd, vis, T, Tm, res, p=HFParams(), log=None):
    """v0 (V,3) local frame, F (M,3) long, vis (V,) soft visibility, T (res,res,3) target normals (object frame),
    Tm (res,res) bool target mask. Returns displaced local-frame vertices (V,3)."""
    out = _levels(_fit_reference, v0, F, frame, P, fwd, vis, T, Tm, res, p, log)
    if p.fold_tau is not None: out, FOLD_STATS['left'], FOLD_STATS['touched'] = fold_guard(v0, out, F, fwd, p.fold_tau, p.fold_iters)
    return out

def _fit_reference(v0, F, frame, P, fwd, vis, T, Tm, res, p, log=None, H0=None):
    X0, uv0, fwd_c, Hf, opt, gq = _setup(v0, frame, P, fwd, res, p, H0); V_ = -fwd_c; smax = slope_limit(X0, vis, frame, P, fwd, res, p); lam_d = None
    s0 = (raster.face_normals(X0, F) @ V_ > 0) if p.orient_init else None
    with torch.enable_grad():
        for it in range(p.iters):
            d = _disp(Hf, gq, vis, p, frame); v = v0 + d[:, None] * fwd_c
            X = frame.to_object(v); clip = camera.clip_coords(X, P)
            if it % p.raster_every == 0:
                with torch.no_grad(): tri = raster.tri_ids(raster.rasterize(clip.detach(), F, res)); msk = (tri >= 0) & Tm
            ti = tri.clamp_min(0); n = _sign(raster.face_normals(X, F)[ti], V_, None if s0 is None else s0[ti])
            L = (1 - (n * T).sum(-1))[msk].mean(); loss = L + _reg(Hf, p, smax)
            if p.depth_T is not None:
                Ld = _depth_term(Hf, p)
                if lam_d is None: lam_d = p.depth_w * float(L.detach()) / max(float(Ld.detach()), 1e-12)
                loss = loss + lam_d * Ld
            opt.zero_grad(); loss.backward(); opt.step()
            if log is not None and it in (0, p.iters - 1): log.append(float(L))
    return _final(v0, Hf, gq, vis, p, frame, fwd_c), Hf

def fit_fast(v0, F, frame, P, fwd, vis, T, Tm, res, p=HFParams(), log=None):
    """Same inputs/outputs as fit_reference; computes only the active set (see module docstring)."""
    out = _levels(_fit_fast, v0, F, frame, P, fwd, vis, T, Tm, res, p, log)
    if p.fold_tau is not None: out, FOLD_STATS['left'], FOLD_STATS['touched'] = fold_guard(v0, out, F, fwd, p.fold_tau, p.fold_iters)
    return out

def _fit_fast(v0, F, frame, P, fwd, vis, T, Tm, res, p, log=None, H0=None):
    X0, uv0, fwd_c, Hf, opt, gq = _setup(v0, frame, P, fwd, res, p, H0); V_ = -fwd_c; smax = slope_limit(X0, vis, frame, P, fwd, res, p); lam_d = None
    s0 = (raster.face_normals(X0, F) @ V_ > 0) if p.orient_init else None
    dev = v0.device; lutF = torch.full((len(F),), -1, dtype=torch.long, device=dev); lutV = torch.full((len(v0),), -1, dtype=torch.long, device=dev)
    with torch.enable_grad():
        for it in range(p.iters):
            if it % p.raster_every == 0:
                with torch.no_grad():
                    d = _disp(Hf, gq, vis, p, frame); X = frame.to_object(v0 + d[:, None] * fwd_c)
                    tri = raster.tri_ids(raster.rasterize(camera.clip_coords(X, P), F, res)); msk = (tri >= 0) & Tm
                    ptri = tri[msk]                                   # mask pixels in row-major order (same as [msk] above)
                    S = torch.unique(ptri); lutF.fill_(-1); lutF[S] = torch.arange(len(S), device=dev); pf = lutF[ptri]
                    FS = F[S]; A = torch.unique(FS); lutV.fill_(-1); lutV[A] = torch.arange(len(A), device=dev); FL = lutV[FS]
                    gqA = gq[:, A]; visA = vis[A]; v0A = v0[A]; TA = T[msk]; s0P = None if s0 is None else s0[ptri]
            d = _disp(Hf, gqA, visA, p, frame); X = frame.to_object(v0A + d[:, None] * fwd_c)
            n = _sign(raster.face_normals(X, FL)[pf], V_, s0P)
            L = (1 - (n * TA).sum(-1)).mean(); loss = L + _reg(Hf, p, smax)
            if p.depth_T is not None:
                Ld = _depth_term(Hf, p)
                if lam_d is None: lam_d = p.depth_w * float(L.detach()) / max(float(Ld.detach()), 1e-12)
                loss = loss + lam_d * Ld
            opt.zero_grad(); loss.backward(); opt.step()
            if log is not None and it in (0, p.iters - 1): log.append(float(L))
    return _final(v0, Hf, gq, vis, p, frame, fwd_c), Hf
