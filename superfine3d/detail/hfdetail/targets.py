"""Target normal maps for the height field (exact ports of mesh_hf.py).
raw:       the estimate itself on its mask.
residual:  normalize(lp_k(M0) + gain * (E - lp_k(E))): the mesh's own coarse shape (M0 = its rendered normals) and
           only the estimate's detail above scale k (optionally a lower gain below fine_k); mask = estimate mask & mesh coverage.  gain 0 = smoothing control."""
import torch, torch.nn.functional as F_
from . import raster

def lowpass(N, m, k):
    """Masked box mean over a (k+1)^2 window (not renormalised). N (H,W,3), m (H,W) bool."""
    Nb = F_.avg_pool2d((N * m[..., None]).permute(2, 0, 1)[None], k + 1, 1, k // 2, count_include_pad=False)[0].permute(1, 2, 0)
    return Nb / F_.avg_pool2d(m[None, None].float(), k + 1, 1, k // 2, count_include_pad=False)[0, 0].clamp_min(1e-6)[..., None]

def mesh_normal_map(X0, F, rast, fwd):
    """Rendered (camera-facing) face normals of the input mesh and its coverage, from an existing raster."""
    t0 = raster.tri_ids(rast); h0 = t0 >= 0
    M0 = raster.orient(raster.face_normals(X0, F)[t0.clamp_min(0)], -torch.as_tensor(fwd, device=X0.device, dtype=X0.dtype))
    return M0, h0

def raw(E, Em):
    return E, Em

def residual(E, Em, M0, h0, k, gain, fine_k=None, fine_gain=None):
    """fine_k / fine_gain (optional): the estimate's bands finer than fine_k pixels get gain fine_gain instead of gain
    (the finest bands are the least reliable part of the estimate); both None = mesh_hf.py exactly."""
    with torch.no_grad():
        T = lowpass(M0, h0, k) + gain * (E - lowpass(E, Em, k))
        if fine_k is not None: T = T + (fine_gain - gain) * (E - lowpass(E, Em, fine_k))
        return T / T.norm(dim=-1, keepdim=True).clamp_min(1e-6), Em & h0

def blend(E, Em, M0, h0, k, gain, r=None, amax=2., eps=1e-3, fine_k=None, fine_gain=None):
    """Confidence-weighted residual target. Detail bands above scale k: dE = E - lp_k(E) (estimate), dM = M0 - lp_k(M0)
    (current mesh). Local detail energies eE, eM (rms over an (r+1)^2 window, r = k by default), rho = eE / eM:
      w = clamp(rho, 0, 1)          (trust in the estimate's detail; 1 where it is at least as strong as the mesh's)
      a = clamp(1 / rho, 1, amax)   (amplitude alignment: a weak estimate is scaled towards the mesh's detail energy)
      T = normalize(lp_k(M0) + gain * (w * a * dE + (1 - w) * dM))
    Where the estimate carries detail (rho >= 1) this is residual(); where it is blank (rho -> 0) the mesh keeps its own
    detail instead of being smoothed. fine_k / fine_gain as in residual (applied to dE)."""
    with torch.no_grad():
        lE = lowpass(E, Em, k); lM = lowpass(M0, h0, k); dE = E - lE; dM = M0 - lM
        if fine_k is not None: dE = dE + (fine_gain / gain - 1.) * (E - lowpass(E, Em, fine_k))
        r = r or k; m = Em & h0
        eE = lowpass((dE * dE).sum(-1, keepdim=True), m, r)[..., 0].clamp_min(0).sqrt()
        eM = lowpass((dM * dM).sum(-1, keepdim=True), m, r)[..., 0].clamp_min(0).sqrt()
        rho = eE / (eM + eps); w = rho.clamp(0, 1)[..., None]; a = (1. / rho.clamp_min(1e-6)).clamp(1., amax)[..., None]
        T = lM + gain * (w * a * dE + (1. - w) * dM)
        return T / T.norm(dim=-1, keepdim=True).clamp_min(1e-6), m
