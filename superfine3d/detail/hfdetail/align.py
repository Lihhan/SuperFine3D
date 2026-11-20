"""Non-rigid alignment of the image-derived normal estimate to the current mesh before building the detail target.
The generated mesh and the input image disagree locally (tens of pixels: the generator draws its own version of a face,
a frieze, ...). Carving the image's fine detail at image positions on top of the mesh's own mid-scale structure gives
doubled contours ("ghosting"). flow_align warps the estimate E onto the mesh's rendered normals M0 with dense optical
flow (DIS) computed on low-passed normal images (no lighting assumption), so the image's detail lands on the mesh's own
structures. The flow is smooth (Gaussian) and bounded (max_shift px); outside the overlap it is zero."""
import numpy as np, torch, cv2

def _enc(N, m, ds, blur):
    """Normals (H,W,3) torch -> two uint8 images (x and y components) at 1/ds resolution, low-passed."""
    a = N.detach().float().cpu().numpy(); mm = m.detach().cpu().numpy().astype(np.float32)
    H, W = mm.shape; h, w = H // ds, W // ds; out = []
    for c in (0, 1):
        x = cv2.resize(a[..., c] * mm, (w, h), interpolation=cv2.INTER_AREA); wgt = cv2.resize(mm, (w, h), interpolation=cv2.INTER_AREA)
        x = x / np.maximum(wgt, 1e-3)
        if blur: x = cv2.GaussianBlur(x, (0, 0), blur)
        out.append(np.clip(x * 127.5 + 127.5, 0, 255).astype(np.uint8))
    return out, cv2.resize(mm, (w, h), interpolation=cv2.INTER_AREA) > .5

def flow(M0, h0, E, Em, ds=4, blur=1.5, sigma=6., max_shift=96., dead=(1., 3.)):
    """Flow f (H,W,2) in full-resolution pixels with M0(x) ~ E(x + f(x)) on the common mask."""
    (mx, my), mk = _enc(M0, h0, ds, blur); (ex, ey), ek = _enc(E, Em, ds, blur); both = mk & ek
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = (dis.calc(mx, ex, None) + dis.calc(my, ey, None)) / 2
    w = both.astype(np.float32); f = f * w[..., None]
    num = np.stack([cv2.GaussianBlur(f[..., i], (0, 0), sigma) for i in (0, 1)], -1); den = cv2.GaussianBlur(w, (0, 0), sigma)[..., None]
    f = num / np.maximum(den, 1e-3) * (den > .05)
    n = np.linalg.norm(f, axis=-1, keepdims=True); f = f * np.minimum(1., (max_shift / ds) / np.maximum(n, 1e-6))
    H, W = h0.shape; f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) * ds
    if dead:                                                          # sub-pixel flow is noise; resampling would only blur
        n = np.linalg.norm(f, axis=-1, keepdims=True); f = f * np.clip((n - dead[0]) / (dead[1] - dead[0]), 0, 1)
    return f

def warp(E, Em, f):
    """E_w(x) = E(x + f(x)) (bicubic, renormalised); mask nearest. Zero flow returns E exactly."""
    if not np.any(f): return E, Em
    H, W = Em.shape; gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx = gx + f[..., 0]; my = gy + f[..., 1]; a = E.detach().float().cpu().numpy()
    w = np.stack([cv2.remap(np.ascontiguousarray(a[..., c]), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) for c in range(3)], -1)
    m = cv2.remap(Em.detach().cpu().numpy().astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    Ew = torch.from_numpy(w).to(E.device); Ew = Ew / Ew.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return Ew * torch.from_numpy(m).to(E.device)[..., None], torch.from_numpy(m).to(Em.device)

def flow_align(E, Em, M0, h0, stats=None, basis=None, **kw):
    """basis: optional (3,3) rows = camera right/up/back axes; the flow is then computed on camera-frame components."""
    to = (lambda N: N @ torch.as_tensor(basis, device=N.device, dtype=N.dtype).T) if basis is not None else (lambda N: N)
    f = flow(to(M0), h0, to(E), Em, **kw); Ew, Emw = warp(E, Em, f)
    if stats is not None:
        n = np.linalg.norm(f, axis=-1)[h0.detach().cpu().numpy()]
        stats.update(align_median_px=float(np.median(n)) if n.size else 0., align_p90_px=float(np.quantile(n, .9)) if n.size else 0.)
    return Ew, Emw
