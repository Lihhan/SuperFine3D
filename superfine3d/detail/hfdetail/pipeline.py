"""Image-detail injection for any mesh (one zoom crop or the whole stitched object), after the zoom refinement.

    v_new = refine(v, F, frame, cam, image=..., model=...)            # estimate normals from the image
    v_new = refine(v, F, frame, cam, target=(E, Em))                    # or pass a precomputed estimate

frame: local mesh frame (heightfield.Frame). One crop: Frame(center, zoom, 1/1024) (crop-frame vertices, as stored in
crops/<arm>/<cid>.npz). Whole object at 4096: Frame((0,0,0), 1., 1/4096) with object-frame vertices; this is the
same physical setting as zoom-4 crops (one voxel = 1/4096, visibility tolerance 2/1024, max displacement 3 voxels),
and one height field over the whole input image avoids crop seams.
cam: dict with P (3x4, object -> normalized image coords), r, u, fwd (object-frame camera axes).
condition: mesh-conditioned estimator; the input mesh's normals rendered at image resolution are its second input.
mode: 'resid' (default; mesh's own coarse normals below `k` px + gain x estimate detail above) or 'raw' (estimate).

CLI: python -m hfdetail.pipeline --mesh in.npz|in.ply --image view.png --cam cam.json --ckpt ckpt.pt --out out.npz
     [--zoom 1 --center 0,0,0 --unit 1/4096 --res 1024 --mode resid --k 32 --gain 0.9 --iters 800 --reference]"""
import math
import argparse, json, time, numpy as np, torch
from . import visibility, targets, heightfield as hf

MULTIVIEW = None   # set by multi-view drivers: dict(cams=[(P, fwd), ...], index=i, smooth=iters)


def view_partition(X0, F, vis_i, res, tol_obj, cams, index, smooth=6):
    """Soft one-hot ownership of each vertex by the view that sees it most frontally: score_j = visible_j * max(0, n.-fwd_j);
    owner = argmax_j; the indicator of view `index` is softened by `smooth` 1-ring averages (no seams)."""
    import torch
    T = X0[F]; fn = torch.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0], dim=1)
    n = torch.zeros_like(X0).index_add_(0, F.reshape(-1), fn.repeat_interleave(3, 0)); n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sc = []
    for j, (Pj, fj) in enumerate(cams):
        vj = vis_i if j == index else visibility.vertex_visibility(X0, F, Pj, fj, res, tol_obj=tol_obj)[0]
        fw = torch.as_tensor(fj, device=X0.device, dtype=X0.dtype)
        sc.append((vj > .5).float() * (n @ (-fw)).abs())     # |cos|: vertex normals may face either way on thin parts
    sc = torch.stack(sc, 1); own = (sc.argmax(1) == index).float()
    return visibility.soften(own, F, smooth)


def refine(v, F, frame, cam, image=None, model=None, target=None, res=None, mode='resid', k=32, gain=.9,
           params=hf.HFParams(), tol_obj=2 / 1024, fast=True, est_kw=None, stats=None, condition=False,
           outlier_deg=None, outlier_rings=1, outlier_iters=10, fine_k=None, fine_gain=None, blend_r=None, blend_amax=2., align=False, align_kw=None, edge_tau=None, depth=None, depth_w=1.0, agree_deg=None, graze=None, conf=None):
    """v (V,3) local frame, F (M,3) long (cuda). Returns displaced local-frame vertices (V,3)."""
    P, fwd = cam['P'], cam['fwd']; t0 = time.time()
    if target is None:
        from . import estimator
        res = res or image.shape[0]; kw = dict(est_kw or {})
        if condition:                                                 # current mesh's normals at image resolution
            from . import raster, camera as cm_
            X = frame.to_object(v); W0 = image.shape[0]
            tri = raster.tri_ids(raster.rasterize(cm_.clip_coords(X, P), F, W0)); Nm, mm = raster.normal_map(X, F, tri, fwd)
            kw['cond'] = estimator.cond_map(Nm, mm, cam); del Nm, mm, tri
        E, Em = estimator.estimate(image, cam, res, model, **kw)
    else:
        E, Em = target; res = E.shape[0]
    t1 = time.time(); X0 = frame.to_object(v)
    vis, rr = visibility.vertex_visibility(X0, F, P, fwd, res, tol_obj=tol_obj)
    if MESH_ALIGN is not None and mode in ('resid', 'blend'):         # opt-in: move the mesh onto the image (the reverse of `align`)
        v = mesh_align(v, F, frame, cam, E, Em, X0, rr, fwd, stats, **MESH_ALIGN); X0 = frame.to_object(v)
        vis, rr = visibility.vertex_visibility(X0, F, P, fwd, res, tol_obj=tol_obj)
    if MULTIVIEW is not None:                                         # opt-in (multi-view injection): this view moves only the
        vis = vis * view_partition(X0, F, vis, res, tol_obj, **MULTIVIEW)   # vertices it sees most frontally (no double injection)
    if mode in ('resid', 'blend'):
        M0, h0 = targets.mesh_normal_map(X0, F, rr, fwd)
        if align:                                                     # warp the image estimate onto the mesh's own structures
            from . import align as al_
            B = None if 'r' not in cam else [cam['r'], cam['u'], [-x for x in fwd]]
            E, Em = al_.flow_align(E, Em, M0, h0, stats=stats, basis=B, **(align_kw or {}))
    if mode == 'resid':
        T, Tm = targets.residual(E, Em, M0, h0, k, gain, fine_k, fine_gain)
    elif mode == 'blend':
        T, Tm = targets.blend(E, Em, M0, h0, k, gain, blend_r, blend_amax, fine_k=fine_k, fine_gain=fine_gain)
    elif mode == 'raw': T, Tm = targets.raw(E, Em)
    else: raise ValueError(mode)
    if AUTO_MIN_PASS is not None and mode == 'resid' and targets.SHRINK_STATS.get('pass_frac') is not None \
            and targets.SHRINK_STATS['pass_frac'] < AUTO_MIN_PASS:      # opt-in evidence-adaptive level: the image shows (almost) no detail
        if stats is not None: stats.update(auto_skipped=1, estimate_s=t1 - t0, fit_s=0., res=res)   # above the gate's noise floor here ->
        return v.clone()                                              # no height-field fit at this level (mesh unchanged)
    if graze is not None and mode in ('resid', 'blend'):             # opt-in: lower trust in the mesh's coarse shape where it is
        import torch.nn.functional as F_                              # grazing (steep walls / troughs, n.v < thr): blend towards the
        thr, wmin, r = graze                                          # pure image target there (weight wmin on the mesh)
        V = -torch.as_tensor(fwd, device=M0.device, dtype=M0.dtype); nv = (M0 @ V).abs()
        g = ((nv < thr) & h0).float()[None, None]; g = F_.max_pool2d(g, 2 * r + 1, 1, r); g = F_.avg_pool2d(g, 2 * r + 1, 1, r)[0, 0]
        w = (1. - (1. - wmin) * g)[..., None]
        Ti, _ = targets.residual(E, Em, E, Em, k, gain, fine_k, fine_gain)
        T = w * T + (1. - w) * Ti; T = T / T.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if stats is not None: stats.update(graze_frac=float((g > .5).float()[h0].mean()))
    if conf is not None and mode in ('resid', 'blend'):               # opt-in confidence blend: where the mesh's mid-scale shape
        import torch.nn.functional as F_                              # (k..kb px) clearly disagrees with the image's, move the coarse
        kb, a0, a1, wmax, r = conf; kb = int(kb)                      # part of the target toward the image's mid band (soft, capped);
        lMb = targets.lowpass(M0, h0, kb); lMk = targets.lowpass(M0, h0, k)   # where they agree the usual residual target is kept
        lEk = targets.lowpass(E, Em, k); lEb = targets.lowpass(E, Em, kb)
        Ci = lMb + (lEk - lEb)                                        # image mid band on the mesh's very coarse (> kb px) shape
        cs = (Ci * lMk).sum(-1) / (Ci.norm(dim=-1) * lMk.norm(dim=-1)).clamp_min(1e-6)
        ang = torch.rad2deg(torch.acos(cs.clamp(-1, 1)))
        w = ((ang - a0) / max(a1 - a0, 1e-3)).clamp(0, 1); w = w * w * (3 - 2 * w) * wmax * (Em & h0).float()
        w = F_.avg_pool2d(w[None, None], 2 * int(r) + 1, 1, int(r), count_include_pad=False)[0, 0][..., None]
        Ti = Ci + gain * (E - lEk); Ti = Ti / Ti.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        T = (1. - w) * T + w * Ti; T = T / T.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if stats is not None:
            mm = (Tm & h0); stats.update(conf_w_mean=float(w[..., 0][mm].mean()) if mm.any() else 0., conf_frac=float((w[..., 0] > .5)[mm].float().mean()) if mm.any() else 0.)
    if agree_deg is not None and mode in ('resid', 'blend'):          # opt-in: inject only where the estimate's coarse layout
        lE = targets.lowpass(E, Em, k); lM = targets.lowpass(M0, h0, k)  # (k-px lowpass) agrees with the mesh's (angle < agree_deg)
        cs = (lE * lM).sum(-1) / (lE.norm(dim=-1) * lM.norm(dim=-1)).clamp_min(1e-6); ok = cs > math.cos(math.radians(agree_deg))
        if stats is not None: stats.update(agree_frac=float((ok & Tm).sum() / Tm.sum().clamp_min(1)))
        Tm = Tm & ok
    if not ((rr[..., 3] > 0) & Tm).any():
        out = v.clone()                                               # mesh_hf.py: 'no overlap' -> unchanged
    else:
        if depth is not None:                                         # mid-scale steps from a monocular depth map
            from dataclasses import replace
            params = replace(params, depth_T=depth_target(depth, X0, F, rr, fwd, frame, res // params.ds, params.depth_f), depth_w=depth_w)
            if stats is not None: stats.update(depth_scale=float(_LAST_SCALE[0]))
        if edge_tau is not None and image is not None:                 # edge-aware smoothing: weak on image edges
            from dataclasses import replace
            params = replace(params, reg_w=edge_weights(image, res // params.ds, edge_tau, v.device))
        out = (hf.fit_fast if fast else hf.fit_reference)(v, F, frame, P, fwd, vis, T, Tm, res, params)
        if EVMASK is not None and mode == 'resid' and targets.SHRINK_S[0] is not None:   # opt-in: move the surface only where the image
            import torch.nn.functional as F_                          # shows detail evidence: displacement x the gate s (box-blurred over
            r_ = int(EVMASK); g_ = targets.SHRINK_S[0][None, None].float()   # 2r+1 px so the transition has no step), sampled at each vertex
            if r_ > 0: g_ = F_.avg_pool2d(g_, 2 * r_ + 1, 1, r_, count_include_pad=False)
            from . import camera as _cm; uvv, _ = _cm.project(frame.to_object(v), P); wv = F_.grid_sample(g_, (uvv * 2 - 1)[None, :, None, :], mode='bilinear', align_corners=False, padding_mode='border').view(-1)
            out = v + wv[:, None] * (out - v)
            if stats is not None: stats.update(evmask_mean=float(wv[vis > .5].mean()) if (vis > .5).any() else 0.)
        if outlier_deg is not None:                                   # smooth away faces that contradict the target
            from . import cleanup
            bad = cleanup.outlier_vertices(frame.to_object(out), F, P, fwd, T, Tm, res, outlier_deg)
            w = cleanup.grow(bad, F, outlier_rings) * (vis > .5).float(); out = cleanup.smooth_region(out, F, w, outlier_iters)
            if stats is not None: stats.update(outlier_vertices=float(bad.float().mean()))
    if stats is not None: stats.update(estimate_s=t1 - t0, fit_s=time.time() - t1, visible=float((vis > .5).float().mean()), res=res)
    return out

_LAST_SCALE = [0.]
MESH_ALIGN = None   # opt-in (inject_lib cfg MALIGN=sigma:max_px[:ds]): image-plane deformation of the mesh onto the image before the fit
def mesh_align(v, F, frame, cam, E, Em, X0, rr, fwd, stats=None, sigma=6., max_shift=96., ds=4):
    """Moves the mesh so that its rendered normals line up with the image's normal estimate: dense flow f with M0(x) ~ E(x + f(x))
    (align.flow: DIS on low-passed camera-frame normal images, Gaussian-smoothed, bounded, zero outside the common mask), then every
    vertex - visible or not - is shifted in the image plane by f at its projection, at its own depth (P = [M | t]: dX = M^-1 (du w,
    dv w, 0)). Whole columns move together, so hidden parts follow the visible surface; the shape along the view axis is unchanged."""
    import torch.nn.functional as F_
    from . import align as al_, targets as tg_, camera as cm_
    M0, h0 = tg_.mesh_normal_map(X0, F, rr, fwd)
    B = None if 'r' not in cam else torch.as_tensor([cam['r'], cam['u'], [-x for x in fwd]], device=M0.device, dtype=M0.dtype)
    to = (lambda N: N @ B.T) if B is not None else (lambda N: N)
    f = al_.flow(to(M0), h0, to(E), Em, ds=int(ds), sigma=float(sigma), max_shift=float(max_shift))
    H, W = h0.shape; ft = torch.as_tensor(f, device=v.device, dtype=torch.float32).permute(2, 0, 1)[None]   # (1,2,H,W) pixels
    uv, w = cm_.project(X0, cam['P']); fv = F_.grid_sample(ft, (uv * 2 - 1)[None, :, None, :].float(), mode='bilinear', align_corners=False, padding_mode='zeros')[0, :, :, 0].T
    Pm = torch.as_tensor(cam['P'], device=v.device, dtype=torch.float64); Mi = torch.linalg.inv(Pm[:, :3])
    d = torch.stack([fv[:, 0].double() / W * w.double(), fv[:, 1].double() / H * w.double(), torch.zeros_like(w).double()], 1)
    dX = (d @ Mi.T).float(); vn = v + dX * frame.zoom
    if stats is not None:
        n = np.linalg.norm(f, axis=-1)[h0.detach().cpu().numpy()]
        stats.update(malign_median_px=float(np.median(n)) if n.size else 0., malign_p90_px=float(np.quantile(n, .9)) if n.size else 0.,
                     malign_moved_obj=float(dX.norm(dim=1).mean()))
    return vn
EVMASK = None   # opt-in (inject_lib cfg EVMASK=r px): scale the fitted displacement by the blurred evidence gate (needs SHRINK)
AUTO_MIN_PASS = None   # opt-in (inject_lib cfg AUTO=frac): skip the fit when the evidence gate passes < frac of the target pixels
def depth_target(depth, X0, F, rast, fwd, frame, G, f=32):
    """Height-field depth target (voxels, (g,g), g = G//4): mid band of the aligned monocular depth minus the mid band of the
    current mesh depth. depth: (H,W) array/tensor on the image grid (NaN = invalid). Alignment: one scale s fitted on the
    coarse band (lowpass_f - lowpass_8f) where the generated shape is reliable; offset irrelevant (band-pass)."""
    import torch.nn.functional as F_
    dev = X0.device; g = max(G // 4, 8)
    Dm = (depth.float().to(dev) if torch.is_tensor(depth) else torch.as_tensor(np.asarray(depth), dtype=torch.float32, device=dev))
    fw = torch.as_tensor(fwd, device=dev, dtype=X0.dtype); z = X0 @ fw                                  # object-unit depth per vertex
    tri = rast[..., 3].long() - 1; m = tri >= 0; t = F[tri.clamp_min(0)]; b0 = rast[..., 0].float(); b1 = rast[..., 1].float()
    D0 = torch.where(m, b0 * z[t[..., 0]] + b1 * z[t[..., 1]] + (1 - b0 - b1) * z[t[..., 2]], torch.tensor(float('nan'), device=dev))
    def down(D):
        ok = torch.isfinite(D).float(); a = torch.nan_to_num(D)[None, None]; o = ok[None, None]
        a = F_.interpolate(a, size=(g, g), mode='area'); w = F_.interpolate(o, size=(g, g), mode='area')
        return torch.where(w > .99, a / w.clamp_min(1e-6), torch.tensor(float('nan'), device=dev))
    Dm = down(Dm); D0 = down(D0); ok = torch.isfinite(Dm) & torch.isfinite(D0); okf = ok.float()
    Dm = Dm - Dm[ok].mean(); D0 = D0 - D0[ok].mean()                  # remove offsets (absolute depths differ by a constant)
    def lp(D, k):                                                     # normalised-convolution lowpass, invalid cells ignored
        up = lambda x: F_.interpolate(F_.avg_pool2d(x, k, k, ceil_mode=True), size=D.shape[-2:], mode='bilinear', align_corners=False)
        return up(torch.nan_to_num(D) * okf) / up(okf).clamp_min(1e-6)
    band = lambda D: torch.nan_to_num(D) - lp(D, f)
    coarse = lambda D: lp(D, f) - lp(D, 8 * f)
    cm, c0 = coarse(Dm)[ok], coarse(D0)[ok]; s = float((cm * c0).sum() / (cm * cm).sum().clamp_min(1e-12)); s = max(s, 0.); _LAST_SCALE[0] = s
    T = (s * band(Dm) - band(D0)) / (frame.unit / frame.zoom)
    return torch.where(ok, T, torch.tensor(float('nan'), device=dev))[0, 0]

def edge_weights(image, G, tau, device='cuda'):
    """(G,G) smoothing weights w = 1 / (1 + (g / (tau * q90))^2), g = gradient magnitude of the (lightly blurred) grey
    input image resampled to the height-field grid, q90 its 90th percentile on the foreground."""
    import torch.nn.functional as F_
    a = torch.as_tensor(np.asarray(image), dtype=torch.float32, device=device).mean(-1)[None, None] / 255.
    a = F_.interpolate(a, size=(G, G), mode='area'); a = F_.avg_pool2d(F_.pad(a, (1, 1, 1, 1), mode='replicate'), 3, 1)
    gx = a[..., :, 2:] - a[..., :, :-2]; gy = a[..., 2:, :] - a[..., :-2, :]
    g = torch.zeros_like(a); g[..., 1:-1, 1:-1] = (gx[..., 1:-1, :] ** 2 + gy[..., :, 1:-1] ** 2).sqrt()
    fg = a > .03; q = torch.quantile(g[fg][:: max(1, int(fg.sum()) // 1000000)], .9) if fg.any() else g.max()
    return (1. / (1. + (g / (tau * q.clamp_min(1e-6))) ** 2))[0, 0]

def _load_mesh(p):
    if p.endswith('.npz'):
        with np.load(p) as a: return torch.from_numpy(a['v']).float().cuda(), torch.from_numpy(a['f']).long().cuda()
    import trimesh; m = trimesh.load(p, process=False); return torch.from_numpy(np.asarray(m.vertices)).float().cuda(), torch.from_numpy(np.asarray(m.faces)).long().cuda()

def main(argv=None):
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mesh', required=True); A('--image', required=True); A('--cam', required=True); A('--ckpt', default=None); A('--out', required=True)
    A('--zoom', type=float, default=1.); A('--center', default='0,0,0'); A('--unit', default='1/4096'); A('--res', type=int, default=None)
    A('--mode', default='resid'); A('--k', type=int, default=32); A('--gain', type=float, default=.9); A('--iters', type=int, default=800)
    A('--tol', type=float, default=2 / 1024); A('--batch', type=int, default=8); A('--reference', action='store_true'); A('--condition', action='store_true')
    a = ap.parse_args(argv)
    from PIL import Image; from . import estimator
    rgba = np.array(Image.open(a.image).convert('RGBA')).astype(np.float32) / 255
    img = (rgba[..., :3] * rgba[..., 3:4] * 255).clip(0, 255).astype(np.uint8)          # premultiplied on black
    cam = json.load(open(a.cam)); v, F = _load_mesh(a.mesh)
    num, den = (a.unit.split('/') + ['1'])[:2]; unit = float(num) / float(den)
    frame = hf.Frame(tuple(float(x) for x in a.center.split(',')), a.zoom, unit); st = {}
    out = refine(v, F, frame, cam, image=img, model=estimator.NiRNE(a.ckpt), res=a.res, mode=a.mode, k=a.k, gain=a.gain,
                 params=hf.HFParams(iters=a.iters), tol_obj=a.tol, fast=not a.reference, est_kw=dict(batch=a.batch), stats=st, condition=a.condition)
    np.savez(a.out, v=out.cpu().numpy(), f=F.cpu().numpy()); print(json.dumps(st))

if __name__ == '__main__': main()
