"""Per-vertex visibility from the input camera (exact port of mesh_hf.py).
A vertex is visible if it projects inside the image, its pixel is covered, and its NDC depth matches the z-buffer
within the NDC depth change of moving the vertex by `tol_obj` (object units) along the view direction.
mesh_hf.py moved crop-frame vertices by fwd*zoom*2/1024, i.e. tol_obj = 2/1024 in the object frame.
The hard visibility is then softened by `smooth_iters` rounds of 1-ring averaging (vis = .5 vis + .5 mean(neighbours))."""
import torch
from . import camera, raster

def vertex_visibility(X0, F, P, fwd, res, tol_obj=2 / 1024, smooth_iters=3, rast=None):
    """X0 (V,3) object frame, F (M,3) long. Returns soft visibility (V,) float32 and the raster at `res` (reused later)."""
    with torch.no_grad():
        uv0, _ = camera.project(X0, P); clip0 = camera.clip_coords(X0, P)
        rr = raster.rasterize(clip0, F, res) if rast is None else rast
        zb = rr[..., 2].float()                                       # float32 compare, as nvdiffrast returns it
        q = (uv0.clamp(0, 1 - 1e-6) * res).long(); ndc = camera.ndc_depth(clip0)
        hitpix = rr[q[:, 1], q[:, 0], 3] > 0
        dz = (ndc - zb[q[:, 1], q[:, 0]]).abs()
        fw = torch.as_tensor(fwd, device=X0.device, dtype=X0.dtype)
        tol = (camera.ndc_depth(camera.clip_coords(X0 + fw * tol_obj, P)) - ndc).abs()
        vis = (hitpix & (dz <= tol) & (uv0 >= 0).all(1) & (uv0 < 1).all(1)).float()
        vis = soften(vis, F, smooth_iters)
    return vis, rr

_DIRECTED = ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2))   # the 6 directed edges of a triangle, as mesh_hf.py's e
def soften(vis, F, iters=3, chunk=32_000_000):
    """1-ring averaging with edge multiplicity, as mesh_hf.py. mesh_hf.py materialises all 6M directed edges at once
    (~14 GB for a 148M-face 4096 mesh); here the same addends are accumulated per directed-edge column pair and face
    chunk. Only the float summation order differs, which CUDA atomics leave unspecified anyway."""
    dev = vis.device; deg = torch.zeros(len(vis), device=dev)
    for a, b in _DIRECTED:
        for c0 in range(0, len(F), chunk): deg.index_add_(0, F[c0:c0 + chunk, a], torch.ones(len(F[c0:c0 + chunk]), device=dev))
    deg = deg.clamp_min(1)
    for _ in range(iters):
        acc = torch.zeros_like(vis)
        for a, b in _DIRECTED:
            for c0 in range(0, len(F), chunk): acc.index_add_(0, F[c0:c0 + chunk, a], vis[F[c0:c0 + chunk, b]])
        vis = 0.5 * vis + 0.5 * acc / deg
    return vis
