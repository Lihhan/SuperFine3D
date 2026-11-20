"""Rasterization and normals (nvdiffrast). Triangle ids are float32 in nvdiffrast (exact below 2^24), so large meshes
are rasterized in face chunks and z-merged, exactly as in oracle_targets.py / hires_render.py."""
import torch
import nvdiffrast.torch as dr
from . import camera

_CTX = {}
def ctx(device='cuda'):
    if device not in _CTX: _CTX[device] = dr.RasterizeCudaContext(device=device)
    return _CTX[device]

def rasterize(clip, F, res, chunk=8_000_000):
    """clip (1,N,4), F (M,3) long -> rast (res,res,4) float64: (bu, bv, ndc z, triangle id + 1 (0 = empty))."""
    out = None
    for a0 in range(0, len(F), chunk):
        r, _ = dr.rasterize(ctx(clip.device.type if clip.device.index is None else f'cuda:{clip.device.index}'),
                            clip, F[a0:a0 + chunk].int().contiguous(), (res, res))
        r = r[0].double(); hit = r[..., 3] > 0
        r[..., 3] = torch.where(hit, r[..., 3] + a0, 0.)
        if out is None: out = r; continue
        better = hit & ((out[..., 3] == 0) | (r[..., 2] < out[..., 2])); out = torch.where(better[..., None], r, out)
    return out

def tri_ids(rast):
    """rast -> (res,res) long face index, -1 where empty."""
    return rast[..., 3].long() - 1

def face_normals(X, F):
    """Unit face normals (M,3) of triangles F over vertices X."""
    T = X[F]; n = torch.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0], dim=1)
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)

def orient(n, view_dir):
    """Flip normals to face the camera: view_dir = -fwd (points toward the camera)."""
    return n * torch.where((n @ view_dir) < 0, -1., 1.)[..., None]

def normal_map(X, F, tri, fwd):
    """Per-pixel face normal (res,res,3) facing the camera, 0 where empty; and hit mask."""
    V = -torch.as_tensor(fwd, device=X.device, dtype=X.dtype); m = tri >= 0
    N = orient(face_normals(X, F)[tri.clamp_min(0)], V)
    return N * m[..., None], m
