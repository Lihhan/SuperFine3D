"""Camera conventions shared by the whole package (identical to the experiment scripts).
P (3x4) maps object/dump-frame points to homogeneous normalized image coords (u*w, v*w, w), u,v in [0,1];
pixel row i <-> v = (i+.5)/res, column j <-> u = (j+.5)/res.  Clip space for nvdiffrast: (2u w - w, 2v w - w, a w + b, w)."""
import torch

NEAR, FAR = .05, 8.

def homog(X, P):
    """X (N,3) -> h (N,3) = [X,1] @ P^T."""
    P = torch.as_tensor(P, device=X.device, dtype=X.dtype)
    return torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], 1) @ P.T

def project(X, P):
    """X (N,3) -> uv (N,2) normalized image coords, w (N,) depth along the camera axis."""
    h = homog(X, P); w = h[:, 2]
    return h[:, :2] / w[:, None], w

def clip_coords(X, P, near=NEAR, far=FAR):
    """X (N,3) -> clip (1,N,4) for nvdiffrast."""
    h = homog(X, P); w = h[:, 2]; a = (far + near) / (far - near); b = -2 * far * near / (far - near)
    return torch.stack([2 * h[:, 0] - w, 2 * h[:, 1] - w, a * w + b, w], 1)[None].contiguous()

def ndc_depth(clip):
    """clip (1,N,4) -> NDC z (N,) (what nvdiffrast stores in rast[...,2])."""
    return clip[0, :, 2] / clip[0, :, 3]

def make_P(eye, r, u, fwd, f):
    """Pinhole camera centred on the image (hires_render.py convention): u = f (x.r)/(x.fwd) + .5, v = -f (x.u)/(x.fwd) + .5."""
    import numpy as np
    eye, r, u, fwd = (np.asarray(x, dtype=np.float64) for x in (eye, r, u, fwd))
    M = np.stack([f * r + .5 * fwd, -f * u + .5 * fwd, fwd]); return np.c_[M, -M @ eye]
