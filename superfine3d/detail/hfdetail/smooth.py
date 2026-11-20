"""Visible-region Taubin smoothing (removes the refinement's sub-pixel speckle before image-driven carving).
Taubin (lambda/mu) alternates a shrinking and an inflating umbrella step, so low frequencies (shape) are kept while
high frequencies (noise) are damped. Each vertex step is scaled by a weight w in [0,1] (e.g. soft visibility): w = 0
vertices never move. Neighbour sums are accumulated per directed-edge column and face chunk (as visibility.soften),
so 100M+ face meshes fit in memory."""
import torch
from .visibility import _DIRECTED

def _umbrella(v, F, deg, chunk):
    acc = torch.zeros_like(v)
    for a, b in _DIRECTED:
        for c0 in range(0, len(F), chunk): acc.index_add_(0, F[c0:c0 + chunk, a], v[F[c0:c0 + chunk, b]])
    return acc / deg[:, None] - v

def taubin(v, F, w=None, iters=10, lam=.5, mu=-.53, chunk=32_000_000):
    """v (V,3), F (M,3) long, w (V,) weights or None (all 1). Returns smoothed vertices (V,3)."""
    dev = v.device; deg = torch.zeros(len(v), device=dev)
    for a, b in _DIRECTED:
        for c0 in range(0, len(F), chunk): deg.index_add_(0, F[c0:c0 + chunk, a], torch.ones(len(F[c0:c0 + chunk]), device=dev))
    deg = deg.clamp_min(1); w = torch.ones(len(v), device=dev) if w is None else w
    with torch.no_grad():
        for _ in range(iters):
            v = v + lam * w[:, None] * _umbrella(v, F, deg, chunk)
            v = v + mu * w[:, None] * _umbrella(v, F, deg, chunk)
    return v
