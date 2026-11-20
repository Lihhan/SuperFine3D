"""Feature-preserving bilateral normal filtering (opt-in callers only; Zheng et al. style). Face normals of edge-adjacent faces
are averaged with an area weight and a range weight on the normal difference (sigma_r): variation below ~sigma_r (decoder
jitter, facet noise) is averaged away, creases above it are kept. Vertices follow the filtered normals with the standard update
x_i += 1/|F_i| sum_f n_f (n_f . (c_f - x_i)). Optional per-vertex weights w in [0,1] scale every vertex step (w = 0: fixed).
bilateral(v, F, iters, sigma_r_deg, v_iters, w=None) -> new vertices (V,3); faces unchanged. float32 CUDA tensors.
Memory-lean: face pairs / faces are processed in chunks of `chunk` (same arithmetic as the unchunked version
denoise_before_chunk_20261115.py; only the order of the atomic index_add accumulations differs, which is unordered on CUDA anyway),
so the extra memory is ~11 GiB on a 218M-face 4096 decode (unchunked: +69.5 GiB)."""
import math, torch

def face_adjacency(F, nbucket=8):
    """Edge-adjacent face pairs (a[i], b[i]), each adjacent pair once (callers apply both directions), int32. Same pairs as the
    one-shot sort of denoise_before_chunk_20261115.py (stable sort of the edge keys; both copies of an edge share its smaller
    vertex id, so sorting bucket by bucket of that id gives the same consecutive pairs) with only ~3F/nbucket keys sorted at once."""
    nF = len(F); M = int(F.max()) + 1; ix = [(0, 1), (1, 2), (2, 0)]
    lo = torch.cat([torch.minimum(F[:, p], F[:, q]).int() for p, q in ix]); hi = torch.cat([torch.maximum(F[:, p], F[:, q]).int() for p, q in ix])
    A, B = [], []
    for r in range(nbucket):
        sel = ((lo % nbucket) == r).nonzero().squeeze(1)                  # edge slots k (face k % nF) of this bucket, ascending
        k, order = (lo[sel].long() * M + hi[sel].long()).sort(stable=True); f = sel[order] % nF; del sel, order
        same = k[1:] == k[:-1]; A.append(f[:-1][same].int()); B.append(f[1:][same].int()); del k, f, same
    del lo, hi
    return torch.cat(A), torch.cat(B)

def _normals(x, F, ch):
    n = torch.empty(len(F), 3, device=x.device); ar = torch.empty(len(F), 1, device=x.device)
    for s in range(0, len(F), ch):
        p = x[F[s:s + ch]]; c = torch.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], dim=-1); r = c.norm(dim=-1, keepdim=True)
        n[s:s + ch] = c / r.clamp_min(1e-20); ar[s:s + ch] = r
    return n, ar

def bilateral(v, F, iters=3, sigma_r_deg=15., v_iters=5, w=None, chunk=1 << 23):
    try: return _bilateral(v, F, iters, sigma_r_deg, v_iters, w, chunk)
    except torch.OutOfMemoryError:                                        # fragmented cache in a long-running caller: release it, retry once
        torch.cuda.empty_cache(); return _bilateral(v, F, iters, sigma_r_deg, v_iters, w, chunk)

def _bilateral(v, F, iters, sigma_r_deg, v_iters, w, chunk):
    sr = 2 * math.sin(math.radians(sigma_r_deg) / 2); a, b = face_adjacency(F); x = v.clone()
    cnt = torch.bincount(F.reshape(-1), minlength=len(v)).float().clamp_min(1)[:, None]
    wv = None if w is None else w.float()[:, None]
    for _ in range(iters):
        n, ar = _normals(x, F, chunk); acc = torch.zeros_like(n)
        for s in range(0, len(a), chunk):                                  # pair (i, j) adds n_j to face i and n_i to face j
            aa = a[s:s + chunk].long(); bb = b[s:s + chunk].long(); na = n[aa]; nb = n[bb]
            g = torch.exp(-((na - nb) ** 2).sum(-1) / (2 * sr * sr))
            acc.index_add_(0, aa, nb * (ar[bb, 0] * g)[:, None]); acc.index_add_(0, bb, na * (ar[aa, 0] * g)[:, None])
        for s in range(0, len(F), chunk):                                  # m = n * area + acc, normalised (in place, by chunks)
            ms = n[s:s + chunk] * ar[s:s + chunk] + acc[s:s + chunk]; acc[s:s + chunk] = ms / ms.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        m = acc; del n, ar
        for _ in range(v_iters):
            step = torch.zeros_like(x)
            for s in range(0, len(F), chunk):
                f = F[s:s + chunk]; q = x[f]; c = q.mean(1, keepdim=True); mm = m[s:s + chunk, None, :]
                step.index_add_(0, f.reshape(-1), ((mm * (c - q)).sum(-1, keepdim=True) * mm).reshape(-1, 3))
            step = step / cnt; x = x + (step if wv is None else wv * step)
    return x
