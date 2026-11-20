"""Exact speed-up of the boundary-attention routing plan (tf6_2/boundary_attention.py).

_spread sorts each candidate bucket by key (morton(xyz), tuple(row)); the original evaluates the pure-Python _morton once
per (bucket, member), i.e. millions of times per plan. Here every token gets its key once (vectorised Morton code, then
lexicographic rank over (morton, batch, x, y, z)); sorting a bucket by that rank gives exactly the original order
(coordinates are unique, so the keys are a strict total order).
install() patches boundary_attention._spread in place; the plan itself is unchanged (tests: test_fastroute.py)."""
import numpy as np

_CACHE = {}


def morton_np(xyz, edge):
    """Vectorised tf6.near_attention._morton for an (N,3) int array."""
    xyz = np.asarray(xyz, np.int64) % edge
    value = np.zeros(len(xyz), np.int64)
    for bit in range(max(1, (edge - 1).bit_length())):
        for axis in range(3):
            value |= ((xyz[:, axis] >> bit) & 1) << (3 * bit + axis)
    return value


def _rank(rows, edge):
    key = (id(rows), len(rows), edge)
    if key not in _CACHE:
        _CACHE.clear()                                   # one plan at a time; rows is the plan's own list
        a = np.asarray(rows, np.int64)
        m = morton_np(a[:, 1:], edge)
        order = np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0], m))
        r = np.empty(len(a), np.int64); r[order] = np.arange(len(a))
        _CACHE[key] = (rows, r.tolist())                 # keep rows alive so id() cannot be reused
    return _CACHE[key][1]


def spread_fast(groups, limit, rows, edge):
    """Same contract and result as boundary_attention._spread."""
    rank = _rank(rows, edge)
    sources = [sorted(groups[k], key=rank.__getitem__) for k in sorted(groups) if groups[k]]
    allocation = [0] * len(sources)
    left = min(limit, sum(map(len, sources)))
    while left:
        for i, source in enumerate(sources):
            if left and allocation[i] < len(source):
                allocation[i] += 1
                left -= 1
    return [source[((2 * k + 1) * len(source)) // (2 * n)]
            for source, n in zip(sources, allocation) for k in range(n)]


def install():
    from tf6_2 import boundary_attention as ba
    if getattr(ba, '_FASTROUTE', False):
        return
    ba._spread_orig = ba._spread
    ba._spread = spread_fast
    ba._FASTROUTE = True
