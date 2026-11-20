"""Resolution-explicit sparse interpolation and residual-flow algebra (FP32)."""
import itertools
import torch
from torch import nn


def sparse_read(lr_coords, lr_feats, hr_coords, lr_edge, hr_edge):
    if hr_edge % lr_edge or hr_edge < lr_edge:
        raise ValueError('Expected integer HR/LR scale ratio')
    for c, edge in ((lr_coords, lr_edge), (hr_coords, hr_edge)):
        if c.ndim != 2 or c.shape[1] != 4 or c.dtype not in (torch.int32, torch.int64):
            raise ValueError('Expected integer batch/xyz coordinates')
        if c.numel() and ((c < 0).any() or (c[:, 1:] >= edge).any()):
            raise ValueError('Coordinates outside declared latent grid')
    def key(c):
        c = c.long()
        return ((c[:, 0]*lr_edge+c[:, 1])*lr_edge+c[:, 2])*lr_edge+c[:, 3]
    keys, order = key(lr_coords).sort()
    if len(keys) > 1 and (keys[1:] == keys[:-1]).any():
        raise ValueError('Duplicate LR coordinates')
    pos = (hr_coords[:, 1:].float()+.5)*(lr_edge/hr_edge)-.5
    lower = pos.floor().long()
    fraction = pos-lower.float()
    sums = torch.zeros(len(hr_coords), lr_feats.shape[1], device=lr_feats.device)
    mass = torch.zeros(len(hr_coords), 1, device=lr_feats.device)
    for bits in itertools.product((0, 1), repeat=3):
        bit = torch.tensor(bits, device=lower.device)
        xyz = lower+bit
        valid = ((xyz >= 0) & (xyz < lr_edge)).all(1)
        query = torch.cat((hr_coords[:, :1].long(), xyz.clamp(0, lr_edge-1)), 1)
        qk = key(query)
        idx = torch.searchsorted(keys, qk)
        if not len(keys):
            continue
        safe = idx.clamp(max=len(keys)-1)
        found = valid & (idx < len(keys)) & (keys[safe] == qk)
        w = torch.where(bit.bool(), fraction, 1-fraction).prod(1)*found
        sums = sums + lr_feats[order[safe]].float()*w[:, None]
        mass = mass + w[:, None]
    return sums/mass.clamp_min(1e-8), fraction, mass


class BaselineMap(nn.Module):
    def __init__(self, lr_stats, hr_stats, width=128):
        super().__init__()
        for name, values in (('lm',lr_stats['mean']),('ls',lr_stats['std']),
                             ('hm',hr_stats['mean']),('hs',hr_stats['std'])):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))
        channels = len(hr_stats['mean'])
        if channels != len(lr_stats['mean']):
            raise ValueError('Baseline version requires matching channel counts')
        self.delta = nn.Sequential(nn.Linear(channels+4, width), nn.SiLU(), nn.Linear(width, channels))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, lr_coords, lr_feats, hr_coords, lr_edge, hr_edge):
        f, offset, coverage = sparse_read(lr_coords, lr_feats, hr_coords, lr_edge, hr_edge)
        # LR and HR use different normalization statistics. Convert exactly once.
        raw = (f*self.ls+self.lm-self.hm)/self.hs
        raw = torch.where(coverage > 0, raw, torch.zeros_like(raw))
        delta = self.delta(torch.cat((raw, offset, coverage), -1))
        mapped = torch.where(coverage > 0, raw+delta, torch.zeros_like(raw))
        return mapped, raw, coverage


def residual_path(clean_hr, baseline, noise, t, sigma):
    residual = clean_hr-baseline
    noisy = (1-t)*residual+(sigma+(1-sigma)*t)*noise
    velocity = (1-sigma)*noise-residual
    return noisy, velocity


def reconstruct(noisy_residual, velocity, baseline, t, sigma):
    return (1-sigma)*noisy_residual-(sigma+(1-sigma)*t)*velocity+baseline
