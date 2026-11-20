"""Small modules only; no geometry baseline is added to the HR latent."""
import torch
from torch import nn
from torch.nn import functional as F


class ImageLRFusion(nn.Module):
    def __init__(self, channels, image_channels, hidden_channels, width=128):
        super().__init__()
        self.lr = nn.Linear(channels, width)
        self.image_kv = nn.Linear(image_channels, 2*width)
        self.mix = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, width))
        self.to_hidden = nn.Linear(width, hidden_channels)
        self.width = width

    def forward(self, features, image):
        # Soft associations, not a calibrated 3D-to-image projection.
        q = self.lr(features)
        k, v = self.image_kv(image).chunk(2, -1)
        evidence = F.scaled_dot_product_attention(q[None,None], k[None,None], v[None,None])[0,0]
        context = self.mix(torch.cat((q, evidence), -1))
        return self.to_hidden(context), context


class GatedResidual(nn.Module):
    def __init__(self, channels, width=128):
        super().__init__()
        self.down = nn.Linear(channels, width, bias=False)
        self.up = nn.Linear(width, channels, bias=False)
        self.gate = nn.Linear(width+2, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, lr_attention, evidence, coverage, time):
        gate = torch.sigmoid(self.gate(torch.cat((evidence, coverage, time), -1)))
        return gate*self.up(F.silu(self.down(lr_attention)))


def perturb_local(coords, features, batch_size, *, generator, probability=.5,
                  radius=2, delete_fraction=.5, noise_std=.1):
    """One region per selected sample; coordinates never move; no GT HR is read."""
    if not 0 <= probability <= 1 or not 0 <= delete_fraction < 1 or radius < 0 or noise_std < 0:
        raise ValueError('Invalid perturbation parameters')
    keep = torch.ones(len(coords), dtype=torch.bool, device=coords.device)
    out = features.clone()
    affected = 0
    for b in range(batch_size):
        ids = torch.where(coords[:,0] == b)[0]
        if not len(ids):
            raise ValueError('Each LR sample must have at least one token')
        if torch.rand((), generator=generator, device=coords.device) >= probability:
            continue
        center = ids[torch.randint(len(ids), (), generator=generator, device=coords.device)]
        region = ids[(coords[ids,1:]-coords[center,1:]).abs().amax(1) <= radius]
        affected += 1
        if torch.rand((), generator=generator, device=coords.device) < .5:
            selected = region[torch.rand(len(region), generator=generator, device=coords.device) < delete_fraction]
            keep[selected] = False
            # A fully empty sample is not part of this first experiment.
            if not keep[ids].any():
                keep[ids[0]] = True
        else:
            noise = torch.randn(out[region].shape, generator=generator, device=out.device, dtype=torch.float32)
            out[region] += (noise_std*noise).to(out.dtype)
    return coords[keep].clone(), out[keep], {'perturbed_samples': affected, 'removed_tokens': int((~keep).sum())}
