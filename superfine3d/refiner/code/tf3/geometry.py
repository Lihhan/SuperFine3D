"""Coarse decoded surface-density consistency, NOT latent-channel matching.

The field is a differentiable proxy, not calibrated occupancy or an SDF.
VAE subdivision remains discrete: this loss cannot add missing HR support.
"""
import itertools
import torch


def clean_from_velocity(noisy, velocity, t, sigma_min):
    """Exact inverse for x_t=(1-t)x0+[sigma+(1-sigma)t]eps."""
    return (1 - sigma_min) * noisy - (sigma_min + (1 - sigma_min) * t) * velocity


def denormalize(slat, statistics):
    # Construct in FP32 directly: BF16.new_tensor(...).float() would already
    # have rounded the statistics before denormalization.
    mean = torch.as_tensor(statistics['mean'], device=slat.feats.device, dtype=torch.float32)
    std = torch.as_tensor(statistics['std'], device=slat.feats.device, dtype=torch.float32)
    if (mean.numel() != slat.feats.shape[-1] or std.shape != mean.shape or
        not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Invalid VAE normalization statistics')
    return slat.replace(slat.feats.float() * std + mean)


def surface_density(raw, resolution, grid_edge, voxel_margin):
    """Trilinear splat of dual vertices, weighted by soft edge intersections.

    Input raw is [batch,x,y,z] plus seven FDG decoder channels. Positions use
    voxel offsets (not voxel centers). Area scaling removes the leading R^2
    sampling-density factor; it is approximate and needs a GT/GT floor audit.
    """
    if resolution < grid_edge or grid_edge < 2:
        raise ValueError('Invalid coarse grid size')
    if raw.feats.shape[-1] != 7:
        raise ValueError('TF3 geometry loss requires the seven-channel FDG decoder')
    xyz = raw.coords[:, 1:].float()
    if ((xyz < 0) | (xyz >= resolution)).any():
        raise ValueError('Decoded coordinates exceed the explicit output resolution')
    values = raw.feats.float()
    offsets = (1 + 2 * voxel_margin) * torch.sigmoid(values[:, :3]) - voxel_margin
    points = ((xyz + offsets) / resolution * grid_edge - .5).clamp(0, grid_edge - 1)
    lower = points.floor().long()
    frac = points - lower
    confidence = 1 - (1 - values[:, 3:6].sigmoid()).prod(-1)
    weight = confidence * (grid_edge / resolution) ** 2
    batch = raw.coords[:, 0].long()
    if batch.numel() == 0:
        raise ValueError('Empty decoded geometry')
    batch_size = raw.shape[0]
    density = values.new_zeros(batch_size * grid_edge ** 3)
    for corner in itertools.product((0, 1), repeat=3):
        bit = lower.new_tensor(corner)
        index = (lower + bit).clamp_max(grid_edge - 1)
        coefficient = torch.where(bit.bool(), frac, 1 - frac).prod(-1) * weight
        flat = ((batch * grid_edge + index[:, 0]) * grid_edge + index[:, 1]) * grid_edge + index[:, 2]
        density = density.index_add(0, flat, coefficient)
    # Bounded soft surface presence. This is not filled interior occupancy.
    return -torch.expm1(-density.reshape(batch_size, grid_edge, grid_edge, grid_edge))


def balanced_field_loss(pred, target):
    # Avoid letting a large empty background dominate the coarse geometry loss.
    weights = 1 + 4 * target.detach()
    return ((pred - target).abs() * weights).sum() / weights.sum().clamp_min(1)


class _BackwardScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class FrozenGeometryConsistency:
    """Lazy frozen decoder; HR retains autograd, LR target is detached.

    Kept outside the denoiser/checkpoint: no inference or optimizer overhead.
    Native sparse-convolution input backward MUST pass GPU preflight before use.
    """
    def __init__(self, decoder_path, hr_statistics, lr_statistics,
                 hr_resolution=1024, lr_resolution=512, grid_edges=(8, 16),
                 decoder_checkpointing=True):
        if hr_resolution != 2 * lr_resolution:
            raise ValueError('First TF3 experiment uses a fixed x2 refinement ratio')
        self.decoder_path = decoder_path
        self.hr_statistics, self.lr_statistics = hr_statistics, lr_statistics
        self.hr_resolution, self.lr_resolution = hr_resolution, lr_resolution
        self.grid_edges = tuple(grid_edges)
        self.decoder_checkpointing = bool(decoder_checkpointing)
        self.decoder = None

    def _load(self, device):
        if self.decoder is None:
            from .frozen_conv_fix import install
            self.conv_backward_fix = install()
            from trellis2 import models
            from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
            from trellis2.modules.sparse.conv.conv import SparseConv3d, SparseInverseConv3d
            # Checkpoint constructors initialize weights before loading. Keep
            # this lazy allocation from perturbing the training noise sequence.
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                decoder = models.from_pretrained(self.decoder_path).to(device).eval()
            if not isinstance(decoder, FlexiDualGridVaeDecoder):
                raise TypeError('Unsupported VAE: do not guess the output channel meaning')
            decoder.requires_grad_(False)
            # Preserve the existing Blackwell sparse convolution dtype fix.
            def match_dtype(module, args):
                x = args[0]
                if x.feats.dtype != module.weight.dtype:
                    return (x.replace(x.feats.to(module.weight.dtype)), *args[1:])
                return args
            for module in decoder.modules():
                if isinstance(module, (SparseConv3d, SparseInverseConv3d)):
                    module.register_forward_pre_hook(match_dtype)
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = self.decoder_checkpointing
            self.decoder = decoder
        if next(self.decoder.parameters()).device != device:
            raise ValueError('Geometry decoder device changed unexpectedly')
        return self.decoder

    def _raw(self, normalized, statistics, resolution):
        from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
        decoder = self._load(normalized.feats.device)
        physical = denormalize(normalized, statistics)
        dtype = next(decoder.parameters()).dtype
        physical = physical.replace(physical.feats.to(dtype))
        decoder.set_resolution(resolution)
        # Bypass mesh extraction and hard intersection thresholding, not the VAE.
        # eval mode is essential: training mode expects GT subdivision caches.
        raw = SparseUnetVaeDecoder.forward(decoder, physical)
        if isinstance(raw, tuple):
            raise RuntimeError('Unexpected native decoder output schema')
        return raw

    def __call__(self, predicted_clean_hr, clean_lr):
        with torch.no_grad():
            raw_lr = self._raw(clean_lr, self.lr_statistics, self.lr_resolution)
            targets = [surface_density(raw_lr, self.lr_resolution, g,
                                       self.decoder.voxel_margin).detach() for g in self.grid_edges]
            del raw_lr
        # Preserve the mathematical derivative while protecting the frozen FP16
        # decoder's internal backward from underflow. Unscale before gradients
        # enter the FP32 clean-latent/DiT graph; no optimizer loss scaling needed.
        protected = predicted_clean_hr.replace(_BackwardScale.apply(predicted_clean_hr.feats, 1/1024.))
        raw_hr = self._raw(protected, self.hr_statistics, self.hr_resolution)
        fields = [surface_density(raw_hr, self.hr_resolution, g,
                                  self.decoder.voxel_margin) for g in self.grid_edges]
        losses = [balanced_field_loss(p, q) for p, q in zip(fields, targets)]
        loss = torch.stack(losses).mean()
        if torch.is_grad_enabled() and predicted_clean_hr.feats.requires_grad and not loss.requires_grad:
            raise RuntimeError('Geometry loss lost the HR latent gradient')
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite TF3 geometry consistency')
        return _BackwardScale.apply(loss, 1024.), {f'coarse_{g}': float(v.detach()) for g, v in zip(self.grid_edges, losses)}
