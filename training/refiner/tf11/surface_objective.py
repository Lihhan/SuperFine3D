"""Local depth/normal losses. Coordinates and depth are in LR patch units.

This module adds no inference operators. Mask erosion excludes silhouettes and
missing pixels from local filtering; silhouette/topology needs separate checks.
"""
import torch
import torch.nn.functional as F


def erode(mask, radius):
    k = 2 * radius + 1
    return F.avg_pool2d(F.pad(mask.float(), (radius,) * 4), k, 1).ge(1 - 1e-6).float()


def masked_mean(value, mask):
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def surface_objective(depth, target_depth, normal, target_normal, valid, scales=(3, 7)):
    valid = valid.detach().float()
    losses = {'position': masked_mean(F.smooth_l1_loss(depth, target_depth,
                       reduction='none', beta=.02), valid)}
    details = []
    for k in scales:
        if k % 2 != 1:
            raise ValueError('Odd filter size required')
        interior = erode(valid, k // 2)
        p = depth - F.avg_pool2d(depth, k, 1, k // 2)
        q = target_depth - F.avg_pool2d(target_depth, k, 1, k // 2)
        details.append(masked_mean(F.smooth_l1_loss(p, q, reduction='none', beta=.01), interior))
    losses['detail'] = torch.stack(details).mean()
    # Equivalent to 1-cos for unit normals, with exact zero at identical inputs.
    losses['normal'] = masked_mean(.5 * (normal-target_normal).square().sum(1,keepdim=True), valid)
    variation = []
    for axis in (-1, -2):
        a = [slice(None)] * 4; b = a.copy()
        a[axis] = slice(1, None); b[axis] = slice(None, -1)
        a, b = tuple(a), tuple(b)
        pair = valid[a] * valid[b]
        residual = (normal[a] - normal[b]) - (target_normal[a] - target_normal[b])
        variation.append(masked_mean(F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none',beta=.02), pair))
    losses['normal_variation'] = torch.stack(variation).mean()
    return losses


def face_objective(logits, target, valid, weights=None):
    """Call only AFTER coordinate+edge-axis correspondence has been verified."""
    valid = valid.bool()
    if logits.shape != target.shape or valid.shape != logits.shape:
        raise ValueError('Face correspondence shape mismatch')
    values = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
    if weights is None: weights=torch.ones_like(values)
    if weights.shape!=values.shape:raise ValueError('Face weight shape mismatch')
    groups=[]
    for group in [valid & (target>.5),valid & (target<=.5)]:
        if group.any():groups.append((values[group]*weights[group]).sum()/weights[group].sum().clamp_min(1e-12))
    return torch.stack(groups).mean() if groups else logits.sum()*0
