"""HR flow matching with frozen M0 and clean, LR-only-adapted global stream."""
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from .sparse_flow_matching import ImageConditionedSparseFlowMatchingCFGTrainer


class M2LRGuidanceTrainer(ImageConditionedSparseFlowMatchingCFGTrainer):
    def __init__(self, models, dataset, *, m0_adapter_ckpt, lr_dropout=0.1, **kwargs):
        self.m0_adapter_ckpt = m0_adapter_ckpt
        self.lr_dropout = float(lr_dropout)
        if not 0 <= self.lr_dropout < 1:
            raise ValueError('Invalid LR dropout')
        if kwargs.get('mix_precision_mode') != 'inflat_all':
            raise ValueError('M2 requires FP32 optimizer master weights (inflat_all)')
        super().__init__(models, dataset, **kwargs)

    def finetune_from(self, finetune_ckpt):
        # BasicTrainer calls this after constructing FP32 optimizer master weights.
        # Official weights load first; then frozen M0 adapters. LR masters are unaffected.
        super().finetune_from(finetune_ckpt)
        self.models['denoiser'].load_m0_adapter(self.m0_adapter_ckpt)

    def training_losses(self, x_0, cond=None, **kwargs):
        if 'lr_cond' not in kwargs:
            raise ValueError('Clean paired LR latent required')
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        xt = self.diffuse(x_0, t, noise=noise)
        image_cond = self.get_cond(cond, **kwargs)
        drop = torch.rand(x_0.shape[0], device=x_0.device) < self.lr_dropout
        pred = self.training_models['denoiser'](xt, t*1000, image_cond, lr_drop_mask=drop, **kwargs)
        target = self.get_v(x_0, noise, t)
        loss = F.mse_loss(pred.feats.float(), target.feats.float())
        terms = edict(loss=loss, mse=loss, lr_dropout_rate=float(drop.float().mean()))
        for i in range(10):
            batch_ids = torch.where((t >= i/10) & (t < (i+1)/10))[0].tolist()
            if batch_ids:
                terms[f'bin_{i}'] = {'mse': float(torch.stack([
                    F.mse_loss(pred.feats[x_0.layout[j]].float(), target.feats[x_0.layout[j]].float()).detach()
                    for j in batch_ids]).mean())}
        return terms, {}
