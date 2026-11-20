from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', int(np.ceil(os.cpu_count() / torch.cuda.device_count()))))),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))

        # inference
        sampler = self.get_sampler()
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch_data = {k: v[i:i+batch_size] for k, v in data.items()}
            batch_data = recursive_to_device(batch_data, 'cuda')
            noise = batch_data['x_0'].replace(torch.randn_like(batch_data['x_0'].feats))
            cond_vis.append(self.vis_cond(**batch_data))
            del batch_data['x_0']
            args = self.get_inference_cond(**batch_data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=12, guidance_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)
        sample = sp.sparse_cat(sample)
        
        sample_gt = {k: v for k, v in data.items()}
        sample = {k: v if k != 'x_0' else sample for k, v in data.items()}
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass


class FrozenBackboneHighNoiseLRImageConditionedSparseFlowMatchingCFGTrainer(
    ImageConditionedMixin, SparseFlowMatchingCFGTrainer
):
    """Diagnostic trainer that asks whether LR latents are usable at high noise.

    The pretrained denoiser is frozen before optimizer construction. Only
    explicitly named LR adapters remain trainable, while timesteps are sampled
    continuously from the configured high-noise interval. This isolates LR
    information use from backbone relearning.
    """
    def __init__(
        self,
        models,
        dataset,
        *,
        trainable_patterns=(
            "lr_global_attn",
            "lr_context_encoder",
            "lr_pyramid_encoder",
            "lr_parent_attn",
            "lr_selected_attn",
        ),
        high_noise_min=0.70,
        high_noise_max=1.0,
        **kwargs,
    ):
        self.trainable_patterns = tuple(str(x) for x in trainable_patterns)
        self.high_noise_min = float(high_noise_min)
        self.high_noise_max = float(high_noise_max)
        if not 0.0 <= self.high_noise_min < self.high_noise_max <= 1.0:
            raise ValueError("high-noise interval must satisfy 0 <= min < max <= 1")
        trainable = []
        for model_name, model in models.items():
            for name, parameter in model.named_parameters():
                keep = any(pattern in name for pattern in self.trainable_patterns)
                parameter.requires_grad_(keep)
                if keep:
                    trainable.append(f"{model_name}.{name}")
        if not trainable:
            raise ValueError(
                "No parameters matched trainable_patterns: "
                + ", ".join(self.trainable_patterns)
            )
        self.high_noise_trainable_names = tuple(trainable)
        super().__init__(models, dataset, **kwargs)

    def sample_t(self, batch_size: int) -> torch.Tensor:
        u = torch.rand(batch_size)
        return self.high_noise_min + (
            self.high_noise_max - self.high_noise_min
        ) * u



class SynchronizedLRConsistencyImageConditionedSparseFlowMatchingCFGTrainer(
    ImageConditionedMixin, SparseFlowMatchingCFGTrainer
):
    """Train synchronized LR guidance with coordinate-aligned consistency."""
    def __init__(
        self,
        models,
        dataset,
        *,
        consistency_weight=0.1,
        consistency_schedule_low=0.5,
        consistency_schedule_high=0.9,
        trainable_patterns=None,
        adapter_init_ckpt=None,
        lr_dropout=0.0,
        **kwargs,
    ):
        self.consistency_weight = float(consistency_weight)
        self.lr_dropout = float(lr_dropout)
        if not 0.0 <= self.lr_dropout < 1.0:
            raise ValueError('lr_dropout must lie in [0,1)')
        if adapter_init_ckpt is not None:
            state = torch.load(adapter_init_ckpt, map_location='cpu', weights_only=True)
            result = models['denoiser'].load_state_dict(state, strict=False)
            allowed = ('lr_global_attn.29.', 'lr_ring_attn.29.')
            unexpected = [k for k in result.unexpected_keys if not k.startswith(allowed)]
            if unexpected:
                raise ValueError(f'unexpected adapter keys: {unexpected[:8]}')
        self.consistency_schedule_low = float(consistency_schedule_low)
        self.consistency_schedule_high = float(consistency_schedule_high)
        if not 0 <= self.consistency_schedule_low < self.consistency_schedule_high <= 1:
            raise ValueError("invalid consistency timestep schedule")
        self.trainable_patterns = (
            None if trainable_patterns is None
            else tuple(str(x) for x in trainable_patterns)
        )
        if self.trainable_patterns is not None:
            found = []
            for model_name, model in models.items():
                for name, parameter in model.named_parameters():
                    keep = parameter.requires_grad and any(
                        pattern in name for pattern in self.trainable_patterns
                    )
                    parameter.requires_grad_(keep)
                    if keep:
                        found.append(f"{model_name}.{name}")
            if not found:
                raise ValueError("trainable_patterns matched no trainable parameters")
            self.synchronized_trainable_names = tuple(found)
        super().__init__(models, dataset, **kwargs)
        for model in self.models.values():
            if hasattr(model, "lr_noise_sigma_min"):
                model.lr_noise_sigma_min = self.sigma_min

    @staticmethod
    def _aligned_cross_resolution_pairs(hr_x0, aligned_lr, lr_resolution=32):
        hr_xyz = torch.div(
            hr_x0.coords[:, 1:].long(), 2, rounding_mode="floor"
        )
        hr_batch = hr_x0.coords[:, 0].long()
        hr_key = (
            hr_batch * (lr_resolution ** 3)
            + hr_xyz[:, 0] * (lr_resolution ** 2)
            + hr_xyz[:, 1] * lr_resolution + hr_xyz[:, 2]
        )
        unique, inverse = torch.unique(
            hr_key, sorted=True, return_inverse=True
        )
        pooled = torch.zeros(
            unique.numel(), hr_x0.feats.shape[-1],
            device=hr_x0.device, dtype=hr_x0.feats.dtype,
        )
        pooled.index_add_(0, inverse, hr_x0.feats)
        count = torch.bincount(
            inverse, minlength=unique.numel()
        ).to(pooled.dtype)[:, None]
        pooled = pooled / count
        lr_xyz = aligned_lr.coords[:, 1:].long()
        lr_batch = aligned_lr.coords[:, 0].long()
        lr_key = (
            lr_batch * (lr_resolution ** 3)
            + lr_xyz[:, 0] * (lr_resolution ** 2)
            + lr_xyz[:, 1] * lr_resolution + lr_xyz[:, 2]
        )
        pos = torch.searchsorted(unique, lr_key)
        safe = pos.clamp(max=unique.numel() - 1)
        valid = (pos < unique.numel()) & (unique[safe] == lr_key)
        if not bool(valid.any()):
            raise RuntimeError("no coordinate-aligned LR/HR tokens")
        return pooled[safe[valid]], aligned_lr.feats[valid]

    def _consistency_scale(self, t):
        u = ((t.float().mean() - self.consistency_schedule_low) / (
            self.consistency_schedule_high - self.consistency_schedule_low
        )).clamp(0, 1)
        return u * u * (3 - 2 * u)

    def training_losses(self, x_0: sp.SparseTensor, cond=None, **kwargs):
        if "lr_cond" not in kwargs:
            raise ValueError("lr_cond is required")
        lr_cond = kwargs["lr_cond"]
        noise = x_0.replace(torch.randn_like(x_0.feats))
        lr_noise = lr_cond.replace(torch.randn_like(lr_cond.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        lr_drop_mask = torch.rand(x_0.shape[0], device=x_0.device) < self.lr_dropout
        pred = self.training_models["denoiser"](
            x_t, t * 1000, cond, lr_noise=lr_noise,
            lr_drop_mask=lr_drop_mask, **kwargs
        )
        target = self.get_v(x_0, noise, t)
        mse = F.mse_loss(pred.feats.float(), target.feats.float())

        token_t = t[x_t.coords[:, 0].long(), None]
        sigma = self.sigma_min + (1 - self.sigma_min) * token_t
        pred_x0 = x_t.replace(
            (1 - self.sigma_min) * x_t.feats - sigma * pred.feats
        )
        denoiser = self.training_models["denoiser"]
        denoiser = denoiser.module if hasattr(denoiser, "module") else denoiser
        with torch.no_grad():
            aligned_lr = denoiser.align_lr(lr_cond)
        pooled_hr, target_lr = self._aligned_cross_resolution_pairs(
            pred_x0, aligned_lr, denoiser.lr_resolution
        )
        consistency = F.mse_loss(
            pooled_hr.float(), target_lr.float()
        )
        cscale = self._consistency_scale(t) * (~lr_drop_mask).float().mean()
        terms = edict()
        terms["mse"] = mse
        terms["consistency"] = consistency
        terms["consistency_scale"] = float(cscale.detach())
        terms["lr_dropout_rate"] = float(lr_drop_mask.float().mean().detach())
        terms["loss"] = mse + self.consistency_weight * cscale * consistency

        mse_per_instance = np.array([
            F.mse_loss(
                pred.feats[x_0.layout[i]].float(),
                target.feats[x_0.layout[i]].float(),
            ).item() for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {
                    "mse": mse_per_instance[time_bin == i].mean()
                }
        return terms, {}


class MultiImageConditionedSparseFlowMatchingCFGTrainer(MultiImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass
