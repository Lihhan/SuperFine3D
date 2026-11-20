from typing import *
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .sparse_elastic_mixin import SparseTransformerElasticMixin
from ..modules.sparse.attention.lr_guidance import (
    smoothstep_schedule, pool_sparse_lr_memory,
    LRHRLatentAligner, SparseLRGlobalAttention, SparseLRContextEncoder,
    SparseLRSynchronizedGlobalEncoder, SparseLRPyramidEncoder,
    SparseLRSelectedAttention, SparseLRDeterministicMultiRingAttention,
    SparseLRParentAttention,
)
    

class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])
            
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)
            
        elif self.initialization == 'scaled':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            
            # Scaled init for to_out and ffn2
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)
            
            # Initialize input layer to make the initial representation have variance 1
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)
            
            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            
            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs
    ) -> sp.SparseTensor:
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h


class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass


class ElasticLocalSLatFlowModel(ElasticSLatFlowModel):
    """Official TRELLIS.2 SLat flow with HR self-attention restricted to 3D windows.

    The parameterization is identical to ElasticSLatFlowModel, so the official
    1024 checkpoint loads strictly. Only the self-attention visibility pattern
    changes. double_windowed gives half the heads unshifted windows and half
    the heads windows shifted by window_size/2.
    """
    def __init__(
        self,
        local_window_size: int = 8,
        local_attn_mode: str = "double_windowed",
        **kwargs,
    ):
        if local_attn_mode not in {"windowed", "double_windowed"}:
            raise ValueError(f"Unsupported local_attn_mode: {local_attn_mode}")
        super().__init__(**kwargs)
        self.local_window_size = int(local_window_size)
        self.local_attn_mode = local_attn_mode
        for block_index, block in enumerate(self.blocks):
            attn = block.self_attn
            attn.attn_mode = local_attn_mode
            attn.window_size = self.local_window_size
            if local_attn_mode == "windowed":
                shift = 0 if block_index % 2 == 0 else self.local_window_size // 2
                attn.shift_window = (shift, shift, shift)
            else:
                attn.shift_window = None


class ElasticPointelligenceMSGASLatFlowModel(ElasticSLatFlowModel):
    """Official TRELLIS.2 weights with Pointelligence MSGA self-attention."""
    def __init__(
        self,
        msga_cell_scaler: float = 8.0,
        msga_ring_mult=(1.0, 0.5),
        msga_samples_per_ring: int = 16,
        msga_rope_mode: str = "pointelligence",
        msga_num_partitions: int = 1,
        msga_partition_gate_threshold_normalized: Optional[float] = None,
        msga_timestep_scale: float = 1000.0,
        msga_fa_backend: str = "auto",
        msga_stable_coverage: bool = False,
        msga_full_ring0: bool = False,
        msga_fixed_shift_fractions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        msga_schedule_mode: str = "hard",
        msga_schedule_low: float = 0.2,
        msga_schedule_high: float = 0.8,
        msga_far_min_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.pe_mode != "rope":
            raise ValueError("Pointelligence MSGA comparison requires pe_mode='rope'")
        self.msga_cell_scaler = float(msga_cell_scaler)
        self.msga_ring_mult = tuple(float(x) for x in msga_ring_mult)
        self.msga_samples_per_ring = int(msga_samples_per_ring)
        self.msga_num_partitions = int(msga_num_partitions)
        self.msga_partition_gate_threshold_normalized = (
            None if msga_partition_gate_threshold_normalized is None
            else float(msga_partition_gate_threshold_normalized)
        )
        self.msga_timestep_scale = float(msga_timestep_scale)
        if self.msga_cell_scaler <= 0:
            raise ValueError("msga_cell_scaler must be > 0")
        if not self.msga_ring_mult or any(x < 0 for x in self.msga_ring_mult) or not any(x > 0 for x in self.msga_ring_mult):
            raise ValueError("msga_ring_mult must contain at least one positive, non-negative value")
        if self.msga_samples_per_ring < 1:
            raise ValueError("msga_samples_per_ring must be >= 1")
        if self.msga_num_partitions < 1:
            raise ValueError("msga_num_partitions must be >= 1")
        if (self.msga_partition_gate_threshold_normalized is not None and
                not 0.0 <= self.msga_partition_gate_threshold_normalized <= 1.0):
            raise ValueError(
                "msga_partition_gate_threshold_normalized must be in [0, 1]"
            )
        if self.msga_timestep_scale <= 0:
            raise ValueError("msga_timestep_scale must be > 0")
        if msga_fa_backend not in {"auto", "fa2", "fa3"}:
            raise ValueError(f"Unknown MSGA FlashAttention backend: {msga_fa_backend}")
        self.msga_fa_backend = msga_fa_backend
        self.msga_stable_coverage = bool(msga_stable_coverage)
        self.msga_full_ring0 = bool(msga_full_ring0)
        self.msga_fixed_shift_fractions = tuple(
            tuple(float(v) for v in shift) for shift in msga_fixed_shift_fractions
        )
        if msga_schedule_mode not in {"hard", "smooth"}:
            raise ValueError("msga_schedule_mode must be 'hard' or 'smooth'")
        self.msga_schedule_mode = msga_schedule_mode
        self.msga_schedule_low = float(msga_schedule_low)
        self.msga_schedule_high = float(msga_schedule_high)
        self.msga_far_min_scale = float(msga_far_min_scale)
        if not 0 <= self.msga_schedule_low < self.msga_schedule_high <= 1:
            raise ValueError("MSGA smooth schedule bounds must satisfy 0<=low<high<=1")
        if not 0 <= self.msga_far_min_scale <= 1:
            raise ValueError("msga_far_min_scale must lie in [0,1]")
        self._msga_base_ring_mult = self.msga_ring_mult
        if msga_rope_mode not in {"pointelligence", "trellis"}:
            raise ValueError(f"Unknown MSGA RoPE mode: {msga_rope_mode}")
        self.msga_rope_mode = msga_rope_mode
        for block_index, block in enumerate(self.blocks):
            attn = block.self_attn
            attn.attn_mode = "msga"
            attn.msga_cell_scaler = self.msga_cell_scaler
            attn.msga_ring_mult = self.msga_ring_mult
            attn.msga_samples_per_ring = self.msga_samples_per_ring
            attn.msga_rope_mode = self.msga_rope_mode
            attn.msga_num_partitions = self.msga_num_partitions
            attn.msga_partition_index = block_index % self.msga_num_partitions
            attn.msga_fa_backend = self.msga_fa_backend
            attn.msga_stable_coverage = self.msga_stable_coverage
            attn.msga_full_ring0 = self.msga_full_ring0
            attn.msga_fixed_shift_fractions = self.msga_fixed_shift_fractions

    def _set_timestep_gated_partitions(self, t: torch.Tensor) -> bool:
        """Select shifted partitions for the current denoising timestep.

        High-noise steps alternate partitions across DiT blocks. Low-noise
        steps keep partition zero in every block to preserve a stable local
        neighborhood while details are refined. The current 4K training setup
        uses one sample per GPU; for larger batches, the batch-mean normalized
        timestep defines the shared sparse graph. TRELLIS passes timesteps to
        the denoiser on a 0..1000 scale, so the gate divides by
        ``msga_timestep_scale`` before applying the normalized threshold.
        """
        threshold = self.msga_partition_gate_threshold_normalized
        if threshold is None or self.msga_num_partitions == 1:
            return False
        normalized_t = t.detach().float() / self.msga_timestep_scale
        if self.msga_schedule_mode == "smooth":
            u = ((normalized_t.mean().item() - self.msga_schedule_low) /
                 (self.msga_schedule_high - self.msga_schedule_low))
            u = min(1.0, max(0.0, u))
            strength = u * u * (3.0 - 2.0 * u)
            candidates = list(range(1, len(self.blocks), 2))
            # Golden-ratio ordering distributes active shifted blocks across depth.
            order = sorted(
                candidates,
                key=lambda i: ((i + 1) * 0.6180339887498949) % 1.0,
            )
            selected = set(order[:int(round(strength * len(candidates)))])
            far_scale = self.msga_far_min_scale + (
                1.0 - self.msga_far_min_scale
            ) * strength
            current_ring_mult = tuple(
                value if ring == 0 else value * far_scale
                for ring, value in enumerate(self._msga_base_ring_mult)
            )
            for block_index, block in enumerate(self.blocks):
                block.self_attn.msga_partition_index = (
                    1 if block_index in selected else 0
                )
                block.self_attn.msga_ring_mult = current_ring_mult
            self._msga_last_schedule = {
                "normalized_t": float(normalized_t.mean().item()),
                "strength": float(strength),
                "shifted_blocks": len(selected),
                "ring_mult": current_ring_mult,
            }
            return bool(selected)
        high_noise = bool(normalized_t.mean().item() >= threshold)
        for block_index, block in enumerate(self.blocks):
            block.self_attn.msga_partition_index = (
                block_index % self.msga_num_partitions if high_noise else 0
            )
        return high_noise

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        self._set_timestep_gated_partitions(t)
        return super().forward(x, t, cond, concat_cond=concat_cond, **kwargs)


class ElasticMSGAGlobalLRGuidanceSLatFlowModel(ElasticPointelligenceMSGASLatFlowModel):
    """Experiment B: stable MSGA plus bounded global LR latent guidance."""
    def __init__(self, lr_guidance_blocks=(5, 11, 17, 23, 29), lr_memory_grid_size=8,
                 lr_resolution=32, lr_coord_scale=2.0, lr_schedule_low=0.2,
                 lr_schedule_high=0.8, lr_min_scale=0.1,
                 lr_context_depth=0, lr_context_channels=256,
                 lr_context_heads=8, lr_context_mlp_ratio=2.0,
                 lr_gate_init=0.0, lr_zero_init_output=False,
                 lr_global_temperature=1.0, lr_coord_offset=0.0, **kwargs):
        super().__init__(**kwargs)
        self.lr_guidance_blocks = tuple(int(i) for i in lr_guidance_blocks)
        if any(i < 0 or i >= self.num_blocks for i in self.lr_guidance_blocks):
            raise ValueError('lr_guidance_blocks contains an invalid block index')
        self.lr_memory_grid_size, self.lr_resolution = int(lr_memory_grid_size), int(lr_resolution)
        self.lr_coord_scale = float(lr_coord_scale)
        self.lr_schedule_low, self.lr_schedule_high = float(lr_schedule_low), float(lr_schedule_high)
        self.lr_min_scale = float(lr_min_scale)
        self.lr_context_depth = int(lr_context_depth)
        self.lr_context_channels = int(lr_context_channels)
        self.lr_context_heads = int(lr_context_heads)
        self.lr_context_mlp_ratio = float(lr_context_mlp_ratio)
        self.lr_gate_init = float(lr_gate_init)
        self.lr_zero_init_output = bool(lr_zero_init_output)
        self.lr_global_temperature = float(lr_global_temperature)
        self.lr_coord_offset = float(lr_coord_offset)
        if self.lr_context_depth < 0:
            raise ValueError('lr_context_depth must be non-negative')
        if self.lr_context_mlp_ratio <= 0:
            raise ValueError('lr_context_mlp_ratio must be positive')
        if not 0 <= self.lr_schedule_low < self.lr_schedule_high <= 1:
            raise ValueError('LR schedule must satisfy 0<=low<high<=1')
        if not 0 <= self.lr_min_scale <= 1:
            raise ValueError('lr_min_scale must lie in [0,1]')
        self.lr_global_attn = nn.ModuleDict({str(i): SparseLRGlobalAttention(
            self.model_channels, self.num_heads,
            self.blocks[i].self_attn.rope.rope_freq,
            gate_init=self.lr_gate_init,
            zero_init_output=self.lr_zero_init_output,
            attention_temperature=self.lr_global_temperature,
            coord_offset=self.lr_coord_offset,
        ) for i in self.lr_guidance_blocks})
        rope_freq = self.blocks[self.lr_guidance_blocks[0]].self_attn.rope.rope_freq
        self.lr_context_encoder = (
            SparseLRContextEncoder(
                self.model_channels, self.lr_context_channels,
                self.lr_context_heads, self.lr_context_depth,
                self.lr_context_mlp_ratio, rope_freq,
            ) if self.lr_context_depth > 0 else None
        )
        self.lr_global_attn.apply(partial(convert_module_to, dtype=self.dtype))
        if self.lr_context_encoder is not None:
            self.lr_context_encoder.apply(partial(convert_module_to, dtype=self.dtype))

    def _lr_strength(self, t):
        return smoothstep_schedule(t, self.msga_timestep_scale, self.lr_schedule_low,
                                   self.lr_schedule_high, self.lr_min_scale)

    def _prepare_lr(self, lr_cond, t_context=None, t=None, lr_noise=None, lr_drop_mask=None):
        lr_h = manual_cast(self.input_layer(lr_cond), self.dtype)
        memory, memory_xyz = pool_sparse_lr_memory(
            lr_h, lr_cond.coords, self.lr_memory_grid_size, self.lr_resolution)
        if self.lr_context_encoder is not None:
            memory = self.lr_context_encoder(memory, memory_xyz, self.lr_coord_scale)
        return lr_h, memory, memory_xyz

    def _apply_lr_guidance(self, i, h, lr_h, memory, memory_xyz, strength):
        if i in self.lr_guidance_blocks:
            h = self.lr_global_attn[str(i)](h, memory, memory_xyz, self.lr_coord_scale, strength)
        return h

    def forward(self, x, t, cond, concat_cond=None, lr_cond=None, **kwargs):
        if lr_cond is None:
            raise ValueError('lr_cond is required for LR-guided MSGA')
        self._set_timestep_gated_partitions(t)
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)
        h = manual_cast(self.input_layer(x), self.dtype)
        raw_t_emb = self.t_embedder(t)
        t_emb = raw_t_emb
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb, cond = manual_cast(t_emb, self.dtype), manual_cast(cond, self.dtype)
        if self.pe_mode == 'ape':
            h = h + manual_cast(self.pos_embedder(h.coords[:, 1:]), self.dtype)
        lr_h, memory, memory_xyz = self._prepare_lr(
            lr_cond, raw_t_emb, t, kwargs.get('lr_noise'), kwargs.get('lr_drop_mask')
        )
        strength = self._lr_strength(t)
        for i, block in enumerate(self.blocks):
            h = block(h, t_emb, cond)
            h = self._apply_lr_guidance(i, h, lr_h, memory, memory_xyz, strength)
        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.out_layer(h)


class ElasticMSGAPyramidLRGuidanceSLatFlowModel(ElasticMSGAGlobalLRGuidanceSLatFlowModel):
    """Experiment B-pyramid: learned spatial LR compression, without selection."""
    def __init__(self, lr_pyramid_depth=2, lr_pyramid_channels=256,
                 lr_pyramid_heads=8, lr_pyramid_mlp_ratio=2.0, **kwargs):
        # The pyramid owns LR contextualization; disable the mean-pool context path.
        kwargs['lr_context_depth'] = 0
        super().__init__(**kwargs)
        rope_freq = self.blocks[self.lr_guidance_blocks[0]].self_attn.rope.rope_freq
        self.lr_pyramid_encoder = SparseLRPyramidEncoder(
            self.model_channels, lr_pyramid_channels, lr_pyramid_heads,
            lr_pyramid_depth, lr_pyramid_mlp_ratio, rope_freq,
        )
        self.lr_pyramid_encoder.apply(partial(convert_module_to, dtype=self.dtype))

    def _prepare_lr(self, lr_cond, t_context=None, t=None, lr_noise=None, lr_drop_mask=None):
        lr_h = manual_cast(self.input_layer(lr_cond), self.dtype)
        memory, memory_xyz = pool_sparse_lr_memory(
            lr_h, lr_cond.coords, self.lr_memory_grid_size, self.lr_resolution
        )
        memory = self.lr_pyramid_encoder(
            lr_h, memory, memory_xyz, self.lr_coord_scale
        )
        return lr_h, memory, memory_xyz


class ElasticMSGASelectLRGuidanceSLatFlowModel(ElasticMSGAPyramidLRGuidanceSLatFlowModel):
    """Experiment B-select: pyramid memory plus selected original LR tokens."""
    def __init__(self, lr_select_topk_blocks=4, lr_select_tokens_per_block=16,
                 lr_select_hr_cell_size=8, lr_select_chunk_size=1024, **kwargs):
        super().__init__(**kwargs)
        rope_freq = self.blocks[self.lr_guidance_blocks[0]].self_attn.rope.rope_freq
        self.lr_selected_attn = nn.ModuleDict({str(i): SparseLRSelectedAttention(
            self.model_channels, self.num_heads, rope_freq,
            lr_select_topk_blocks, lr_select_tokens_per_block,
            lr_select_hr_cell_size, self.lr_resolution // self.lr_memory_grid_size,
            lr_select_chunk_size,
        ) for i in self.lr_guidance_blocks})
        self.lr_selected_attn.apply(partial(convert_module_to, dtype=self.dtype))

    def _apply_lr_guidance(self, i, h, lr_h, memory, memory_xyz, strength):
        h = super()._apply_lr_guidance(i, h, lr_h, memory, memory_xyz, strength)
        if i in self.lr_guidance_blocks:
            h = self.lr_selected_attn[str(i)](
                h, lr_h, memory, memory_xyz, self.lr_global_attn[str(i)],
                self.lr_coord_scale, strength,
            )
        return h


class ElasticMSGADeterministicLRGuidanceSLatFlowModel(
    ElasticMSGAPyramidLRGuidanceSLatFlowModel
):
    """Pyramid global memory plus deterministic parent and multi-ring routing."""
    def __init__(self, lr_ring_radii=(0, 1, 2, 4),
                 lr_ring_budgets=(1, 12, 12, 8),
                 lr_ring_chunk_size=1024, lr_ring_output_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        scale = int(round(self.lr_coord_scale))
        if abs(scale - self.lr_coord_scale) > 1e-6:
            raise ValueError("deterministic routing requires integer scale")
        rope_freq = self.blocks[
            self.lr_guidance_blocks[0]
        ].self_attn.rope.rope_freq
        self.lr_ring_attn = nn.ModuleDict({
            str(i): SparseLRDeterministicMultiRingAttention(
                self.model_channels, self.num_heads, rope_freq,
                self.lr_resolution, scale, lr_ring_radii,
                lr_ring_budgets, lr_ring_chunk_size,
                coord_offset=self.lr_coord_offset,
                output_scale=lr_ring_output_scale,
            ) for i in self.lr_guidance_blocks
        })
        self.lr_ring_attn.apply(partial(convert_module_to, dtype=self.dtype))

    def _apply_lr_guidance(self, i, h, lr_h, memory, memory_xyz, strength):
        h = super()._apply_lr_guidance(
            i, h, lr_h, memory, memory_xyz, strength
        )
        if i in self.lr_guidance_blocks:
            h = self.lr_ring_attn[str(i)](
                h, lr_h, self.lr_global_attn[str(i)], strength
            )
        return h


class ElasticMSGASynchronizedLRGuidanceSLatFlowModel(
    ElasticMSGADeterministicLRGuidanceSLatFlowModel
):
    """Timestep-synchronized LR global stream with deterministic HR routing."""
    def __init__(
        self,
        lr_alignment_hidden=256,
        lr_alignment_depth=3,
        lr_alignment_ckpt=None,
        lr_alignment_frozen=True,
        lr_sync_depth=2,
        lr_sync_channels=256,
        lr_sync_heads=8,
        lr_sync_mlp_ratio=2.0,
        lr_noise_time_scale=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr_noise_time_scale = float(lr_noise_time_scale)
        if not 0.0 <= self.lr_noise_time_scale <= 1.0:
            raise ValueError('lr_noise_time_scale must lie in [0,1]')
        self.lr_aligner = LRHRLatentAligner(
            self.in_channels, lr_alignment_hidden, lr_alignment_depth
        )
        if lr_alignment_ckpt is not None:
            state = torch.load(
                lr_alignment_ckpt, map_location="cpu", weights_only=True
            )
            self.lr_aligner.load_state_dict(state, strict=True)
        if lr_alignment_frozen:
            self.lr_aligner.requires_grad_(False)
        rope_freq = self.blocks[
            self.lr_guidance_blocks[0]
        ].self_attn.rope.rope_freq
        self.lr_sync_encoder = SparseLRSynchronizedGlobalEncoder(
            self.model_channels, self.model_channels,
            lr_sync_channels, lr_sync_heads, lr_sync_depth,
            lr_sync_mlp_ratio, rope_freq,
        )
        self.lr_sync_encoder.apply(
            partial(convert_module_to, dtype=self.dtype)
        )
        self.lr_null_embedding = nn.Parameter(
            torch.zeros(self.in_channels, dtype=self.dtype)
        )

    def align_lr(self, lr_cond):
        aligned = self.lr_aligner(lr_cond.feats.float()).to(lr_cond.feats.dtype)
        return lr_cond.replace(aligned)

    def _prepare_lr(self, lr_cond, t_context=None, t=None, lr_noise=None, lr_drop_mask=None):
        if t_context is None or t is None:
            raise ValueError("synchronized LR stream requires timestep context")
        if lr_noise is None:
            raise ValueError(
                "synchronized LR stream requires fixed lr_noise for all timesteps"
            )
        clean = self.align_lr(lr_cond)
        batch_index = clean.coords[:, 0].long()
        if lr_drop_mask is not None:
            drop = lr_drop_mask.to(clean.device, dtype=torch.bool)[batch_index, None]
            null = self.lr_null_embedding.to(clean.feats.dtype)[None].expand_as(clean.feats)
            clean = clean.replace(torch.where(drop, null, clean.feats))
        normalized_t = ((t.float() / self.msga_timestep_scale) * self.lr_noise_time_scale).clamp(0, 1)[batch_index, None]
        sigma = self.lr_noise_sigma_min + (
            1.0 - self.lr_noise_sigma_min
        ) * normalized_t
        lr_t = clean.replace(
            (1.0 - normalized_t) * clean.feats
            + sigma * lr_noise.feats.to(clean.feats.dtype)
        )
        lr_h = manual_cast(self.input_layer(lr_t), self.dtype)
        memory, memory_xyz = pool_sparse_lr_memory(
            lr_h, lr_t.coords, self.lr_memory_grid_size, self.lr_resolution
        )
        memory = self.lr_pyramid_encoder(
            lr_h, memory, memory_xyz, self.lr_coord_scale
        )
        memory = self.lr_sync_encoder(
            memory, memory_xyz, self.lr_coord_scale, t_context
        )
        return lr_h, memory, memory_xyz

    lr_noise_sigma_min = 1e-5

class ElasticMSGAGlobalParentLRGuidanceSLatFlowModel(ElasticMSGAGlobalLRGuidanceSLatFlowModel):
    """Experiment C: experiment B plus coordinate-aligned 3x3x3 LR parents."""
    def __init__(self, lr_parent_chunk_size=2048, **kwargs):
        super().__init__(**kwargs)
        scale = int(round(self.lr_coord_scale))
        if abs(scale - self.lr_coord_scale) > 1e-6:
            raise ValueError('Parent guidance requires an integer lr_coord_scale')
        self.lr_parent_attn = nn.ModuleDict({str(i): SparseLRParentAttention(
            self.model_channels, self.num_heads, self.lr_resolution, scale, lr_parent_chunk_size
        ) for i in self.lr_guidance_blocks})
        self.lr_parent_attn.apply(partial(convert_module_to, dtype=self.dtype))

    def _apply_lr_guidance(self, i, h, lr_h, memory, memory_xyz, strength):
        h = super()._apply_lr_guidance(i, h, lr_h, memory, memory_xyz, strength)
        if i in self.lr_guidance_blocks:
            h = self.lr_parent_attn[str(i)](
                h, lr_h, self.lr_global_attn[str(i)], strength
            )
        return h
