"""TF1: macroblock-bounded HR MSGA and one-way clean LR memory, based on T1."""
import torch
from .m2_joint_flow import ElasticM2JointSLatFlowModel
from .dual_resolution_flow import attend
from .tf1_routing import bound_routes, lr_self_post


class ElasticTF1SLatFlowModel(ElasticM2JointSLatFlowModel):
    def __init__(self, *args, hr_block_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.hr_block_size = int(hr_block_size)
        if self.hr_block_size < self.cell_size or self.hr_block_size % self.cell_size:
            raise ValueError('Macroblock edge must be a positive multiple of cell edge')
        if self.lr_pooling != 'none':
            raise ValueError('TF1 requires native LR tokens')

    def _signature(self):
        return ('TF1', self.hr_block_size, super()._signature())

    def _routes(self, x, shift):
        key = (f'tf1_bound_{self.hr_block_size}_{self.cell_size}_{shift}_'
               f'{self.halo_voxels}_{self.msga_ring_mult}_{self.msga_samples_per_ring}')
        cached = x.get_spatial_cache(key)
        if cached is None:
            cached = bound_routes(x.coords, super()._routes(x, shift), self.hr_block_size)
            x.register_spatial_cache(key, cached)
        return cached

    def _lr_step(self, i, lr, lr_mod, cond, positions):
        block = self.blocks[i]
        hidden, gate, shift, scale, mlp_gate = self._pre(block, lr, lr_mod)
        q, k, v = self._lr_qkv(i, hidden, positions)
        if i == self.num_blocks - 1:
            return lr, k, v
        output = torch.cat([attend(q[sl], k[sl], v[sl]) for sl in lr.layout])
        flat = output.reshape(-1, self.model_channels)
        base = block.self_attn.to_out(flat)
        output = hidden.replace(base + self.lr_out_delta[i](flat).to(base.dtype))
        return lr_self_post(block, lr, output, gate, shift, scale, mlp_gate), k, v
