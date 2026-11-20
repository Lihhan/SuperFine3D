import os
import numpy as np
import torch

from .structured_latent import SLat
from .structured_latent_shape import FilteredImageConditionedSLatShape
from ..modules.sparse import SparseTensor


class PairedLRImageConditionedSLatShape(FilteredImageConditionedSLatShape):
    """HR Shape SLat paired with the corresponding real 512 Shape SLat."""
    def __init__(self, *args, lr_root_key='shape_latent_lr', lr_normalization=None,
                 max_lr_tokens=8192, lr_mode='paired', **kwargs):
        if lr_mode not in {'paired', 'null', 'shuffled'}:
            raise ValueError(f'Unsupported LR control: {lr_mode}')
        self.lr_root_key, self.lr_normalization = lr_root_key, lr_normalization
        self.max_lr_tokens, self.lr_mode = int(max_lr_tokens), lr_mode
        if lr_normalization is not None:
            self.lr_mean = torch.tensor(lr_normalization['mean']).reshape(1, -1)
            self.lr_std = torch.tensor(lr_normalization['std']).reshape(1, -1)
        super().__init__(*args, **kwargs)
        self.instances = [(root, instance) for root, instance in self.instances
                          if os.path.exists(os.path.join(root[self.lr_root_key], f'{instance}.npz'))]
        self.loads = [self.metadata.loc[instance, f'{self.latent_key}_tokens']
                      for _, instance in self.instances]

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        data = np.load(os.path.join(root[self.lr_root_key], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if coords.shape[0] > self.max_lr_tokens:
            raise ValueError(f'LR token count {coords.shape[0]} exceeds {self.max_lr_tokens}')
        if self.lr_normalization is not None:
            feats = (feats - self.lr_mean) / self.lr_std
        if self.lr_mode == 'null':
            feats = torch.zeros_like(feats)
        elif self.lr_mode == 'shuffled':
            seed = int(instance[:16], 16) % (2 ** 32)
            order = torch.from_numpy(np.random.default_rng(seed).permutation(len(feats)))
            feats = feats[order]
        pack['lr_coords'], pack['lr_feats'] = coords, feats
        return pack

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is not None:
            from ..utils.data_utils import load_balanced_group_indices
            idx_groups = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
            groups = [[batch[i] for i in group] for group in idx_groups]
        else:
            groups = [batch]
        base_groups = [[
            {k: v for k, v in sample.items() if k not in {'lr_coords', 'lr_feats'}}
            for sample in group
        ] for group in groups]
        packs = [SLat.collate_fn(group) for group in base_groups]
        for pack, group in zip(packs, groups):
            coords, feats, layout, start = [], [], [], 0
            for i, sample in enumerate(group):
                n = sample['lr_coords'].shape[0]
                coords.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), sample['lr_coords']], -1))
                feats.append(sample['lr_feats'])
                layout.append(slice(start, start + n))
                start += n
            lr = SparseTensor(coords=torch.cat(coords), feats=torch.cat(feats))
            lr._shape = torch.Size([len(group), feats[0].shape[-1]])
            lr.register_spatial_cache('layout', layout)
            pack['lr_cond'] = lr
            pack.pop('lr_coords', None)
            pack.pop('lr_feats', None)
        return packs if split_size is not None else packs[0]
