"""Controlled M0/M2 continuation with configurable existing HR attention LoRA.

No attention, routing, conditioning or loss changes are introduced here.
"""
import json
from pathlib import Path
import torch
from .m2_flow import ElasticM2SLatFlowModel


LR_PREFIXES = ('lr_input_delta.', 'lr_qkv_delta.', 'lr_out_delta.', 'lr_null_embedding')


def is_hr_attention_lora(name):
    return (name.startswith('blocks.') and '.self_attn.' in name
            and ('.to_qkv.' in name or '.to_out.' in name) and 'lora_' in name)


class ElasticM2JointSLatFlowModel(ElasticM2SLatFlowModel):
    def __init__(self, *args, experiment_arm, hr_lora_last_n=None, **kwargs):
        if experiment_arm not in ('m0', 'm2_frozen', 'm2_joint'):
            raise ValueError('Unknown continuation experiment arm')
        super().__init__(*args, **kwargs)
        self.experiment_arm = experiment_arm
        if hr_lora_last_n is None:
            hr_lora_last_n = self.num_blocks
        if not 1 <= hr_lora_last_n <= self.num_blocks:
            raise ValueError('Invalid HR LoRA layer range')
        self.hr_lora_last_n = int(hr_lora_last_n)
        for name, p in self.named_parameters():
            lr_train = experiment_arm != 'm0' and name.startswith(LR_PREFIXES)
            hr_train = (experiment_arm != 'm2_frozen' and is_hr_attention_lora(name)
                        and int(name.split('.')[1]) >= self.num_blocks-self.hr_lora_last_n)
            p.requires_grad_(lr_train or hr_train)
        self.assert_trainable_contract()

    def assert_trainable_contract(self):
        names = [n for n,p in self.named_parameters() if p.requires_grad]
        if not names:
            raise ValueError('No trainable parameters')
        for name in names:
            if not name.startswith(LR_PREFIXES) and not is_hr_attention_lora(name):
                raise ValueError(f'Unexpected trainable parameter: {name}')
        hr = [n for n in names if is_hr_attention_lora(n)]
        lr = [n for n in names if n.startswith(LR_PREFIXES)]
        if bool(hr) != (self.experiment_arm != 'm2_frozen'):
            raise ValueError('HR trainable contract failed')
        if bool(lr) != (self.experiment_arm != 'm0'):
            raise ValueError('LR trainable contract failed')
        if hr and len(hr) != 4*self.hr_lora_last_n:
            raise ValueError('Each selected HR layer must have QKV/out A/B tensors')
        return {'arm': self.experiment_arm, 'hr_tensors': len(hr), 'lr_tensors': len(lr),
                'hr_parameters': sum(p.numel() for n,p in self.named_parameters() if n in hr),
                'lr_parameters': sum(p.numel() for n,p in self.named_parameters() if n in lr)}

    def load_lr_warmstart(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        expected = {n for n,p in self.named_parameters() if n.startswith(LR_PREFIXES)}
        if set(state) != expected:
            raise ValueError('Warm-start checkpoint must contain exactly the original M2 LR adapter')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError('Unexpected LR checkpoint keys')

    def forward(self, *args, lr_enabled=None, **kwargs):
        enabled = self.experiment_arm != 'm0' if lr_enabled is None else bool(lr_enabled)
        if self.experiment_arm == 'm0' and enabled:
            raise ValueError('M0 control cannot enable LR')
        return super().forward(*args, lr_enabled=enabled, **kwargs)


def load_continuation_model(config_path, adapter_path, device='cuda'):
    cfg = json.loads(Path(config_path).read_text())
    model = ElasticM2JointSLatFlowModel(**cfg['models']['denoiser']['args'])
    args = cfg['trainer']['args']
    state = torch.load(args['finetune_ckpt']['denoiser'], map_location='cpu', weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any('lora_' not in n and not n.startswith('lr_')
                                     for n in result.missing_keys):
        raise ValueError('Official checkpoint contract mismatch')
    del state
    model.load_m0_adapter(args['m0_adapter_ckpt'])
    if model.experiment_arm != 'm0':
        model.load_lr_warmstart(args['lr_warmstart_ckpt'])
    state = torch.load(adapter_path, map_location='cpu', weights_only=True)
    expected = {n for n,p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('Continuation checkpoint trainable key mismatch')
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError('Unexpected continuation checkpoint keys')
    return model.to(device).eval(), cfg['dataset']['args']['lr_normalization']
