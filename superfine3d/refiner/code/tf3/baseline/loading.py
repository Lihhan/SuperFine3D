import hashlib
import json
from pathlib import Path
import torch
from .model import ElasticTF2SLatFlowModel


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def initialize(model,official,hr_checkpoint=None,hr_report=None):
    state=torch.load(official,map_location='cpu',weights_only=True)
    if any('lora_' in n or n.startswith('lr_') for n in state):
        raise ValueError('Official base must not contain adapters')
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys or any('lora_' not in n and not n.startswith('lr_') for n in result.missing_keys):
        raise ValueError(f'Base schema mismatch: {result}')
    if model.tf2_stage=='lr_match':
        if not hr_checkpoint or not hr_report:
            raise ValueError('LR matching requires a qualified HR checkpoint and quality report')
        report=json.loads(Path(hr_report).read_text())
        if (report.get('approved_for_lr_matching') is not True or
            report.get('cell_size') != model.cell_size or
            report.get('checkpoint_sha256') != sha256(hr_checkpoint) or
            report.get('official_sha256') != sha256(official)):
            raise ValueError('HR qualification report does not match this interface/base')
        model.load_m0_adapter(hr_checkpoint)
    elif hr_checkpoint or hr_report:
        raise ValueError('HR stage starts from official base and fresh HR adapters')
    return {'official':str(official),'hr_checkpoint':hr_checkpoint,
            'contract':model.assert_trainable_contract()}


def load_tf2(config,adapter=None,device='cpu'):
    cfg=json.loads(Path(config).read_text())
    model=ElasticTF2SLatFlowModel(**cfg['models']['denoiser']['args'])
    train=cfg['trainer']['args']
    initialize(model,train['finetune_ckpt']['denoiser'],
               train.get('hr_interface_checkpoint'),train.get('hr_interface_report'))
    if adapter:
        state=torch.load(adapter,map_location='cpu',weights_only=True)
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        if set(state)!=expected: raise ValueError('TF2 adapter stage/schema mismatch')
        model.load_state_dict(state,strict=False)
    return model.to(device).eval(),cfg
