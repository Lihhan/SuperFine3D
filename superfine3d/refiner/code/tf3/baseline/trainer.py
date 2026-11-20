"""Two-stage initialization; optimizer masters synchronized after checkpoint load."""
import json
from pathlib import Path
from trellis2.trainers.flow_matching.m3_mapping import M3MappingTrainer
from .loading import initialize


class TF2Trainer(M3MappingTrainer):
    def __init__(self,*args,hr_interface_checkpoint=None,hr_interface_report=None,
                 lr_dropout=0.,m0_adapter_ckpt=None,lr_warmstart_ckpt=None,**kwargs):
        if lr_dropout or m0_adapter_ckpt or lr_warmstart_ckpt:
            raise ValueError('TF2 uses no LR dropout or implicit old-adapter initialization')
        self.hr_interface_checkpoint=hr_interface_checkpoint
        self.hr_interface_report=hr_interface_report
        super().__init__(*args,lr_dropout=0.,m0_adapter_ckpt=None,lr_warmstart_ckpt=None,**kwargs)

    def finetune_from(self,finetune_ckpt):
        if set(finetune_ckpt)!={'denoiser'}: raise ValueError('Expected official denoiser only')
        audit=initialize(self.models['denoiser'],finetune_ckpt['denoiser'],
                         self.hr_interface_checkpoint,self.hr_interface_report)
        states={n:m.state_dict() for n,m in self.models.items()}
        self._state_dicts_to_master_params(self.master_params,states)
        masters=self._master_params_to_state_dicts(self.master_params)
        import torch
        for n,p in self.models['denoiser'].named_parameters():
            if p.requires_grad and not torch.equal(masters['denoiser'][n].float(),p.detach().float()):
                raise ValueError('FP32 master mismatch: '+n)
        if self.step!=0 or self.optimizer.state: raise ValueError('Expected fresh optimizer')
        if self.is_master:
            for ema in self.ema_params: self._state_dicts_to_master_params(ema,states)
            Path(self.output_dir,'initialization_audit.json').write_text(json.dumps(audit,indent=2))
