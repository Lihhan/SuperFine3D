"""Ultra clean-LR HR flow loss, with synchronized FP32 optimizer masters."""
import torch
from .m2 import M2LRGuidanceTrainer


class T2UltraTrainer(M2LRGuidanceTrainer):
    def __init__(self,*args,lr_dropout=0.,**kwargs):
        if lr_dropout != 0:
            raise ValueError('Ultra-reference T2 has no LR dropout during training')
        schedule = kwargs.get('t_schedule', {})
        if schedule != {'name':'logitNormal','args':{'mean':0,'std':1}}:
            raise ValueError('T2 reference requires Ultra logit-normal timesteps')
        if kwargs.get('sigma_min') != 0:
            raise ValueError('T2 reference requires sigma_min=0')
        super().__init__(*args,lr_dropout=lr_dropout,**kwargs)

    def finetune_from(self,finetune_ckpt):
        super().finetune_from(finetune_ckpt)
        self.models['denoiser'].assert_trainable_contract()
        states = {name:m.state_dict() for name,m in self.models.items()}
        self._state_dicts_to_master_params(self.master_params,states)
        masters = self._master_params_to_state_dicts(self.master_params)
        for name,p in self.models['denoiser'].named_parameters():
            if p.requires_grad and not torch.equal(masters['denoiser'][name].float(),p.detach().float()):
                raise ValueError(f'T2 optimizer master mismatch: {name}')
        if self.is_master:
            for params in self.ema_params:
                self._state_dicts_to_master_params(params,states)
