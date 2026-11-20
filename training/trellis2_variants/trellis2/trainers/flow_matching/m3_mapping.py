"""Same HR flow loss for matched M0, frozen-M2 and joint-M2 continuations."""
import hashlib
import json
from pathlib import Path
import torch
from .m2 import M2LRGuidanceTrainer


class M3MappingTrainer(M2LRGuidanceTrainer):
    def __init__(self, *args, lr_warmstart_ckpt=None, **kwargs):
        self.lr_warmstart_ckpt = lr_warmstart_ckpt
        super().__init__(*args, **kwargs)

    def finetune_from(self, finetune_ckpt):
        # Parent loads official weights, then M0 LoRA. BasicTrainer has already
        # constructed master weights, so synchronize AFTER loading all adapters.
        super().finetune_from(finetune_ckpt)
        model = self.models['denoiser']
        if model.experiment_arm != 'm0':
            if self.lr_warmstart_ckpt:
                raise ValueError('M3 first comparison requires fresh LR adapters')
        elif self.lr_warmstart_ckpt:
            raise ValueError('M0 control must not load an LR warm-start')
        model.assert_trainable_contract()
        states = {name: module.state_dict() for name,module in self.models.items()}
        self._state_dicts_to_master_params(self.master_params, states)
        master_states = self._master_params_to_state_dicts(self.master_params)
        for name, param in model.named_parameters():
            if param.requires_grad and not torch.equal(master_states['denoiser'][name].float(), param.detach().float()):
                raise ValueError(f'FP32 optimizer master warm-start mismatch: {name}')
        if self.is_master:
            for params in self.ema_params:
                self._state_dicts_to_master_params(params, states)
            print('CONTINUATION_TRAINABLE_CONTRACT', model.assert_trainable_contract(), flush=True)

    # Inherit the exact M2 flow-MSE implementation, including the same random
    # timestep/noise/image dropout/LR dropout draws. M0 consumes the dropout draw
    # too, but its model skips LR computation. This preserves RNG consumption.

    def training_losses(self, x_0, cond=None, **kwargs):
        trace = None
        if self.step < 50:
            def digest(tensor):
                return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
            trace = {'step': self.step+1, 'hr_coords': digest(x_0.coords),
                     'hr_features': digest(x_0.feats), 'image': digest(cond),
                     'cuda_rng_before': digest(torch.cuda.get_rng_state(x_0.device))}
        result = super().training_losses(x_0, cond, **kwargs)
        if trace is not None:
            trace['cuda_rng_after'] = digest(torch.cuda.get_rng_state(x_0.device))
            with (Path(self.output_dir)/'matched_input_trace.jsonl').open('a') as f:
                f.write(json.dumps(trace)+'\n')
        return result
