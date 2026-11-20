"""Full-latent flow supervision; synthetic LR perturbations never change HR targets."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from easydict import EasyDict as edict
from trellis2.modules import sparse as sp
from tf3.baseline.trainer import TF2Trainer
from tf5.conditioning import perturb_local


class TF6Trainer(TF2Trainer):
    def __init__(self,*args,lr_perturbation=None,**kwargs):
        self.lr_perturbation=lr_perturbation or dict(probability=.5,radius=2,delete_fraction=.5,noise_std=.1)
        super().__init__(*args,**kwargs)
        model=self.models['denoiser']
        self._initial={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        self._grad={n:0. for n in self._initial}
        self._first_grad={}
        self._gate_written=False;self._micro_count=0
        # Resume checkpoints after the initial 50 steps have already passed the
        # gradient/update gate. Do not rerun that one-time check on continuation.
        if self.step >= 50:
            self._gate_written=True
            self._initial.clear()
            self._grad.clear()
        for n,p in model.named_parameters():
            if p.requires_grad:
                def record(g,name=n):
                    if not torch.isfinite(g).all(): raise RuntimeError('Nonfinite TF6 gradient: '+name)
                    if self.step<50:
                        value=float(g.detach().abs().max())
                        self._grad[name]=max(self._grad[name],value)
                        if value>0: self._first_grad.setdefault(name,self.step+1)
                    return g
                p.register_hook(record)

    def training_losses(self,x_0,cond=None,**kwargs):
        model=self.models['denoiser']
        if self.step == 49 and not self._gate_written:
            updates={n:float((p.detach().float().cpu()-self._initial[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad}
            passed=all(v>0 for v in updates.values()) and all(v>0 for v in self._grad.values())
            Path(self.output_dir,'tf6_gradient_gate.json').write_text(json.dumps(dict(
                passed=passed,completed_steps=self.step,parameter_updates=updates,
                max_gradients=self._grad,first_nonzero_gradient_step=self._first_grad),indent=2))
            if not passed: raise RuntimeError('TF6 gradient/update gate failed; inspect per-parameter report')
            self._gate_written=True;self._initial.clear()
        if model.tf2_stage == 'hr_local':
            return TF2Trainer.training_losses(self,x_0,cond,**kwargs)
        lr=kwargs.get('lr_cond')
        if lr is None: raise ValueError('Missing paired LR training condition')
        # Independent RNG avoids coupling LR corruption to flow-noise/timestep draws.
        generator=torch.Generator(device=lr.device)
        generator.manual_seed(530000+int(self.step)*100+int(self._micro_count%2))
        self._micro_count+=1
        coords,feats,stats=perturb_local(lr.coords,lr.feats,x_0.shape[0],generator=generator,**self.lr_perturbation)
        changed=sp.SparseTensor(feats=feats,coords=coords,shape=lr.shape)
        kwargs=dict(kwargs,lr_cond=changed)
        noise=x_0.replace(torch.randn_like(x_0.feats))
        t=self.sample_t(x_0.shape[0]).to(x_0.device).float()
        xt=self.diffuse(x_0,t,noise=noise)
        image_cond=self.get_cond(cond,**kwargs)
        pred=self.training_models['denoiser'](xt,t*1000,image_cond,**kwargs)
        target=self.get_v(x_0,noise,t)
        loss=F.mse_loss(pred.feats.float(),target.feats.float())
        terms=edict(loss=loss,mse=loss,lr_perturbed_fraction=stats['perturbed_samples']/x_0.shape[0],
                    lr_removed_fraction=stats['removed_tokens']/len(lr.coords))
        for i in range(10):
            ids=torch.where((t>=i/10)&(t<(i+1)/10))[0].tolist()
            if ids:
                terms[f'bin_{i}']={'mse':float(torch.stack([F.mse_loss(pred.feats[x_0.layout[j]].float(),target.feats[x_0.layout[j]].float()).detach() for j in ids]).mean())}
        return terms,{}
