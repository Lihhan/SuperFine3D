"""SC2-B continuation control. Objective arms are gated on decoder audit."""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys, importlib.util, json, hashlib
from pathlib import Path
import torch
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/training/refiner/stages')
spec=importlib.util.spec_from_file_location('sc2_training',f'{_E["SF3D_ROOT"]}/training/refiner/stages/sc2_train.py')
sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
SOURCE=Path(f'{_E["SF3D_CKPT"]}/stages/sc2_B_high/denoiser_step0003000.pt')

def initialize(self,ckpt):
    sc.TF6Trainer.finetune_from(self,ckpt)
    model=self.models['denoiser'];state=torch.load(SOURCE,map_location='cpu',weights_only=True)
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    assert set(state)==expected,(set(state)^expected)
    model.load_state_dict(state,strict=False)
    states={n:m.state_dict() for n,m in self.models.items()}
    self._state_dicts_to_master_params(self.master_params,states)
    for ema in self.ema_params:self._state_dicts_to_master_params(ema,states)
    # Reset gradient diagnostic origin after loading the actual experimental start.
    if hasattr(self,'_initial'):
        self._initial={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    report=dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        contract=model.assert_trainable_contract(),optimizer_reset=True,
        groups=[dict(name=g['group_name'],lr=g['lr']) for g in self.optimizer.param_groups])
    Path(self.output_dir,'refinement_initialization.json').write_text(json.dumps(report,indent=2))
    print('REFINEMENT_INIT',json.dumps(report),flush=True)

sc.AblationTrainer.finetune_from=initialize
original_init=sc.AblationTrainer.__init__
original_losses=sc.AblationTrainer.training_losses

def trainer_init(self,*args,refinement_arm='R0',refinement_weights=None,refinement_calibration=False,**kwargs):
    self.refinement_arm=refinement_arm;self.refinement_weights=refinement_weights or {}
    self.refinement_calibration=refinement_calibration;self.refinement_records=[]
    original_init(self,*args,**kwargs)
    if refinement_arm!='R0':
        from refine_surface import RefinementSurface
        cfg=json.loads(__import__('os').path.expandvars(open(f'{_E["SF3D_ROOT"]}/training/refiner/configs/tf10_B_retry.json').read()))['trainer']['args']['structure']
        self.refinement_surface=RefinementSurface(**{k:cfg[k] for k in ['decoder_path','hr_statistics','lr_statistics']},max_tokens=16384)

def training_losses(self,x_0,cond=None,**kwargs):
    capture={}
    active=self.refinement_arm!='R0' and (self.refinement_calibration or self.step%4==0)
    if active:
        def hook(module,inputs,output):capture.update(xt=inputs[0],t=inputs[1]/1000,pred=output)
        handle=self.training_models['denoiser'].register_forward_hook(hook)
    try:terms,extra=original_losses(self,x_0,cond,**kwargs)
    finally:
        if active:handle.remove()
    terms.refinement_active=0.
    if not active or float(capture['t'][0])>.5:return terms,extra
    from tf3.geometry import clean_from_velocity
    from tf10.geometry import NoValidPatch
    clean=clean_from_velocity(capture['xt'].feats.float(),capture['pred'].feats.float(),capture['t'][0],self.sigma_min)
    import time
    start=time.perf_counter()
    try:losses,detail=self.refinement_surface(x_0.replace(clean),x_0,kwargs['lr_cond'],seed=910000+self.step*2+self._micro_count%2)
    except NoValidPatch as error:
        terms.refinement_skipped=1.
        with open(Path(self.output_dir,'refinement_skips.jsonl'),'a') as stream:
            stream.write(json.dumps(dict(step=self.step,reason=str(error),tokens=len(x_0.coords)))+'\n')
        return terms,extra
    terms.refinement_active=1.;terms.refinement=losses
    record=dict(step=self.step,seconds=time.perf_counter()-start,detail=detail,gradients={})
    if self.refinement_calibration:
        output=capture['pred'].feats
        gf=torch.autograd.grad(terms.mse,output,retain_graph=True)[0].float()
        for key,value in losses.items():
            g=torch.autograd.grad(value,output,retain_graph=True)[0].float()
            assert torch.isfinite(g).all()
            record['gradients'][key]=dict(norm=float(g.norm()),flow_norm=float(gf.norm()),
                cosine=float((gf*g).sum()/(gf.norm()*g.norm()).clamp_min(1e-20)))
    else:
        keys=['keep'] if self.refinement_arm=='R1' else ['keep','detail']
        assert all(k in self.refinement_weights for k in keys)
        terms.loss=terms.loss+min(1.,(self.step+1)/100)*sum(self.refinement_weights[k]*losses[k] for k in keys)
    self.refinement_records.append(record)
    Path(self.output_dir,'refinement_records.json').write_text(json.dumps(self.refinement_records,indent=2))
    return terms,extra

sc.AblationTrainer.__init__=trainer_init
sc.AblationTrainer.training_losses=training_losses
if __name__=='__main__':sc.main()
