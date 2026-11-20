from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import json
from pathlib import Path
R=Path(f'{_E["SF3D_RUNS"]}/R3_geometry_v2');R.mkdir(exist_ok=True)
cfg=json.load(open(f'{_E["SF3D_ROOT"]}/training/refiner/configs/stage4_R2.json'))
t=cfg['trainer']['args'];t.update(max_steps=2000,save_steps=[500,2000],gradient_sanity_steps=10,refinement_arm='R3_geometry_v2')
(R/'train.json').write_text(json.dumps(cfg,indent=2))
cal=json.loads(json.dumps(cfg));t=cal['trainer']['args']
t.update(max_steps=32,save_steps=[32],refinement_calibration=True,gradient_sanity_steps=0,i_log=1,i_print=1)
t['optimizer']['args']['lr']=0.
(R/'calibration.json').write_text(json.dumps(cal,indent=2))
(R/'plan.json').write_text(json.dumps(dict(name='R3_geometry_v2',display_name='R3 (new geometry)',
    source=f'{_E["SF3D_CKPT"]}/stages/R2/denoiser_step0003000.pt',
    steps=2000,effective_batch=2,hr_lr=1e-5,lr_lr=2e-5,training_assets=10000,
    changes=['coordinate-matched continuous edge BCE','exact teacher vertex normal displacement'],
    safeguards=['unknown teacher coordinates excluded','crease normals excluded','no tangential smoothing','full decode context'],
    eval='same 10 assets, generated LR, predicted support, own 512->1024->2048->4096; toy first',
    inference_architecture_unchanged=True,status='preflight'),indent=2))
print('R3_SETUP',R)
