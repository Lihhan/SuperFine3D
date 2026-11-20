from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import json,statistics
from pathlib import Path
R=Path(f'{_E["SF3D_RUNS"]}/R3_geometry_v2')
assert json.load(open(R/'preflight.json'))['passed']
records=json.load(open(R/'calibration/refinement_records.json'))
cfg=json.load(open(R/'train.json'));weights=cfg['trainer']['args']['refinement_weights'];audit={}
for key in ['edge','plane']:
    rows=[r['gradients'][key] for r in records if r['gradients'].get(key,{}).get('norm',0)>1e-12]
    assert len(rows)>=3,(key,len(rows))
    weights[key]=statistics.median(.025*r['flow_norm']/r['norm'] for r in rows)
    audit[key]=dict(samples=len(rows),weight=weights[key],median_cosine=statistics.median(r['cosine'] for r in rows))
(R/'train.json').write_text(json.dumps(cfg,indent=2))
(R/'calibration_summary.json').write_text(json.dumps(dict(terms=audit,target_gradient_ratio_per_term=.025,
    existing_R2_weights_preserved=True,ramp_steps=100,geometry_every=4,max_t=.5),indent=2))
print('RELEASED',json.dumps(audit),flush=True)
