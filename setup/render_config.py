"""Expand ${VARIABLES} (setup/env.sh) in a JSON config and write the concrete copy to $SF3D_RUNS/configs/; prints its path.
usage: render_config.py CONFIG.json [key.path=value ...]   (overrides, e.g. trainer.args.max_steps=10)"""
import os, sys, json
src, sets = sys.argv[1], sys.argv[2:]
cfg = json.loads(os.path.expandvars(open(src).read()))
left = [k for k in json.dumps(cfg).split('"') if '${' in k]
assert not left, f'unset variables in {src}: {left}'
for kv in sets:
    k, v = kv.split('=', 1); d = cfg; ks = k.split('.')
    for x in ks[:-1]: d = d[x]
    d[ks[-1]] = json.loads(v)
out = os.path.join(os.environ['SF3D_RUNS'], 'configs', os.path.basename(src)); os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(cfg, open(out, 'w'), indent=2); print(out)
