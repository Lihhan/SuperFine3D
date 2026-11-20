from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,time,json,traceback,inspect,textwrap
from pathlib import Path
root=Path(f'{_E["SF3D_ROOT"]}/superfine3d/refiner')
sys.path.insert(0,str(root/'decode_code'))
os.environ.update(DECODE_CONV_CPU_EXPANSION='1',DECODE_CPU_FINAL_STAGE='1',DECODE_HOST_ALL_STAGES='0',DECODE_SINGLE_BATCH_VIEW='0',DECODE_INPLACE_C2S_NORM='1',DECODE_STREAM_NEIGHBORS='1',DECODE_CPU_C2S_CONV2='1',DECODE_CPU_MESH_ASSEMBLY='1',DECODE_STREAM_MESH_QUADS='1',DECODE_CPU_EXPAND='1',DECODE_INPLACE_FINAL='1',DECODE_RETIRE_MESH_MAPS='1',DECODE_CPU_RESIDUAL='1',DECODE_TILE_CONV='1',DECODE_CHUNK_MESH_QUERY='1')
if os.environ.get('DECODE_FAST')=='1':   # opt-in exact: GPU-resident decode (no host staging / streaming); bit-identical mesh (sha256, eff_opt/dec), ~4x faster, ~2.6x peak memory
 os.environ.update({k:'0' for k in ('DECODE_CONV_CPU_EXPANSION','DECODE_CPU_FINAL_STAGE','DECODE_STREAM_NEIGHBORS','DECODE_CPU_C2S_CONV2','DECODE_CPU_MESH_ASSEMBLY','DECODE_STREAM_MESH_QUADS','DECODE_CPU_EXPAND','DECODE_CPU_RESIDUAL','DECODE_TILE_CONV')})
import torch
torch.set_grad_enabled(False)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEM_FRACTION','.70')))
torch.backends.cuda.matmul.allow_tf32=True
import o_voxel.convert.flexible_dual_grid as fdg
src=textwrap.dedent(inspect.getsource(fdg._init_hashmap))
src=src.replace('VOL = (grid_size[0] * grid_size[1] * grid_size[2]).item()','VOL = int(grid_size[0].item()) * int(grid_size[1].item()) * int(grid_size[2].item())')
ns=dict(fdg._init_hashmap.__globals__);exec(compile(src,'<verified-int64-volume>','exec'),ns)
fdg._init_hashmap=ns['_init_hashmap']
assert fdg._init_hashmap(torch.tensor([2048]*3,dtype=torch.int32),8,'cpu')[0].dtype==torch.uint64
import portable_pipeline as p
from decode_deferred_skip import prepare_install
install=prepare_install();old=p.install_decoder_lowmem_patch
def patch():
 old();install()
p.install_decoder_lowmem_patch=patch
key=sys.argv[1];res=int(sys.argv[2])
p.OUT=root/key
try:
 p.decode(res)
 os._exit(0)
except Exception:
 _tb=traceback.format_exc()
 if os.environ.get('DECODE_FAST')=='1' and 'out of memory' in _tb.lower():   # same mesh from the low-memory path: rerun this process with it
  print('DECODE_FAST: out of memory -> low-memory decode',flush=True);os.environ['DECODE_FAST']='0';os.execv(sys.executable,[sys.executable]+sys.orig_argv[1:])
 (p.OUT/'decode_error.txt').write_text(_tb)
 traceback.print_exc()
 os._exit(1)
