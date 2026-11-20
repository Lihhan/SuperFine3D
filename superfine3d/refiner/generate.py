from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,json,time,gc,hashlib,inspect,textwrap,types
from pathlib import Path
R=Path(f'{_E["SF3D_ROOT"]}/superfine3d/refiner')
sys.path[:0]=[str(R/'code'),f'{_E["SF3D_ROOT"]}/superfine3d/refiner/decode_code']
import torch,numpy as np
from PIL import Image
import trellis2.models as models
import trellis2.modules.sparse.attention as attention
models.__path__.insert(0,str(R/'code/custom_models'))
attention.__path__.insert(0,str(R/'code/custom_attention'))
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
key,stage=sys.argv[1:3]
D=R/key;D.mkdir(exist_ok=True)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32=True
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEM_FRACTION','.50')))
CACHE=Path(f'{_E["SF3D_TRELLIS2_WEIGHTS"]}')
def write(name,value):
 p=D/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)
def status(s,**kw):
 write('status.json',dict(stage=s,time=time.time(),**kw));print(s,kw,flush=True)
def clean():gc.collect();torch.cuda.empty_cache()
def save(z,res):
 np.savez_compressed(D/f'latent_{res}.npz',coords=z.coords.cpu().numpy(),feats=z.feats.float().cpu().numpy())
def load(res):
 with np.load(D/f'latent_{res}.npz') as f:return SparseTensor(coords=torch.from_numpy(f['coords'].copy()).cuda(),feats=torch.from_numpy(f['feats'].copy()).cuda())
def pipe(names):
 import trellis2.pipelines.trellis2_image_to_3d as pm
 class Dummy:
  def __getattr__(self,n):return lambda **kw: (lambda im:im.convert('RGBA'))
 pm.rembg=Dummy()
 Trellis2ImageTo3DPipeline.model_names_to_load=names
 p=Trellis2ImageTo3DPipeline.from_pretrained(str(CACHE));p.low_vram=True;p.to('cuda');return p
def bounded(model):
 import lr_chunk
 lr_chunk.install(model)
 for block in model.blocks:
  dense=block.mlp.mlp
  def forward(self,x,_dense=dense):
   output=torch.empty_like(x.feats)
   for a in range(0,len(output),16384):
    z=type(x)(coords=x.coords[a:a+16384],feats=x.feats[a:a+16384]);output[a:a+16384]=_dense(z).feats
   return x.replace(output)
  block.mlp.forward=types.MethodType(forward,block.mlp)
 for adapter in model.lr_gated_residual:
  original=adapter.forward
  def forward(self,*args,_f=original):
   return torch.cat([_f(*(v[a:a+16384] for v in args)) for a in range(0,len(args[0]),16384)])
  adapter.forward=types.MethodType(forward,adapter)
def frontend():
 if (D/'latent_512.npz').exists() and (D/'image_cond.pt').exists():return
 status('LOADING_FRONTEND')
 p=pipe(['sparse_structure_flow_model','sparse_structure_decoder','shape_slat_flow_model_512'])
 im=p.preprocess_image(Image.open(D/'input.png').convert('RGBA'))
 im.save(D/'conditioning.png')
 status('IMAGE_CONDITIONING');c=p.get_cond([im],512);cond=p.get_cond([im],1024)
 torch.save({k:v.cpu() for k,v in cond.items()},D/'image_cond.pt')
 torch.manual_seed(730004);status('GENERATING_512_SUPPORT')
 coords=p.sample_sparse_structure(c,resolution=32,num_samples=1,sampler_params=dict(p.sparse_structure_sampler_params))
 torch.manual_seed(731004);status('GENERATING_512_SHAPE')
 z=p.sample_shape_slat(c,p.models['shape_slat_flow_model_512'],coords);save(z,512)
 write('frontend.json',dict(gt_used=False,structure_seed=730004,shape_seed=731004,tokens=len(z.coords),input_sha256=hashlib.sha256((D/'input.png').read_bytes()).hexdigest()))
 status('FRONTEND_COMPLETE',tokens=len(z.coords))
def generate():
 from ablation_model import AblationFlow
 from session import StructureSession
 from safetensors.torch import load_file
 from portable_pipeline import setup_decoder
 from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
 import trellis2.pipelines.samplers.flow_euler as em
 status('LOADING_REFINEMENT')
 CK=Path(_E['SF3D_CKPT'])/'refiner';cfg=json.loads((CK/'train.json').read_text());model=AblationFlow(**cfg['models']['denoiser']['args'])
 base=load_file(str(CACHE/'ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors'))
 result=model.load_state_dict(base,strict=False)
 assert not result.unexpected_keys and all('lora_' in n or n.startswith('lr_') for n in result.missing_keys),result
 del base
 ck=CK/'denoiser_step0002000.pt';state=torch.load(ck,map_location='cpu',weights_only=True)
 assert set(state)=={n for n,p in model.named_parameters() if p.requires_grad}
 # This checkpoint includes all HR attention LoRA and LR adapters; no old HR adapter is needed.
 model.load_state_dict(state,strict=False);del state
 bounded(model);model.eval();model.use_checkpoint=False
 HRONLY=os.environ.get('REFINER')=='hr_only'                       # ablation: HR-only refiner (tf6_2 HR adapter, trained without LR)
 if HRONLY:
  from tf6_2.loading import load_tf62
  del model;model,_hrcfg=load_tf62(Path(f'{_E["SF3D_ROOT"]}/superfine3d/refiner/code/hr/config.json'),adapter=Path(f'{_E["SF3D_CKPT"]}/stages/tf6_2_hr/denoiser_step0005000.pt'),device='cpu')
  assert model.tf2_stage=='hr_local' and not _hrcfg['models']['denoiser']['args']['use_lr']
  model.eval();model.use_checkpoint=False
  class HRSession(torch.nn.Module):
   def __init__(self,m,r):super().__init__();self.m=m;self.r=r;self.kvs={};self.calls={}
   def forward(self,x,t,c,**kw):return self.m(x,t,c,lr_enabled=False,hr_resolution=self.r,**kw)
   def close(self):pass
 p=pipe(['shape_slat_decoder']);decoder=p.models['shape_slat_decoder'];decoder.eval();decoder.requires_grad_(False);dtype=setup_decoder(decoder)
 norm=cfg['dataset']['args']['normalization'];ln=cfg['dataset']['args']['lr_normalization']
 mean=torch.tensor(norm['mean'],device='cuda')[None];std=torch.tensor(norm['std'],device='cuda')[None]
 lm=torch.tensor(ln['mean'],device='cuda')[None];ls=torch.tensor(ln['std'],device='cuda')[None]
 cond={k:v.cuda() for k,v in torch.load(D/'image_cond.pt',weights_only=True).items()}
 source=textwrap.dedent(inspect.getsource(FlowEulerSampler.sample)).replace('ret.pred_x_t.append(out.pred_x_prev)','pass').replace('ret.pred_x_0.append(out.pred_x_0)','pass')
 ns=dict(vars(em));exec(compile(source,'<no-history>','exec'),ns);FlowEulerSampler.sample=ns['sample']
 # Refinement (2048/4096) CFG defaults to 1: with the LR in both branches, cond-uncond is mostly noise and CFG 7.5
 # amplifies it into chaotic fine texture (pixalign_20261104 crop_diag + L17 check). Base 512/1024 keep the pipeline CFG.
 sp=dict(p.shape_slat_sampler_params,guidance_strength=float(os.environ.get('REFINE_CFG','1')))
 report=dict(refiner='hr_only_tf6_2_step5000' if HRONLY else 'lr_guided',hr_checkpoint_sha256=hashlib.sha256(Path(f'{_E["SF3D_CKPT"]}/stages/tf6_2_hr/denoiser_step0005000.pt').read_bytes()).hexdigest() if HRONLY else None,fused=os.environ.get('FUSED','1')=='1' and not HRONLY,gt_lr=False,gt_support=False,checkpoint_sha256=None if HRONLY else hashlib.sha256(ck.read_bytes()).hexdigest(),sampler=sp,stages={})
 prev=load(1024)
 for res,offset in [(2048,3000),(4096,4000)]:
  if (D/f'latent_{res}.npz').exists():prev=load(res);continue
  status('CONSTRUCTING_SUPPORT',resolution=res);clean();start=time.time()
  decoder.set_resolution(res//2);decoder.cuda();decoder.low_vram=True
  fine=decoder.upsample(prev.replace(prev.feats.to(dtype)),upsample_times=4)
  decoder.cpu();decoder.low_vram=False
  coords=torch.cat((fine[:,:1],((fine[:,1:]+.5)/(res//2)*(res//16)).int()),1).unique(dim=0)
  del fine;clean();model.cuda();model.set_resolution_pair(res//16,res//32)
  torch.manual_seed(730004+offset)
  noise=SparseTensor(feats=torch.randn(len(coords),model.in_channels,device='cuda'),coords=coords)
  lr=prev.replace((prev.feats.float()-lm)/ls)
  status('REFINEMENT_CACHE',resolution=res,tokens=len(coords))
  session=HRSession(model,res//16) if HRONLY else StructureSession(model,lr,coords,[cond['cond'],cond['neg_cond']],res//16,variant='K2_hr_allcells',runtime_variant=os.environ.get('LR_KV_POLICY','V5_cpu'),verify_kernel=False)   # LR_KV_POLICY=V5_none (opt-in): no LR K/V cache (single-step SDEdit never reuses it; same values)
  if os.environ.get('FUSED','1')=='1' and not HRONLY:sys.path.append(f'{_E["SF3D_ROOT"]}/superfine3d/detail/bench');import ours_fast;report_fused=ours_fast.install(model,session)   # fused DiT kernels (FUSED=0: original path)
  status('REFINEMENT_SAMPLING',resolution=res,tokens=len(coords))
  try:result=p.shape_slat_sampler.sample(session,noise,**cond,**sp,verbose=True,tqdm_desc=f'{key} {res}').samples
  finally:session.close()
  prev=result.replace(result.feats*std+mean);save(prev,res)
  report['stages'][str(res)]={'tokens':len(coords),'seconds':time.time()-start};write('generation.json',report)
  del result,session,lr,noise,coords
  if hasattr(model,'_anchor_cache'):model._anchor_cache.entry=None
  model.cpu();clean();status('LATENT_COMPLETE',resolution=res)
 status('GENERATION_COMPLETE')
try:
 if stage=='frontend':frontend()
 elif stage=='generate':generate()
 else:raise ValueError(stage)
except Exception:
 import traceback
 write('error_'+stage+'.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
