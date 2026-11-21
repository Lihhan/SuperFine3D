"""SuperFine3D end-to-end pipeline (image -> 1024^3 / 2048^3 / 4096^3 mesh) with the Table-1 configuration.
512^3 start, chosen with SF3D_START:
  pixal3d  (default, Table 1) Pixal3D native 512 output, placed in the TRELLIS.2 reference frame and re-fitted to the input
  trellis2 the TRELLIS.2 512 shape of the reference frame itself (with its fitted camera); Pixal3D is not run
Everything after the start (injection, encodes, refinement, decoding, export) is the same for both starts.
Original header: end-to-end benchmark of the Table-1 configuration (Pixal3D-512 start, T1 injection settings), derived from
bench_v3_body.py. Pixal3D 512 runs in its own environment as a concurrent subprocess (started at T0, overlapping the
TRELLIS.2 reference frame and its camera fit) together with its own camera fit (bench/pixal_chain.sh = start.sh pixal512_M +
pixal512_M_cam); the placed start's camera is fitted and refined as start.sh cam_M; injection = arms/T1.sh (route,
evidence gate, per-level K, estimate computed once, EST_CACHE); SDEdit t0 0.15 at 2048 and 0.05 at 4096.
Original header (bench_v3_body.py): End-to-end benchmark of the image-guided pipeline in ONE process (protocol of the paper's Table 2: image -> exported mesh,
includes model loads, transfers and IO; excludes interpreter start and imports; synchronized stage boundaries; peak
allocated GPU memory over the whole run; a foreign process on the GPU invalidates the trial).
Stages: frontend (preprocess, 512/1024 image conditions, 512 structure + shape) -> decode 512 -> camera (silhouette fit +
appearance refinement) -> input crop -> inject@512 -> encode 1024 -> [1024: decode + export]
 -> inject@1024 -> encode 2048 -> refine 2048 (partial noise t0) -> decode 2048 -> [2048: final inject + cap + export]
 -> inject@2048 -> encode 4096 -> refine 4096 -> decode 4096 -> inject@4096 -> cap -> export.
Optimisation switches (env, default off = round 0, faithful to pipeline_v3):
  OPT_EST_CACHE=1  estimate detail normals once and reuse at every level (grey route: the estimate does not depend on the mesh)
  OPT_NSLAB=k      slabs for the 4096 encode (pipeline default 3)
  OPT_FIT_FAST=1   camera: single refinement start from the silhouette fit + mirrors only (no extra grid starts)
  OPT_DEC_RESIDENT=1  keep decoder/refiner resident between stages
  OPT_CAMTHREADS=n exact camera speed-up, in process: convex-hull bounding boxes, threaded grid/Powell starts (own CUDA stream
                   and rasterizer context per thread), fewer host-device round trips; same values as the sequential scripts
  OPT_PRELOAD=1    load the detail estimator and the refiner weights in a background thread while the camera is searched
  OPT_ROUTE_FAST=1 exact: boundary-attention routing keys computed once per token (fastroute.py)
  OPT_DEC_FAST=1   exact: shape decoder built once and reused; decoded mesh kept in memory, no PLY round trip (decfast.py)
  OPT_VOX_MIN=1024 parallel voxeliser also for the 1024 encode (bitwise equal)
  OPT_EXPORT_LEAN=1 export without the T-junction log statistics (same mesh)
  OPT_DEC_GPU=1    decoder fully on the GPU instead of the CPU-offload switches of decode.py (bitwise-equal meshes, more memory)
  OPT_CAMPOOL=n    camera Powell starts in n pre-started worker processes, combined by the scripts' own selection (campool.py)
  OPT_CAMSHARE=1   with CAMPOOL: grid searches computed once and shared; cached face normals; memoised repeated evaluations
  OPT_CAMLOCK=1    camera refinement in this process: 12 Powell threads rasterized together per step (one batched call)
usage: bench_t1.py INPUT_PNG TARGET OUT_JSON WORKDIR   (env SF3D_START=pixal3d|trellis2)"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,time,json,subprocess,shutil,runpy,types,gc,inspect,textwrap
T_START=time.time()
INP,TARGET,OUTJ,WD=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4];os.makedirs(WD,exist_ok=True)
N=f'{_E["SF3D_ROOT"]}/superfine3d/refiner';P=f'{_E["SF3D_ROOT"]}/superfine3d/detail'
sys.path[:0]=[f'{_E["SF3D_ROOT"]}/third_party/site-packages',f'{P}/whole',P,f'{_E["SF3D_ROOT"]}/superfine3d/encode',f'{N}/code',f'{N}/decode_code']+[f'{P}/bench']
if os.environ.get('OPT_DEC_GPU','0')!='1':   # OPT_DEC_GPU=1: decoder fully on the GPU (no CPU-offload switches); meshes bitwise equal (test_decflags.py)
    os.environ.update(DECODE_CONV_CPU_EXPANSION='1',DECODE_CPU_FINAL_STAGE='1',DECODE_HOST_ALL_STAGES='0',DECODE_SINGLE_BATCH_VIEW='0',DECODE_INPLACE_C2S_NORM='1',DECODE_STREAM_NEIGHBORS='1',DECODE_CPU_C2S_CONV2='1',DECODE_CPU_MESH_ASSEMBLY='1')
    dec_src=open(f'{N}/decode.py').read();_envline=[l for l in dec_src.split('\n') if l.startswith('os.environ.update(')][0];exec(_envline)
import numpy as np,torch
from PIL import Image
import trellis2.models as models
import trellis2.modules.sparse.attention as attention
models.__path__.insert(0,f'{N}/code/custom_models');attention.__path__.insert(0,f'{N}/code/custom_attention')
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
import inject_lib as L
from hfdetail import latent_io,cleanup,pipeline as hpipe,heightfield as hf,estimator as hest
latent_io.patch_int64()                                   # encoder patches first (they read the original sources)
import o_voxel.convert.flexible_dual_grid as _fdg;assert _fdg._init_hashmap(torch.tensor([2048]*3,dtype=torch.int32),8,'cpu')[0].dtype==torch.uint64   # same int64 volume patch decode.py applies
torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False   # as the production scripts: TF32 only in refinement (generate.py) and decoding (decode.py), see tf32()
class tf32:
    def __enter__(s):s.o=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=True
    def __exit__(s,*a):torch.backends.cuda.matmul.allow_tf32=s.o
OPT=dict(parvox=int(os.environ.get('OPT_PARVOX','0')),campar=os.environ.get('OPT_CAMPAR','0')=='1',est_cache=os.environ.get('OPT_EST_CACHE','0')=='1',nslab=int(os.environ.get('OPT_NSLAB','3')),fit_fast=os.environ.get('OPT_FIT_FAST','0')=='1',camthreads=int(os.environ.get('OPT_CAMTHREADS','0')),preload=os.environ.get('OPT_PRELOAD','0')=='1',route_fast=os.environ.get('OPT_ROUTE_FAST','0')=='1',dec_fast=os.environ.get('OPT_DEC_FAST','0')=='1',export_lean=os.environ.get('OPT_EXPORT_LEAN','0')=='1',campool=int(os.environ.get('OPT_CAMPOOL','0')),camshare=os.environ.get('OPT_CAMSHARE','0')=='1',camlock=os.environ.get('OPT_CAMLOCK','0')=='1',vox_min=int(os.environ.get('OPT_VOX_MIN','2048')))
OPT['camfast']=os.environ.get('OPT_CAMFAST','0')=='1'   # exact camera speed-ups (bench/camab: refine bitwise; fit = same arithmetic, see FASTNORM)
OPT['encpipe']=int(os.environ.get('OPT_ENCPIPE','0'))   # exact: one parvox pool of this many workers; 4096 slabs voxelised concurrently (bench/encpipe.py)
OPT['preload_split']=os.environ.get('OPT_PRELOAD_SPLIT','0')=='1'   # estimators loaded in the main thread while Pixal3D is awaited (no camera code runs then)
OPT['bilat']=os.environ.get('OPT_BILAT','')   # 'iters:sigma_deg:v_iters': decoder-jitter removal (hfdetail/denoise.py) as pipeline_v3_fast DEC/INJ/FINAL_BILAT
OPT['bilat_ec']=os.environ.get('OPT_BILAT_EC')=='1'
OPT['bilat_dec4096']=os.environ.get('OPT_BILAT_MODE')=='dec4096'   # opt-in: no filtering at 2048 (the 4096 encode gets the unfiltered 2048 mesh), = pipeline DEC4096_BILAT + FINAL_BILAT   # diagnostic: release the CUDA cache after each bilateral call (no numeric change)
OPT['noinit']=os.environ.get('OPT_NOINIT','0')=='1'     # exact: models built without random init, every parameter from the checkpoint (bench/noinit.py)
START=os.environ.get('SF3D_START','pixal3d').lower().replace('.','').replace('-','').replace('_','')   # 512^3 start: pixal3d (Table 1) | trellis2
assert START in ('pixal3d','trellis2'),f'SF3D_START must be pixal3d or trellis2, not {START!r}'
PIXAL=START=='pixal3d'
CAMFAST_REF=dict(CAMREF_THREADS='12',CAMREF_LOCKSTEP='1',CAMREF_FNCACHE='1',CAMREF_MEMO='1',CAMREF_GRIDBATCH='72',CAMREF_FASTNORM='1')
CACHE=f'{_E["SF3D_TRELLIS2_WEIGHTS"]}'
CK=f'{_E["SF3D_CKPT"]}/estimator/grey/ckpt_04000.pt'   # grey-route detail estimator
def kv(s):return dict(x.split('=',1) for x in s.split())
CAL=f"ROUTE=1 RGB_CKPT={_E['SF3D_CKPT']}/estimator/rgb/ckpt_last.pt COND=0 CKPT={CK} K=64 GAIN=1.55 FK=4 FG=1.2 MAXD=6 LAP=0.003 LAPH=0.05 EDGE=0.5 LEVELS=3 ALIGN=1 TTA=0 RGB_GAIN=1.5 RGB_FK=4"
X=' GAIN=1.0 RGB_GAIN=1.0 FG=1.0 SHRINK=8 SHRINK_R=2 K=32 ALIGN=0';XF=' K=16 SHRINK=16'   # arms/T1.sh
INJ=dict(MAXD='3');INJ.update(kv(CAL+X));INJ512=dict(INJ,K='64');INJ1=dict(INJ,K='64');FIN=dict(INJ);FIN.update(kv(XF))
OPT['malign']=os.environ.get('OPT_MALIGN','')
if OPT['malign']:INJ512['MALIGN']=OPT['malign']   # opt-in (metric round 9/10): move the start onto the image at the 512 level (= INJ512CFG MALIGN)   # = pipeline_v3_fast.py cfg construction (INJ512CFG / INJ1024 K=64)
T0S={2048:0.15,4096:0.05}
PIXAL_PY=f'{_E["SF3D_PIXAL3D_PYTHON"]}'
times={};marks=[];PEAK=[0.0];PEAKS={}
gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
SHARED=os.environ.get('GPU_SHARED','1')=='1';SEEN=set();LOADS={};NCPU=os.cpu_count()   # GPU_SHARED=1: other users' processes allowed (recorded, not rejected)
def _own(pid):
    while pid>1:
        if pid==os.getpid():return True
        try:pid=int(open(f'/proc/{pid}/stat').read().rsplit(')',1)[1].split()[1])
        except Exception:return False
    return False
def foreign():
    q=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True).stdout.split()
    f=[int(x) for x in q if x.strip().isdigit() and not _own(int(x))]   # own process tree (Pixal3D chain, camera workers) is not foreign
    if SHARED:SEEN.update(f);return []
    return f
def stage(name):
    class _S:
        def __enter__(s):torch.cuda.synchronize();s.t=time.time()
        def __exit__(s,*a):
            torch.cuda.synchronize();times[name]=round(time.time()-s.t,2);f=foreign()
            pk=torch.cuda.max_memory_allocated()/2**30;PEAK[0]=max(PEAK[0],pk);PEAKS[name]=round(pk,2);torch.cuda.reset_peak_memory_stats()
            if f:marks.append(dict(stage=name,foreign=f))
            l1=os.getloadavg()[0];LOADS[name]=round(l1,1)
            if l1>NCPU:marks.append(dict(stage=name,cpu_load1=l1))      # CPU oversubscribed: CPU-threaded stages are not representative
            print(f'[{name}] {times[name]:.1f}s  stage peak {pk:.2f} GiB',flush=True)
    return _S()
assert os.environ.get('BENCH_DEBUG')=='1' or not foreign(),'GPU not idle'
if OPT['parvox']:                     # exact parallel voxelisation for the 2048/4096 encodes (hfdetail/parvox.py)
    from hfdetail import parvox
    import encpipe
    import pairs as _pairs,o_voxel as _ov
    _orig_vox=_ov.convert.mesh_to_flexible_dual_grid
    class _Conv:
        def __getattr__(s,n):return getattr(_ov.convert,n)
        @staticmethod
        def mesh_to_flexible_dual_grid(vertices,faces,grid_size,**kw):
            if OPT['encpipe'] and encpipe.PRE:return encpipe.PRE.pop(0).get()                # pre-submitted slab (same tasks)
            if grid_size>=OPT['vox_min'] and OPT['encpipe']:return parvox.voxelize_async(vertices,faces,grid_size,OPT['parvox'],OPT['encpipe']).get()
            if grid_size>=OPT['vox_min']:return parvox.voxelize(vertices,faces,grid_size,OPT['parvox'])   # OPT_VOX_MIN=1024: also the 1024 encode (bitwise equal, test_parvox_exact.py)
            return _orig_vox(vertices=vertices,faces=faces,grid_size=grid_size,**kw)
    class _OV:
        convert=_Conv()
    _pairs.o_voxel=_OV()
if os.environ.get('BENCH_PROF')=='1':  # debug only: synchronised sub-stage timers of the encodes (voxelisation / encoder network)
    import pairs as _pp
    _ovx=_pp.o_voxel.convert.mesh_to_flexible_dual_grid;_oenc=_pp.encoder
    def _tvox(*a,**k):
        torch.cuda.synchronize();t_=time.time();r_=_ovx(*a,**k);print(f'  PROF voxelize {k.get("grid_size")} {time.time()-t_:.2f}s',flush=True);return r_
    class _PC:
        def __getattr__(s,n):return _tvox if n=='mesh_to_flexible_dual_grid' else getattr(_pp.o_voxel.convert,n)
    class _PO:
        convert=_PC()
    _pp.o_voxel=_PO()
    def _tenc():
        e=_oenc()
        if not getattr(e,'_prof',False):
            _f=e.forward
            def fw(*a,**k):
                torch.cuda.synchronize();t_=time.time();r_=_f(*a,**k);torch.cuda.synchronize();print(f'  PROF encoder {time.time()-t_:.2f}s',flush=True);return r_
            e.forward=fw;e._prof=True
        return e
    _pp.encoder=_tenc
if OPT['campool']:import campool        # exact: one Powell start per pre-started worker process (whole/campool.py, test_campool.py)
# ---------------- Pixal3D 512 (own environment), concurrent with the TRELLIS.2 reference frame and its camera fit
PX=f'{WD}/pixal'
if PIXAL:os.makedirs(PX,exist_ok=True)
penv={k:v_ for k,v_ in os.environ.items() if not k.startswith(('DECODE_','OPT_','EST_','CAMREF_','CAMFIT_'))}
penv.update(HF_HUB_OFFLINE=os.environ.get('PIXAL_HF_OFFLINE','1'),HF_HUB_CACHE=f'{_E["SF3D_PIXAL3D_HF_CACHE"]}',PIXAL_FAST='1',
            PYTHONPATH=f'{_E["SF3D_ROOT"]}/third_party/site-packages')
penv['PXCHAIN_THREADS']=str(OPT['camthreads'] or 4)   # camera thread count of the chain (exact; 4 = queue lib.sh)
if OPT['camfast']:penv['PXCHAIN_FAST']='1'
if OPT['noinit']:penv['PIXAL_NOINIT']='1'
OPT['pxwarm']=os.environ.get('OPT_PXWARM','0')=='1' and PIXAL   # Pixal3D process + its camera driver started and imported before the timed window
WARM=0.                                                   # (interpreter start-up and imports excluded, as for this process); all work after T0
if not PIXAL and os.environ.get('OPT_PXWARM','0')=='1' and OPT['parvox']:   # TRELLIS.2 start: only the voxeliser pool is started before the window
    _tw=time.time();parvox.start(OPT['encpipe'] or OPT['parvox']).map(time.sleep,[.3]*(OPT['encpipe'] or OPT['parvox']),chunksize=1);WARM=time.time()-_tw
if OPT['pxwarm']:
    _tw=time.time();penv['PIXAL_LEAN']='1'
    PXW=subprocess.Popen([PIXAL_PY,f'{P}/bench/pixal_warm.py',INP,PX],env=penv,cwd=f'{P}/fig5',stdin=subprocess.PIPE,stdout=open(f'{WD}/pixal_worker.log','w'),stderr=subprocess.STDOUT)
    PXP=subprocess.Popen(['bash',f'{_E["SF3D_ROOT"]}/setup/python.sh',f'{P}/bench/cam_warm.py',PX],env=penv,stdin=subprocess.PIPE,stdout=open(f'{WD}/pixal.log','w'),stderr=subprocess.STDOUT)
    while not (os.path.exists(f'{PX}/pixal_ready') and os.path.exists(f'{PX}/cam_ready')):
        assert PXW.poll() is None and PXP.poll() is None,'warm Pixal3D process failed during start-up';time.sleep(.05)
    if OPT['parvox']:                                     # voxeliser worker pool: interpreters + imports before the window as well
        parvox.start(OPT['encpipe'] or OPT['parvox']).map(time.sleep,[.3]*(OPT['encpipe'] or OPT['parvox']),chunksize=1)
    WARM=time.time()-_tw
torch.cuda.reset_peak_memory_stats();T0=time.time()
if OPT['pxwarm']:
    for _q in (PXW,PXP):_q.stdin.write(b'go\n');_q.stdin.flush()
elif PIXAL:
    PXP=subprocess.Popen(['bash',f'{P}/bench/pixal_chain.sh',INP,PX],env=penv,stdout=open(f'{WD}/pixal.log','w'),stderr=subprocess.STDOUT)   # Pixal3D 512 + its camera (start.sh pixal512_M + _cam)
OVERLAP=[]                                   # stages that run while Pixal3D runs: their peaks add to the Pixal3D peak
# ---------------- frontend
def pipe(names):
    import trellis2.pipelines.trellis2_image_to_3d as pm
    class Dummy:
        def __getattr__(self,n):return lambda **kw:(lambda im:im.convert('RGBA'))
    pm.rembg=Dummy();Trellis2ImageTo3DPipeline.model_names_to_load=names
    p=Trellis2ImageTo3DPipeline.from_pretrained(CACHE);p.low_vram=True;p.to('cuda');return p
if OPT['noinit']:import noinit;noinit.wrap(models)
with stage('frontend'):
    p=pipe(['sparse_structure_flow_model','sparse_structure_decoder','shape_slat_flow_model_512','shape_slat_decoder'])
    if OPT['parvox']:parvox.start(OPT['encpipe'] or OPT['parvox'])      # worker pools start after the model loads, their imports overlap the GPU sampling
    if OPT['campool']:campool.start(OPT['campool'])
    im=p.preprocess_image(Image.open(INP).convert('RGBA'))
    c512=p.get_cond([im],512);cond1024=p.get_cond([im],1024)
    os.makedirs(f'{WD}/frontend',exist_ok=True);torch.save({k:v_.cpu() for k,v_ in cond1024.items()},f'{WD}/frontend/image_cond.pt')   # for the block replay
    torch.manual_seed(42);coords=p.sample_sparse_structure(c512,resolution=32,num_samples=1,sampler_params=dict(p.sparse_structure_sampler_params))   # = native512.py: upstream run(pipeline_type='512', seed=42), one seed
    z512=p.sample_shape_slat(c512,p.models['shape_slat_flow_model_512'],coords)
with stage('decode512'):
    m=p.decode_shape_slat(z512,512)[0][0];v512=m.vertices.float();F512=m.faces.long()
    for k in ('sparse_structure_flow_model','shape_slat_flow_model_512'):p.models[k].cpu()
    del m;gc.collect();torch.cuda.empty_cache()   # the decoder stays for the placed start
    L.write_ply(f'{WD}/mesh_512.ply',v512,F512);shutil.copy(INP,f'{WD}/input.png')
# ---------------- camera + crop (the pipeline's own scripts, run in this process)
def runscript(path,argv,env=None):
    old=sys.argv;sys.argv=[path]+argv
    try:runpy.run_path(path,run_name='__main__')
    except SystemExit:pass
    finally:sys.argv=old
PRELOADED={}
def preload_nirne():                # the detail estimators (diffusers): loading toggles the process-wide torch default dtype (fp16), so it
    q_=L.parse(dict(INJ512,UNIT='1/512'));L.model(q_['CK']);L.model(q_['RGB_CK'])   # must never overlap camera code of this process
def preload():                      # OPT_PRELOAD: weights only (no computation that depends on the camera)
    if not OPT['preload_split']:preload_nirne()   # both routes (grey / mesh-conditioned RGB)
    if TARGET>1024:
        from ablation_model import AblationFlow
        from safetensors.torch import load_file
        cfg=json.load(open(f'{_E["SF3D_CKPT"]}/refiner/train.json'))
        m=AblationFlow(**cfg['models']['denoiser']['args'])   # random init kept: lr_null_embedding is in neither checkpoint (noinit would change it)
        base=load_file(f'{CACHE}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors');_ks=set(base);m.load_state_dict(base,strict=False);del base
        state=torch.load(f'{_E["SF3D_CKPT"]}/refiner/denoiser_step0002000.pt',map_location='cpu',weights_only=True);_ks|=set(state);m.load_state_dict(state,strict=False);del state

        PRELOADED['refiner']=(cfg,m)
if OPT['campar']:                    # exact: one Powell start per process, the scripts' own selection (whole/campar.py)
    with stage('camera'):
        e=dict(os.environ);e['PYTHONPATH']=':'.join([f'{_E["SF3D_ROOT"]}/third_party/site-packages']+[q for q in sys.path if q])
        r=subprocess.run([sys.executable,f'{P}/whole/campar.py',f'{WD}/input.png',f'{WD}/mesh_512.ply',WD],env=e,capture_output=True,text=True,cwd=_E['SF3D_ROOT'])
        assert r.returncode==0,r.stderr[-3000:];print(r.stdout.strip().split(chr(10))[-1],flush=True)
else:
  if OPT['preload']:PRE=__import__('concurrent.futures',fromlist=['x']).ThreadPoolExecutor(1).submit(lambda:preload())
  WPK=[]
  if OPT['camthreads']:os.environ.update(CAMFIT_HULL='1',CAMFIT_THREADS=str(min(4,OPT['camthreads'])),CAMREF_HULL='1',CAMREF_LEAN='1',CAMREF_THREADS=str(OPT['camthreads']))
  if OPT['camfast']:os.environ['CAMFIT_FASTNORM']='2'
  FA=[f'{WD}/input.png',f'{WD}/mesh_512.ply',f'{WD}/cam_fit.json'];RA=FA+[f'{WD}/cam_refined.json']
  with stage('camera_fit'):
    if OPT['campool']:
        if OPT['camshare']:                 # exact: grid once in this process, shared with the workers and the combine pass
            os.environ['CAMFIT_GRIDFILE']=f'{WD}/fitgrid.json';os.environ['CAMFIT_GRIDONLY']='1';runscript(f'{P}/fit_input_camera.py',FA);os.environ.pop('CAMFIT_GRIDONLY')
        WE={'CAMFIT_HULL':'1'};WE.update({'CAMFIT_GRIDFILE':f'{WD}/fitgrid.json'} if OPT['camshare'] else {})
        WPK=[campool.run(OPT['campool'],f'{P}/fit_input_camera.py',FA,int(os.environ.get('CAMFIT_NSTART','4')),'CAMFIT',f'{WD}/fitparts',WE)];os.environ['CAMFIT_RESULTS']=f'{WD}/fitparts'
    runscript(f'{P}/fit_input_camera.py',FA)
  with stage('camera_refine'):
    os.environ.pop('CAMFIT_FASTNORM',None)
    if OPT['camfast']:os.environ.update(CAMREF_HULL='1',CAMREF_LEAN='1',**CAMFAST_REF)
    if OPT['camlock']:                  # exact: Powell starts as threads in this process, rasterized together per step (lockstep)
        if OPT['campool']:campool.close()
        os.environ.update(CAMREF_THREADS='12',CAMREF_LOCKSTEP='1',CAMREF_HULL='1',CAMREF_LEAN='1',CAMREF_FNCACHE='1',CAMREF_MEMO='1',CAMREF_GRIDBATCH='72')
    elif OPT['campool']:
        if OPT['camshare']:
            os.environ['CAMREF_GRIDFILE']=f'{WD}/refgrid.json';os.environ['CAMREF_GRIDONLY']='1';os.environ['CAMREF_GRIDBATCH']='72';runscript(f'{P}/whole/camera_refine.py',RA);os.environ.pop('CAMREF_GRIDONLY');os.environ.pop('CAMREF_GRIDBATCH')
        WE={'CAMREF_HULL':'1','CAMREF_LEAN':'1'};WE.update({'CAMREF_GRIDFILE':f'{WD}/refgrid.json','CAMREF_FNCACHE':'1','CAMREF_MEMO':'1'} if OPT['camshare'] else {})
        WPK.append(campool.run(OPT['campool'],f'{P}/whole/camera_refine.py',RA,4+int(os.environ.get('CAMREF_NGRID','8')),'CAMREF',f'{WD}/refparts',WE));os.environ['CAMREF_RESULTS']=f'{WD}/refparts'
    runscript(f'{P}/whole/camera_refine.py',RA)
    for k in ('CAMREF_LOCKSTEP','CAMREF_FNCACHE','CAMREF_MEMO','CAMREF_GRIDBATCH','CAMREF_FASTNORM'):os.environ.pop(k,None)
    if OPT['campool'] and not OPT['camlock']:campool.close()     # workers exit: their CUDA contexts and caches are released before the generation stages
  for k in ('CAMFIT_RESULTS','CAMREF_RESULTS','CAMFIT_GRIDFILE','CAMREF_GRIDFILE'):os.environ.pop(k,None)
  if OPT['campool']:                  # concurrent worker memory counts toward the peak: main-process camera peak + sum of worker peaks
    CAMW=max(sum(w) for w in WPK)/2**30 if WPK else 0.;PEAKS['camera_workers_sum']=round(CAMW,2);PEAK[0]=max(PEAK[0],max(PEAKS['camera_fit'],PEAKS['camera_refine'])+CAMW)
  for k in ('CAMFIT_HULL','CAMFIT_THREADS','CAMREF_HULL','CAMREF_LEAN','CAMREF_THREADS'):os.environ.pop(k,None)
D=f'{WD}/start';os.makedirs(D,exist_ok=True)
if not PIXAL:                                   # TRELLIS.2 start: the reference frame itself (its 512 latent, decoded mesh and fitted camera)
  PXPEAK={}
  with stage('start512'):
    if OPT['preload'] and OPT['preload_split']:PRE.result();preload_nirne()   # refiner thread done first (no concurrent model construction)
    np.savez(f'{D}/latent_512.npz',coords=z512.coords.cpu().numpy(),feats=z512.feats.float().cpu().numpy())
    for f_ in ('mesh_512.ply','input.png','cam_fit.json','cam_refined.json'):shutil.copy(f'{WD}/{f_}',f'{D}/{f_}')
    p.models['shape_slat_decoder'].cpu();del p;gc.collect();torch.cuda.empty_cache()
else:
 OVERLAP.extend([k for k in ('frontend','decode512','camera_fit','camera_refine')])
 with stage('pixal3d_wait'):                    # remaining Pixal3D time after the overlap (0 if it finished first)
    if OPT['preload'] and OPT['preload_split']:PRE.result();preload_nirne()   # refiner thread done first (no concurrent model construction)
    rc=PXP.wait();assert rc==0,open(f'{WD}/pixal.log').read()[-3000:]
    if OPT['pxwarm']:assert PXW.wait()==0,open(f'{WD}/pixal_worker.log').read()[-3000:]
 PXPEAK=json.load(open(f'{PX}/peak.json'));PXPEAK['chain_times']=json.load(open(f'{PX}/chain_times.json'))
 with stage('place512'):                         # cand_start.py with the fitted Pixal3D camera (as start.sh cand_M), then the official 512 decode (dec_res.py)
    runscript(f'{_E["SF3D_ROOT"]}/superfine3d/start/cand_start.py',[f'{PX}/zup/mesh_512.ply',f'{PX}/zup/cam_refined.json',f'{WD}/mesh_512.ply',f'{WD}/cam_refined.json',D])
    a_=np.load(f'{D}/latent_512.npz');z_=SparseTensor(feats=torch.from_numpy(a_['feats']).float().cuda(),coords=torch.from_numpy(a_['coords']).int().cuda())
    m2=p.decode_shape_slat(z_,512)[0][0];L.write_ply(f'{D}/mesh_512.ply',m2.vertices.float(),m2.faces.long());shutil.copy(INP,f'{D}/input.png')
    p.models['shape_slat_decoder'].cpu();del p,m2,z_;gc.collect();torch.cuda.empty_cache()
 with stage('camera_start'):                     # full silhouette fit + refinement on the placed start, as start.sh cam_M (a refine-only
    os.environ.update(CAMFIT_HULL='1',CAMFIT_THREADS='4',**({'CAMFIT_FASTNORM':'2'} if OPT['camfast'] else {}))   # start from the TRELLIS.2 camera got stuck when that camera is poor: TC001H IoU .88 vs .95)
    runscript(f'{P}/fit_input_camera.py',[f'{D}/input.png',f'{D}/mesh_512.ply',f'{D}/cam_fit.json'])
    for k in ('CAMFIT_HULL','CAMFIT_THREADS','CAMFIT_FASTNORM'):os.environ.pop(k,None)
    _ce=dict(CAMREF_HULL='1',CAMREF_LEAN='1',CAMREF_THREADS=str(max(1,OPT['camthreads'] or 12)))
    if OPT['camfast']:_ce.update(CAMFAST_REF)
    os.environ.update(_ce)
    runscript(f'{P}/whole/camera_refine.py',[f'{D}/input.png',f'{D}/mesh_512.ply',f'{D}/cam_fit.json',f'{D}/cam_refined.json'])
    for k in _ce:os.environ.pop(k,None)
with stage('crop'):
    if OPT['preload']:PRE.result()
    os.symlink(f'{D}/mesh_512.ply',f'{D}/mesh_1024.ply') if not os.path.exists(f'{D}/mesh_1024.ply') else None
    runscript(f'{P}/relief24/crop_input.py',[D])
img=L.load_image(f'{D}/input_crop.png');cam=L.load_cam(f'{D}/cam_refined.json')
_vm,_Fm=L.read_mesh(f'{D}/mesh_512.ply');v512=torch.from_numpy(_vm).float().cuda();F512=torch.from_numpy(_Fm).long().cuda()
os.environ.update(EST_CACHE=f'{WD}/est.npz',EST_CACHE_F32='1',EST_GPU_ACC='1',EST_CTRL_CACHE='1')   # arms/T1.sh + pipeline_v3_fast exact switches
EST={}
def inject(v,F,unit,cfg):
    c=dict(cfg);c['UNIT']=unit
    if True:return L.inject(v,F,img,cam,c)[:2]       # T1: route + evidence gate + EST_CACHE (estimate once), as pipeline_v3_fast
    q=L.parse(c)
    if 'E' not in EST:EST['E']=hest.estimate(img,cam,img.shape[0],L.model(q['CK']),batch=8)
    if q['SUB']:
        from hfdetail import subdivide;v,F=subdivide.midpoint(v,F)
    params=hf.HFParams(iters=q['IT'],maxd=q['MAXD'],lap=q['LAP'],lap_huber=q['LAPH'],levels=q['LEVELS'])
    out=hpipe.refine(v,F,hf.Frame((0.,0.,0.),1.,q['UNIT']),cam,image=img,target=EST['E'],res=img.shape[0],mode=q['MODE'],k=q['K'],gain=q['GAIN'],
                     params=params,fine_k=q['FK'],fine_gain=q['FG'] if q['FK'] else None,edge_tau=q['EDGE'])
    return out,F
def encode(v,F,res,ns=1):
    import pairs;pairs.encoder().eval().requires_grad_(False)   # inference only: no backward neighbour caches
    if res==1024:c,f=latent_io.encode_direct(v.cpu(),F.cpu(),res)         # run_512.sh: enc_direct.py
    elif OPT['encpipe']:c,f=encpipe.encode_slabs(v.cpu(),F.cpu(),res,ns,OPT['parvox'],OPT['encpipe'])   # same slabs, concurrent voxelisation
    else:c,f=latent_io.encode_slabs(v.cpu(),F.cpu(),res,nslab=ns)          # pipeline_v3: encode_slabs
    return c,f
v,F=v512,F512
with stage('inject512'):v,F=inject(v,F,'1/512',INJ512)
with stage('encode1024'):c1024,f1024=encode(v,F,1024);torch.cuda.empty_cache()
# ---------------- decoder + refiner
import portable_pipeline as pp
from decode_deferred_skip import prepare_install
_inst=prepare_install();_old=pp.install_decoder_lowmem_patch
_DONE=[False]
def _patch():
    if _DONE[0]:return
    _old();_inst();_DONE[0]=True
pp.install_decoder_lowmem_patch=_patch
from pathlib import Path
if OPT['route_fast']:                # exact: per-token routing keys computed once (bench/fastroute.py, test_fastroute.py)
    import fastroute;fastroute.install()
def decode(res,coords,feats):
  np.savez(f'{WD}/latent_{res}.npz',coords=coords,feats=feats)
  with tf32():                          # decode.py sets allow_tf32=True
    if OPT['dec_fast']:                 # exact: decoder built once, mesh kept in memory (bench/decfast.py, test_decfast.py)
        import decfast;return decfast.decode(pp,res,coords,feats)
    pp.OUT=Path(WD);pp.decode(res)
    v,F=L.read_mesh(f'{WD}/mesh_{res}.ply');return torch.from_numpy(v).float().cuda(),torch.from_numpy(F).long().cuda()
def bil(v,F):                        # OPT_BILAT: bilateral normal filtering of a decoded / injected / final mesh (metric round 6 config)
    if not OPT['bilat']:return v
    from hfdetail import denoise;i_,sg_,vi_=OPT['bilat'].split(':');v=denoise.bilateral(v.float(),F.long(),int(i_),float(sg_),int(vi_))
    if OPT['bilat_ec']:torch.cuda.empty_cache()
    return v
def export(v,F,name):
    st={};v,F=cleanup.cap_small_holes(v,F,max_edges=60,max_extent=8/4096.,stats=st);F=cleanup.close_tjunctions(v,F,.5/4096.,stats=None if OPT['export_lean'] else st)   # stats only re-count boundary edges for a log line
    v=bil(v,F);L.write_ply(f'{WD}/{name}',v,F)      # after the repair, as pipeline_v3_fast FINAL_BILAT
if TARGET==1024:
    with stage('decode1024'):v,F=decode(1024,c1024,f1024)
    with stage('export'):export(v,F,'final_1024.ply')
else:
    import inspect,textwrap
    from ablation_model import AblationFlow
    from session import StructureSession
    from safetensors.torch import load_file
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
    import trellis2.pipelines.samplers.flow_euler as em
    gen=open(f'{N}/generate.py').read();g_ns={}
    g_ns.update(types=types,torch=torch);exec(compile(gen[gen.index('def bounded'):gen.index('def frontend')],'<bounded>','exec'),g_ns)
    with stage('load_refiner'):
        if 'refiner' in PRELOADED:cfg,model=PRELOADED.pop('refiner')
        else:
            cfg=json.load(open(f'{_E["SF3D_CKPT"]}/refiner/train.json'));model=AblationFlow(**cfg['models']['denoiser']['args'])
            base=load_file(f'{CACHE}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors');model.load_state_dict(base,strict=False);del base
            state=torch.load(f'{_E["SF3D_CKPT"]}/refiner/denoiser_step0002000.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=False);del state
        g_ns['bounded'](model);model.eval();model.use_checkpoint=False
        sp_pipe=pipe([]);sampler=sp_pipe.shape_slat_sampler;sp=dict(sp_pipe.shape_slat_sampler_params,guidance_strength=1.0)
        norm=cfg['dataset']['args']['normalization'];ln=cfg['dataset']['args']['lr_normalization']
        mean=torch.tensor(norm['mean'],device='cuda')[None];std=torch.tensor(norm['std'],device='cuda')[None]
        lm=torch.tensor(ln['mean'],device='cuda')[None];ls=torch.tensor(ln['std'],device='cuda')[None]
        cond={k:v_.cuda() for k,v_ in cond1024.items()}
        src=textwrap.dedent(inspect.getsource(FlowEulerSampler.sample)).replace('ret.pred_x_t.append(out.pred_x_prev)','pass').replace('ret.pred_x_0.append(out.pred_x_0)','pass')
        src=src.replace('t_seq = t_seq.tolist()','t_seq = t_seq.tolist(); _T0 = __import__("builtins")._SDEDIT_T0; t_seq = [_T0] + [t for t in t_seq if t < _T0 - 1e-9]').replace('for i in range(steps))','for i in range(len(t_seq) - 1))')
        ns_=dict(vars(em));exec(compile(src,'<sdedit>','exec'),ns_);FlowEulerSampler.sample=ns_['sample']
    def refine(res,init_c,init_f,lr_c,lr_f):
        coords=torch.from_numpy(init_c).int().cuda();x0=(torch.from_numpy(init_f).float().cuda()-mean)/std;sm=sampler.sigma_min
        TSD=T0S[res];__import__('builtins')._SDEDIT_T0=TSD
        torch.manual_seed(730004+(3000 if res==2048 else 4000))
        noise=SparseTensor(feats=(1-TSD)*x0+(sm+(1-sm)*TSD)*torch.randn_like(x0),coords=coords)
        lr=SparseTensor(coords=torch.from_numpy(lr_c).int().cuda(),feats=(torch.from_numpy(lr_f).float().cuda()-lm)/ls)
        model.cuda();model.set_resolution_pair(res//16,res//32)
        session=StructureSession(model,lr,coords,[cond['cond'],cond['neg_cond']],res//16,variant='K2_hr_allcells',runtime_variant='V5_cpu',verify_kernel=False)
        if os.environ.get('FUSED','1')=='1':sys.path.append(f'{_E["SF3D_ROOT"]}/superfine3d/detail/bench');import ours_fast;report_fused=ours_fast.install(model,session)   # fused DiT kernels (FUSED=0: original path)
        try:
            with tf32():out=sampler.sample(session,noise,**cond,**sp,verbose=False).samples   # generate.py sets allow_tf32=True
        finally:session.close()
        out=out.replace(out.feats*std+mean);model.cpu();gc.collect();torch.cuda.empty_cache()
        return out.coords.cpu().numpy(),out.feats.float().cpu().numpy()
    with stage('inject1024'):v,F=inject(v,F,'1/1024',INJ1)
    with stage('encode2048'):c2048,f2048=encode(v,F,2048);del v,F;torch.cuda.empty_cache()
    with stage('refine2048'):z2048=refine(2048,c2048,f2048,c1024,f1024)
    with stage('decode2048'):v,F=decode(2048,*z2048)
    if OPT['bilat'] and not OPT['bilat_dec4096']:
        with stage('denoise2048'):v=bil(v,F)
    if TARGET==2048:
        with stage('inject2048_final'):v,F=inject(v,F,'1/2048',INJ)
        with stage('export'):export(v,F,'final_2048.ply')
    else:
        with stage('inject2048'):v,F=inject(v,F,'1/2048',INJ)
        if OPT['bilat'] and not OPT['bilat_dec4096']:
            with stage('denoise_inj2048'):v=bil(v,F)
        if os.environ.get('BENCH_SUPPORT_ONLY')=='1':    # untimed supports on shared GPUs: free the estimator/encoder first
            import pairs as _p
            for _m in L._MODELS.values():
                try:_m.net.cpu()
                except Exception:pass
            L._MODELS.clear();EST.clear();gc.collect();torch.cuda.empty_cache()
            print('support mode: GPU allocated before encode4096 %.2f GiB'%(torch.cuda.memory_allocated()/2**30),flush=True)
        if os.environ.get('BENCH_DUMP'):np.savez(os.environ['BENCH_DUMP'],v=v.cpu().numpy(),F=F.cpu().numpy())   # debug: encode4096 input for component tests
        with stage('encode4096'):c4096,f4096=encode(v,F,4096,OPT['nslab']);del v,F;torch.cuda.empty_cache()
        print(f'refine4096 input: {len(c4096)} tokens, lr {len(z2048[0])}, feats mean {float(np.abs(f4096).mean()):.5f}',flush=True)   # debug line (stdout only)
        with stage('refine4096'):z4096=refine(4096,c4096,f4096,*z2048)
        if os.environ.get('BENCH_SUPPORT_ONLY')=='1':   # block-replay supports: refined 2048 (LR) and 4096 (HR) latents only, untimed
            np.savez(f'{WD}/latent_4096.npz',coords=z4096[0],feats=z4096[1])
            json.dump(dict(target=4096,support_only=True,times=times,opt=OPT,input=INP,hr_tokens=len(z4096[0]),lr_tokens=len(z2048[0])),open(OUTJ,'w'),indent=1)
            print('SUPPORT_DONE',len(z4096[0]),len(z2048[0]),flush=True);os._exit(0)
        with stage('decode4096'):v,F=decode(4096,*z4096)
        if OPT['bilat']:
            with stage('denoise4096'):v=bil(v,F)
        with stage('inject4096'):v,F=inject(v,F,'1/4096',FIN)
        with stage('export'):export(v,F,'final_4096.ply')
torch.cuda.synchronize();total=time.time()-T0
_ov=max([PEAKS.get(k,0) for k in OVERLAP]+[0]);PEAK[0]=max(PEAK[0],_ov+(PXPEAK.get('peak_gib') or 0))   # concurrent Pixal3D process counts toward the peak
rep=dict(config=f'Table-1 configuration, {START} 512 start',start=START,pixal3d=PXPEAK,overlap_stages=OVERLAP,target=TARGET,total_seconds=round(total,2),startup_and_imports_seconds=round(T0-T_START,2),pixal_warmup_seconds=round(WARM,2),times=times,peak_allocated_gib=round(PEAK[0],3),stage_peak_gib=PEAKS,
         peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,3),opt=OPT,input=INP,contention=marks,timing_valid=not marks,gpu_name=torch.cuda.get_device_name(0),
         gpu_shared=SHARED,foreign_pids_seen=sorted(SEEN),stage_load1=LOADS)
json.dump(rep,open(OUTJ,'w'),indent=1);print('BENCH',json.dumps(rep),flush=True)
if OPT['parvox']:parvox.close()
