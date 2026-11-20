from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,gc,json,time,types
from pathlib import Path
import numpy as np
import torch
import trellis2.models as models
from trellis2.modules.sparse import SparseTensor
from decode_probe import install_chunked_mlp
b=types.SimpleNamespace(SparseTensor=SparseTensor)
OUT=Path('.')
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def clean():
 gc.collect();torch.cuda.empty_cache()
def load(res):
 with np.load(OUT/f'latent_{res}.npz') as f:
  return SparseTensor(feats=torch.from_numpy(f['feats'].copy()).cuda(),coords=torch.from_numpy(f['coords'].copy()).cuda())
def named_pipe(names):
 assert names==['shape_slat_decoder']
 ckpt=f'{_E["SF3D_TRELLIS2_WEIGHTS"]}/ckpts/shape_dec_next_dc_f16c32_fp16'
 decoder=models.from_pretrained(ckpt).eval()
 return types.SimpleNamespace(models={'shape_slat_decoder':decoder})
def setup_decoder(decoder):
    sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/refiner/decode_code')
    from chunk_norm import install
    from functional_norm_guard import install as guard
    install(decoder);guard(decoder)
    from trellis2.modules.sparse.conv.conv import SparseConv3d,SparseInverseConv3d
    def dtype_guard(m,args):
        x=args[0]
        return (x.replace(x.feats.to(m.weight.dtype)),*args[1:]) if x.feats.dtype!=m.weight.dtype else args
    for m in decoder.modules():
        if isinstance(m,(SparseConv3d,SparseInverseConv3d)):m.register_forward_pre_hook(dtype_guard)
    # Pointwise activation has no interactions across rows; use in-place SiLU
    # to remove its full expanded activation allocation without changing GEMMs.
    count=0
    # Verify the exact production MLP weights and dtype, then change only SiLU.
    checked=set()
    for block in decoder.modules():
        if block.__class__.__name__!='SparseConvNeXtBlock3d':continue
        mlp=block.mlp
        signature=(mlp[0].in_features,mlp[0].out_features,mlp[0].weight.dtype)
        if signature in checked:continue
        device=mlp[0].weight.device
        mlp.cuda()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            x=torch.randn(513,mlp[0].in_features,device='cuda',dtype=mlp[0].weight.dtype)
            ref=mlp(x);old=mlp[1].inplace;mlp[1].inplace=True;value=mlp(x)
            assert torch.equal(ref,value),'Inplace activation changed MLP output'
            mlp[1].inplace=old
        del x,ref,value
        mlp.to(device);checked.add(signature)
    for m in decoder.modules():
        if isinstance(m,torch.nn.SiLU):m.inplace=True;count+=1
    print('DECODER_INPLACE_SILU',count,'EXACT_MLP_SHAPES',len(checked),flush=True)
    for m in decoder.modules():
        if isinstance(m,torch.nn.SiLU):m.inplace=False
    install_chunked_mlp(decoder,65536)
    return next(decoder.parameters()).dtype

def install_decoder_lowmem_patch():
    import inspect,textwrap
    import trellis2.models.sc_vaes.sparse_unet_vae as vae
    for cls in [vae.SparseResBlockC2S3d,vae.SparseConvNeXtBlock3d]:
        source=textwrap.dedent(inspect.getsource(cls._forward))
        source=source.replace('h = h + self.skip_connection(x)','h.feats.add_(self.skip_connection(x).feats)')
        source=source.replace('return h + x','h.feats.add_(x.feats)\n    return h')
        ns=dict(vars(vae))
        exec(compile(source,'inference_inplace_residual','exec'),ns)
        cls._forward=ns['_forward']
    def retire_conv_maps(h):
        for scale_cache in h._spatial_cache.values():
            if isinstance(scale_cache,dict):
                for key in list(scale_cache):
                    if str(key).startswith('SubMConv3d_neighbor_cache_'):scale_cache.pop(key)
        gc.collect();torch.cuda.empty_cache()
    source=textwrap.dedent(inspect.getsource(vae.SparseUnetVaeDecoder.forward))
    source=source.replace('h = h.type(x.dtype)','retire_conv_maps(h)\n    h = h.type(x.dtype)')
    ns=dict(vars(vae));ns['retire_conv_maps']=retire_conv_maps
    exec(compile(source,'inference_retire_maps','exec'),ns)
    vae.SparseUnetVaeDecoder.forward=ns['forward']

def decode(res):
    import trimesh
    from trellis2.modules.sparse.conv import conv as sparse_conv_dispatch
    pipe=named_pipe(['shape_slat_decoder']);decoder=pipe.models['shape_slat_decoder'];dtype=setup_decoder(decoder)
    install_decoder_lowmem_patch()
    import trellis2.models.sc_vaes.sparse_unet_vae as vae
    assert vae.SparseUnetVaeDecoder.forward.__globals__['F'].__class__.__name__=='FunctionalProxy'
    def final_norm_gate(module,args):
        x=args[0];n=min(8192,len(x.feats));idx=torch.arange(n,device=x.feats.device,dtype=torch.int64)*(len(x.feats)-1)//max(n-1,1)
        f=x.feats[idx].float();rms=f.square().mean(-1).sqrt()
        assert torch.isfinite(f).all() and bool((rms>.85).all()) and bool((rms<1.15).all()),'Invalid final normalized features'
        print('FINAL_NORM_GATE_PASS',len(x.feats),float(rms.min()),float(rms.max()),flush=True)
    decoder.output_layer.register_forward_pre_hook(final_norm_gate)
    decoder.eval().requires_grad_(False)
    import importlib
    conv_impl=importlib.import_module('flex_gemm.ops.spconv.submanifold_conv3d')
    original_cache=conv_impl.SubMConv3dFunction._compute_neighbor_cache
    cache_calls=[]
    def checked_cache(coords,shape,kernel_size,dilation,needs_grad):
        assert not needs_grad, 'Inference unexpectedly requested backward cache'
        cache_calls.append(int(coords.shape[0]))
        print('NO_BWD_CACHE',coords.shape[0],flush=True)
        return original_cache(coords,shape,kernel_size,dilation,False)
    conv_impl.SubMConv3dFunction._compute_neighbor_cache=staticmethod(checked_cache)
    from o_voxel.convert.flexible_dual_grid import _init_hashmap
    assert _init_hashmap(torch.tensor([4096]*3,dtype=torch.int32),8,'cpu')[0].dtype==torch.uint64
    z=load(res);z=z.replace(z.feats.to(dtype));decoder.set_resolution(res);decoder.low_vram=True;decoder.cuda()
    clean();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    z=z.replace(z.feats.detach())
    with torch.inference_mode():
        decoded=decoder(z,return_subs=False)[0]
    torch.cuda.synchronize()
    mesh=trimesh.Trimesh(decoded.vertices.float().cpu().numpy(),decoded.faces.cpu().numpy(),process=False)
    target=OUT/f'mesh_{res}.ply'
    if res==2048 and target.exists():
        reference=trimesh.load(target,process=False)
        same_vertices=np.array_equal(np.asarray(reference.vertices),np.asarray(mesh.vertices))
        same_faces=np.array_equal(np.asarray(reference.faces),np.asarray(mesh.faces))
        write(OUT/'nobwd_2048_equivalence.json',dict(vertices_exact=same_vertices,faces_exact=same_faces,cache_calls=cache_calls))
        assert same_vertices and same_faces, '2048 mesh changed after no-backward-cache fix'
        del reference
        print('2048_MESH_PASS_EXACT',flush=True)
    else:
        mesh.export(target)
    write(OUT/f'decode_{res}.json',dict(seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,vertices=len(mesh.vertices),faces=len(mesh.faces),protocol='frozen decoder, inference_mode, asserted no-backward neighbor caches; original GEMMs and precision',cache_calls=cache_calls))
    print('DECODE_COMPLETE',res,flush=True)
