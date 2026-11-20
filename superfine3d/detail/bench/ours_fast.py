"""Candidate speed patches for the Ours HR block (lab only). apply(name, m, s) installs one patch; names below.
lora_merge  : fold LoRA (to_qkv, to_out) into the base weight (math identical, bf16 rounding differs)
kv_gpu      : keep LR K/V of each layer on the GPU after first use (exact; costs GPU memory)
qk_fused    : RMSNorm(q,k)+RoPE in one compiled kernel (fp32 inside, same op order)
ln_fused    : LayerNorm32 (+ adaLN affine) compiled, no fp32 round trip through memory
ffn_fused   : FFN linear-GELU-linear compiled (GELU fused into the GEMM epilogue / one pass)
xq_fused    : cross-attention q/k RMSNorm compiled
lr_indexed  : HR->LR read through the index-only kernel (O4_lr_indexed, bitwise exact; used by the timing replay)
install(m, s): production entry (generate.py / bench_v3_body.py after StructureSession): lr_indexed + PROD."""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,torch,torch.nn.functional as F
# kernel_arithmetic lives in superfine3d/refiner/code (already on sys.path)
PROD=['lora_merge','qk_fused','ln_fused','xq_fused']
import torch._inductor.config as ic
ic.emulate_precision_casts=True            # keep eager's intermediate bf16 roundings inside fused kernels
CO=dict(dynamic=True,fullgraph=True)
import os as _os
TRITON=_os.environ.get('FUSED_IMPL','triton')=='triton'   # 'triton': hand kernels (default, no Dynamo); 'compile': torch.compile versions
@torch.compile(**CO)
def _norm_rope(x,gamma,scale,cos,sin):
    y=(F.normalize(x.float(),dim=-1)*gamma*scale).to(x.dtype).float()
    N,H,D=y.shape;y=y.reshape(N,H,D//2,2);a=y[...,0];b=y[...,1];c=cos[:,None,:];s=sin[:,None,:]
    return torch.stack((a*c-b*s,a*s+b*c),-1).reshape(N,H,D).to(x.dtype)
@torch.compile(**CO)
def _rms(x,gamma,scale):return (F.normalize(x.float(),dim=-1)*gamma*scale).to(x.dtype)
@torch.compile(**CO)
def _ln(x,eps:float):return F.layer_norm(x.float(),x.shape[-1:],None,None,eps).to(x.dtype)
@torch.compile(**CO)
def _ln_w(x,w,b,eps:float):return F.layer_norm(x.float(),x.shape[-1:],w,b,eps).to(x.dtype)
@torch.compile(**CO)
def _ln_affine(x,scale,shift,eps:float):
    h=F.layer_norm(x.float(),x.shape[-1:],None,None,eps).to(x.dtype);return h*(1+scale)+shift
@torch.compile(**CO)
def _ffn(x,w1,b1,w2,b2):return F.linear(F.gelu(F.linear(x,w1,b1),approximate='tanh'),w2,b2)
if TRITON:
    import ours_triton as _ot
    _norm_rope=lambda x,gamma,scale,cos,sin:_ot.norm_rope(x,gamma,scale,cos,sin)
    _rms=lambda x,gamma,scale:_ot.norm_rope(x,gamma,scale)
    _ln=lambda x,eps:_ot.layer_norm(x,eps)
    _ln_w=lambda x,w,b,eps:_ot.layer_norm(x,eps,w,b)
    _ln_affine=lambda x,scale,shift,eps:_ot.layer_norm(x,eps,scale=scale,shift=shift)
def apply(name,m,s):
    if name=='lora_merge':
        for blk in m.blocks:
            for lin in (blk.self_attn.to_qkv,blk.self_attn.to_out):
                if not hasattr(lin,'lora_A'):continue
                W=(lin.weight.float()+lin.lora_B.float()@lin.lora_A.float()).to(lin.weight.dtype);b=lin.bias
                lin.forward=(lambda x,_W=W,_b=b:F.linear(x,_W,_b))
    elif name=='kv_gpu':
        prior=s.lr_step;cache={}
        def lr_step(i,lr,lm,cond,pos):
            key=(s.slot,i)
            if key in cache:return cache[key]
            r=prior(i,lr,lm,cond,pos);cache[key]=r;return r
        s.lr_step=lr_step
    elif name=='qk_fused':
        prior=m._qkv;pc={}
        def qkv(attn,x,positions):
            q,k,v=attn.to_qkv(x.feats).reshape(-1,3,m.num_heads,m.model_channels//m.num_heads).unbind(1)
            key=(id(positions),positions._version if not positions.is_inference() else None)
            if key not in pc:
                angle=torch.outer(positions.reshape(-1).float(),m.physical_rope.freqs);ph=torch.polar(torch.ones_like(angle),angle).reshape(len(positions),-1)
                pad=m.model_channels//m.num_heads//2-ph.shape[-1]
                if pad:ph=torch.cat((ph,torch.ones(len(positions),pad,device=ph.device,dtype=ph.dtype)),1)
                pc.clear();pc[key]=(ph.real.contiguous(),ph.imag.contiguous())
            c,sn=pc[key];qn,kn=attn.q_rms_norm,attn.k_rms_norm
            return _norm_rope(q,qn.gamma,qn.scale,c,sn),_norm_rope(k,kn.gamma,kn.scale,c,sn),v
        m._qkv=qkv
    elif name=='ln_fused':
        import kernel_arithmetic as ka
        def pre(block,x,mod):
            z=(block.modulation+mod).to(mod.dtype) if block.share_mod else block.adaLN_modulation(mod)
            shift,scale,gate,sm,scm,gm=z.chunk(6,dim=1)
            return x.replace(_ln_affine(x.feats,scale,shift,block.norm1.eps)),gate,sm,scm,gm
        def post(block,x,h,gate,sm,scm,gm,cond):
            x=x.replace(ka.residual(x.feats,h.feats,gate))
            x=x+block.cross_attn(x.replace(_ln_w(x.feats,block.norm2.weight,block.norm2.bias,block.norm2.eps)),cond)
            h=x.replace(_ln_affine(x.feats,scm,sm,block.norm3.eps))
            return x.replace(ka.residual(x.feats,block.mlp(h).feats,gm))
        m._pre=pre;m._post=post
    elif name=='ffn_fused':
        for blk in m.blocks:
            f,_,l=blk.mlp.mlp;blk.mlp.forward=(lambda x,_f=f,_l=l:x.replace(_ffn(x.feats,_f.weight,_f.bias,_l.weight,_l.bias)))
    elif name=='xq_fused':
        for blk in m.blocks:
            for n in (blk.cross_attn.q_rms_norm,blk.cross_attn.k_rms_norm):
                n.forward=(lambda x,_n=n:x.replace(_rms(x.feats,_n.gamma,_n.scale)) if hasattr(x,'feats') else _rms(x,_n.gamma,_n.scale))
    elif name=='lr_indexed':
        from kernel_session import indexed
        def read(sq,q,k,v,prep,plans):
            result=torch.zeros_like(sq)
            for qi,ki,cq,ck,mq,mk in plans:
                z=indexed(sq[qi],k,v,cq,ck,mq,mk,s.idx(ki));result.index_copy_(0,qi,z)
            return result[prep.inv_perm]
        merged={}
        def read_merged(sq,q,k,v,prep,plans):
            # opt-in (env LR_MERGED=1, eff_opt r7): all LR read groups of a layer in ONE indexed launch. Every query row keeps its own
            # key set, key order and tile partition (sequences are independent in the varlen kernel), so the values are unchanged.
            c=merged.get(id(plans))
            if c is None or c[0] is not plans:
                qis=[p[0] for p in plans];oq=ok=0;cqs=[plans[0][2][:1]];cks=[plans[0][3][:1]]
                for qi,ki,cq,ck,mq,mk in plans:
                    assert int(cq[-1])==len(qi) and int(ck[-1])==len(ki)
                    cqs.append(cq[1:]+oq);cks.append(ck[1:]+ok);oq+=len(qi);ok+=len(ki)
                CQ=torch.cat(cqs).int();CK=torch.cat(cks).int()
                if len(CQ)-1>65535:c=(plans,None)          # grid.y limit of the varlen launch: keep the per-group launches
                else:c=(plans,torch.cat(qis),torch.cat([p[1] for p in plans]).to(torch.int32),CQ,CK,max(p[4] for p in plans),max(p[5] for p in plans))
                if len(merged)>64:merged.clear()
                merged[id(plans)]=c
            if c[1] is None:return read(sq,q,k,v,prep,plans)
            _,QI,KI,CQ,CK,MQ,MK=c
            result=torch.zeros_like(sq);result.index_copy_(0,QI,indexed(sq[QI],k,v,CQ,CK,MQ,MK,KI))
            return result[prep.inv_perm]
        s.read=read_merged if _os.environ.get('LR_MERGED')=='1' else read
    else:raise ValueError(name)
def install(m,s):
    for n in ['lr_indexed']+PROD:apply(n,m,s)
    return ['lr_indexed']+PROD
