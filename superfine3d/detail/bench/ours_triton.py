"""Hand-written Triton kernels for the fused DiT ops (no torch.compile / Dynamo: per-process cost = Triton JIT, disk-cached).
Same arithmetic order as eager: fp32 inside, bf16 roundings where eager stores bf16 tensors."""
import torch,triton,triton.language as tl
@triton.jit
def _norm_rope_k(X,G,C,S,O,R,sxn,sxh,H:tl.constexpr,D2:tl.constexpr,SCALE,ROPE:tl.constexpr,BR:tl.constexpr):
    r=tl.program_id(0).to(tl.int64)*BR+tl.arange(0,BR);rm=r<R;n=r//H;h=r%H;j=tl.arange(0,D2)   # int64: offsets exceed 2^31 at 700k tokens
    base=X+n[:,None]*sxn+h[:,None]*sxh;m=rm[:,None]
    a=tl.load(base+2*j[None,:],m,0.).to(tl.float32);b=tl.load(base+2*j[None,:]+1,m,0.).to(tl.float32)
    nrm=tl.maximum(tl.sqrt(tl.sum(a*a,1)+tl.sum(b*b,1)),1e-12)[:,None]
    ga=tl.load(G+h[:,None]*(2*D2)+2*j[None,:],m,0.);gb=tl.load(G+h[:,None]*(2*D2)+2*j[None,:]+1,m,0.)
    a=((a/nrm)*ga*SCALE).to(tl.bfloat16).to(tl.float32);b=((b/nrm)*gb*SCALE).to(tl.bfloat16).to(tl.float32)
    ob=O+r[:,None]*(2*D2)
    if ROPE:
        c=tl.load(C+n[:,None]*D2+j[None,:],m,1.);s=tl.load(S+n[:,None]*D2+j[None,:],m,0.)
        tl.store(ob+2*j[None,:],(a*c-b*s).to(tl.bfloat16),m);tl.store(ob+2*j[None,:]+1,(a*s+b*c).to(tl.bfloat16),m)
    else:
        tl.store(ob+2*j[None,:],a.to(tl.bfloat16),m);tl.store(ob+2*j[None,:]+1,b.to(tl.bfloat16),m)
def norm_rope(x,gamma,scale,cos=None,sin=None):
    """x (N,H,D) bf16 (any row/head strides, last dim contiguous), gamma (H,D) fp32 -> (N,H,D) bf16 contiguous."""
    shp=x.shape
    if x.dim()!=3:x=x.reshape(-1,*shp[-2:])
    if x.stride(2)!=1:x=x.contiguous()
    N,H,D=x.shape;assert x.dtype==torch.bfloat16;out=torch.empty((N,H,D),device=x.device,dtype=x.dtype);R=N*H;BR=16
    g=gamma.float().contiguous();rope=cos is not None
    _norm_rope_k[(triton.cdiv(R,BR),)](x,g,cos if rope else g,sin if rope else g,out,R,x.stride(0),x.stride(1),H,D//2,float(scale),rope,BR,num_warps=4)
    return out.reshape(shp)
@triton.jit
def _ln_k(X,W,B,SC,SH,O,R,C:tl.constexpr,BC:tl.constexpr,EPS,AFF:tl.constexpr,MOD:tl.constexpr):
    r=tl.program_id(0).to(tl.int64);k=tl.arange(0,BC);m=k<C
    x=tl.load(X+r*C+k,m,0.).to(tl.float32);mu=tl.sum(x,0)/C;d=tl.where(m,x-mu,0.);var=tl.sum(d*d,0)/C
    y=d*tl.rsqrt(var+EPS)
    if AFF:y=y*tl.load(W+k,m,1.)+tl.load(B+k,m,0.)
    y=y.to(tl.bfloat16)
    if MOD:
        s1=(1.+tl.load(SC+k,m,0.).to(tl.float32)).to(tl.bfloat16).to(tl.float32)         # (1 + scale) rounded as eager
        y=(y.to(tl.float32)*s1).to(tl.bfloat16).to(tl.float32)+tl.load(SH+k,m,0.).to(tl.float32);y=y.to(tl.bfloat16)
    tl.store(O+r*C+k,y,m)
def layer_norm(x,eps,weight=None,bias=None,scale=None,shift=None):
    """LayerNorm32 (fp32 statistics) of a bf16 (N,C) tensor, optional affine weight/bias, optional adaLN (1+scale), shift."""
    assert x.dtype==torch.bfloat16 and x.is_contiguous();N,C=x.shape;out=torch.empty_like(x);aff=weight is not None;mod=scale is not None
    d=x.new_zeros(1)
    _ln_k[(N,)](x,weight if aff else d,bias if aff else d,scale.reshape(-1) if mod else d,shift.reshape(-1) if mod else d,out,N,C,triton.next_power_of_2(C),float(eps),aff,mod,num_warps=8)
    return out
