"""Single-sample inference pointwise fusion with explicit intermediate rounding."""
import torch
import triton
import triton.language as tl

@triton.jit
def _affine(X,S,B,O,N:tl.constexpr,C:tl.constexpr,T:tl.constexpr):
    i=tl.program_id(0)*T+tl.arange(0,T);mask=i<N
    x=tl.load(X+i,mask,0);s=tl.load(S+i%C);b=tl.load(B+i%C)
    dtype=x.dtype
    product=(x.to(tl.float32)*s.to(tl.float32)).to(dtype).to(tl.float32)
    tl.store(O+i,product+b.to(tl.float32),mask)

@triton.jit
def _residual(X,H,G,O,N:tl.constexpr,C:tl.constexpr,T:tl.constexpr):
    i=tl.program_id(0)*T+tl.arange(0,T);mask=i<N
    x=tl.load(X+i,mask,0);h=tl.load(H+i,mask,0);g=tl.load(G+i%C)
    product=(h.to(tl.float32)*g.to(tl.float32)).to(h.dtype).to(tl.float32)
    tl.store(O+i,x.to(tl.float32)+product,mask)

def affine(x,scale,shift):
    assert scale.shape[0]==shift.shape[0]==1 and x.dtype==scale.dtype==shift.dtype
    s=1+scale # preserve original BF16 rounding of (1 + scale)
    out=torch.empty_like(x)
    _affine[(triton.cdiv(x.numel(),1024),)](x,s,shift,out,x.numel(),x.shape[-1],1024,enable_fp_fusion=False)
    return out

def residual(x,h,g):
    assert g.shape[0]==1 and x.dtype==h.dtype==g.dtype
    out=torch.empty_like(x)
    _residual[(triton.cdiv(x.numel(),1024),)](x,h,g,out,x.numel(),x.shape[-1],1024,enable_fp_fusion=False)
    return out

def install(m):
    prior=(m._pre,m._post)
    def pre(block,x,mod):
        assert len(x.layout)==1
        z=(block.modulation+mod).to(mod.dtype) if block.share_mod else block.adaLN_modulation(mod)
        shift,scale,gate,sm,scm,gm=z.chunk(6,dim=1)
        return x.replace(affine(block.norm1(x.feats),scale,shift)),gate,sm,scm,gm
    def post(block,x,h,gate,sm,scm,gm,cond):
        x=x.replace(residual(x.feats,h.feats,gate))
        x=x+block.cross_attn(x.replace(block.norm2(x.feats)),cond)
        h=x.replace(affine(block.norm3(x.feats),scm,sm))
        return x.replace(residual(x.feats,block.mlp(h).feats,gm))
    m._pre=pre;m._post=post
    return prior
