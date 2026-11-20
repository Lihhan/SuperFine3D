import torch
from torch import nn
def install_chunked_mlp(decoder,chunk):
    patched=[]
    for name,owner in decoder.named_modules():
        if owner.__class__.__name__!='SparseConvNeXtBlock3d':continue
        mlp=owner.mlp
        if not (isinstance(mlp,nn.Sequential) and len(mlp)==3 and isinstance(mlp[0],nn.Linear) and isinstance(mlp[1],nn.SiLU) and isinstance(mlp[2],nn.Linear)):
            raise ValueError('Unsupported decoder FFN: '+name)
        prior=mlp.forward
        def forward(x,_prior=prior):
            assert not torch.is_grad_enabled()
            if len(x)<=chunk:return _prior(x)
            result=None
            for first in range(0,len(x),chunk):
                value=_prior(x[first:first+chunk])
                if result is None:result=torch.empty((len(x),value.shape[-1]),device=value.device,dtype=value.dtype)
                result[first:first+len(value)].copy_(value)
            return result
        mlp.forward=forward;patched.append(name)
    if not patched:raise ValueError('No supported pointwise MLP found')
    return patched
