"""Process-local functional LayerNorm guard for very large decoder tensors."""
import importlib,torch
class FunctionalProxy:
 def __init__(self,original):self.original=original
 def __getattr__(self,name):return getattr(self.original,name)
 def layer_norm(self,x,normalized_shape,weight=None,bias=None,eps=1e-5):
  fn=self.original.layer_norm
  if torch.is_grad_enabled() or x.ndim!=2 or len(normalized_shape)!=1 or x.shape[0]<1048576:
   return fn(x,normalized_shape,weight,bias,eps)
  out=torch.empty_like(x)
  for start in range(0,len(x),65536):
   out[start:start+65536]=fn(x[start:start+65536],normalized_shape,weight,bias,eps)
  print('FINAL_FUNCTIONAL_LAYERNORM_CHUNKED',list(x.shape),str(x.dtype),flush=True)
  return out
def install(dec):
 module=importlib.import_module('trellis2.models.sc_vaes.sparse_unet_vae')
 if not isinstance(module.F,FunctionalProxy):module.F=FunctionalProxy(module.F)
 print('FUNCTIONAL_LAYERNORM_GUARD',module.__file__,flush=True)
