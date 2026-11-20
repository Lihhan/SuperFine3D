import types,torch
import torch.nn.functional as F

def chunked_forward(self,x):
 if torch.is_grad_enabled():raise RuntimeError('Decoder chunk norm is inference only')
 if x.ndim!=2 or len(self.normalized_shape)!=1 or x.shape[-1]!=self.normalized_shape[0]:
  return self._original_forward(x)
 # Normalize each row across channels; rows are independent.
 if len(x)<65536:return self._original_forward(x)
 y=torch.empty_like(x)
 w=self.weight.float() if self.weight is not None else None
 b=self.bias.float() if self.bias is not None else None
 for start in range(0,len(x),65536):
  end=min(start+65536,len(x))
  y[start:end]=F.layer_norm(x[start:end].float(),self.normalized_shape,w,b,self.eps).to(x.dtype)
 return y

def install(decoder):
 count=0
 for m in decoder.modules():
  if type(m).__name__=='LayerNorm32':
   m._original_forward=m.forward;m.forward=types.MethodType(chunked_forward,m);count+=1
 print('CHUNKED_DECODER_LAYERNORM',count,flush=True)

def check():
 from trellis2.modules.norm import LayerNorm32
 torch.set_grad_enabled(False)
 for dtype in [torch.float16,torch.bfloat16,torch.float32]:
  m=LayerNorm32(32).cuda()
  m.weight.copy_(torch.randn_like(m.weight));m.bias.copy_(torch.randn_like(m.bias))
  x=torch.randn(70003,32,device='cuda',dtype=dtype)
  expected=m(x);m._original_forward=m.forward
  actual=chunked_forward(m,x)
  torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 print('CHUNKED_NORM_EXACT_GATE_PASS',flush=True)
if __name__=='__main__':check()
