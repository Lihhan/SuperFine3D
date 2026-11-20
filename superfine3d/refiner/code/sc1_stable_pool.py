"""Candidate: deterministic, differentiable grouped sum for the structure field."""
import inspect,textwrap,torch
from structure import StructureField
def install():
 source=textwrap.dedent(inspect.getsource(StructureField.forward))
 old="sums=torch.zeros(edge**3,self.width,device=f.device,dtype=torch.float32).index_add(0,key,f[sl].float());count=torch.zeros(edge**3,1,device=f.device).index_add(0,key,torch.ones(len(key),1,device=f.device))"
 new="""order=torch.argsort(key,stable=True)
  unique,lengths=torch.unique_consecutive(key[order],return_counts=True)
  sums=torch.zeros(edge**3,self.width,device=f.device,dtype=torch.float32)
  count=torch.zeros(edge**3,1,device=f.device)
  if len(unique):
   reduced=torch.segment_reduce(f[sl][order].float(),reduce='sum',lengths=lengths)
   sums=sums.index_copy(0,unique,reduced)
   count=count.index_copy(0,unique,lengths[:,None].float())"""
 assert old in source
 source=source.replace(old,new)
 namespace=dict(StructureField.forward.__globals__)
 exec(compile(source,'<SC1-stable-pool>','exec'),namespace)
 StructureField.forward=namespace['forward']
