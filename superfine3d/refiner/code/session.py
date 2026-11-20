"""Same optimized HR kernel and CPU LR-KV cache, plus cached structure field."""
import inspect,textwrap,types
import runtime
from kernel_session import Session as BaseSession
from structure import FORWARD_SOURCE

def compiled():
 src=FORWARD_SOURCE
 start=src.index('        lr,lp=self._prepare_lr_input');end=src.index('        times=torch.cat',start);fixed=src[start:end]
 src=src[:start]+('        if self._speed.slot in self._speed.features:\n            lr,lp,evidence,coverage,anchor,structure=self._speed.features[self._speed.slot]\n        else:\n'+textwrap.indent(fixed,'    ')+'            self._speed.features[self._speed.slot]=(lr,lp,evidence,coverage,anchor,structure)\n')+src[end:]
 replacements={'plans,self.routing_stats=lr_plan(':'plans,self.routing_stats=self._speed.plan(','next_lr,lk,lv=self._lr_step(i,l,lm_,c,lp_)':'next_lr,lk,lv=self._speed.lr_step(i,l,lm_,c,lp_)','local=near_attention(q,k,v,prep,near)':'sorted_q=q[prep.perm]\n            local=self._speed.near(sorted_q,q,k,v,prep,near)','read=lr_attention(q,lk,lv,prep,plans).flatten(1)':'read=self._speed.read(sorted_q,q,lk,lv,prep,plans).flatten(1)'}
 for a,b in replacements.items():assert a in src,a;src=src.replace(a,b)
 ns=dict(runtime.original.__dict__);exec(compile(src,'<SC1-optimized>','exec'),ns);return ns['forward']
FORWARD=compiled()
class StructureSession(BaseSession):
 def forward(self,x,t,cond,**kwargs):
  assert not self.sorted_layout and not self.model.training and not __import__('torch').is_grad_enabled();assert x.coords is self.coords
  assert (self.coords._version,self.lr.coords._version,self.lr.feats._version)==self.versions
  self.slot=next((i for i,c in enumerate(self.conditions) if cond is c),None);assert self.slot is not None and cond._version==self.condition_versions[self.slot]
  self.calls['forward']+=1;object.__setattr__(self.model,'_speed',self)
  try:return FORWARD(self.model,x,t,cond,lr_cond=self.lr,hr_resolution=self.resolution,**kwargs)
  finally:del self.model._speed
