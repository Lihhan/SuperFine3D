from sc1_stable_pool import install
install()
from structure import StructureFlow
def is_hr(n):return n.startswith('blocks.') and '.self_attn.' in n and 'lora_' in n

class AblationFlow(StructureFlow):
 def __init__(self,*args,joint_hr=False,**kw):
  self._ablation_ready=False
  super().__init__(*args,**kw)
  self.joint_hr=joint_hr
  for n,p in self.named_parameters():
   if is_hr(n):p.requires_grad_(joint_hr)
  self._ablation_ready=True
  self.assert_trainable_contract()
 def assert_trainable_contract(self):
  if not getattr(self,'_ablation_ready',False):return super().assert_trainable_contract()
  actual={n for n,p in self.named_parameters() if p.requires_grad}
  expected={n for n,p in self.named_parameters() if (n.startswith('lr_') and n!='lr_null_embedding') or (self.joint_hr and is_hr(n))}
  assert actual==expected,(actual^expected)
  return dict(joint_hr=self.joint_hr,tensors=len(actual),parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))