"""TF8 ablations: continuous sparse anchor, optionally independent branch gates."""
import torch
from torch import nn
from torch.nn import functional as F
from tf7.model import ElasticTF7SLatFlowModel
from tf7.anchor import AnchoredResidual

class ContinuousAnchor(nn.Module):
    def __init__(self,channels,width):
        super().__init__()
        # Evaluate once on LR tokens, never an MLP per HR token per layer.
        self.project=nn.Sequential(nn.Linear(channels,width),nn.SiLU(),nn.Linear(width,width))
    def encode(self,sampled,plan):
        # Normalized read multiplied by continuous coverage = unnormalized
        # trilinear field with absent lattice values set to zero.
        # No sawtooth fractional coordinates or discrete neighbor-count input.
        return sampled*plan.mass.to(sampled.dtype)

class DualGate(AnchoredResidual):
    def __init__(self,channels,width):
        super().__init__(channels,width)
        self.anchor_gate=nn.Linear(width+2,1)
        self.anchor_hr_gate=nn.Linear(channels,1,bias=False)
        nn.init.zeros_(self.anchor_gate.weight);nn.init.zeros_(self.anchor_gate.bias)
        nn.init.zeros_(self.anchor_hr_gate.weight)
    def forward(self,read,evidence,coverage,time,anchor,hr):
        inputs=torch.cat((evidence,coverage,time),-1)
        g=torch.sigmoid(self.gate(inputs)+self.hr_gate(hr))
        # Relative anchor gain initially 1; retains the baseline context gate
        # and shared up projection, avoiding a second large HR projection.
        ga=2*torch.sigmoid(self.anchor_gate(inputs)+self.anchor_hr_gate(hr))
        low=self.down(read)
        if self.anchor_enabled:low=low+ga*anchor
        return g*self.up(F.silu(low))

class ElasticTF8SLatFlowModel(ElasticTF7SLatFlowModel):
    def __init__(self,*args,tf8_variant='continuous',**kwargs):
        if tf8_variant not in ('continuous','continuous_dual'):raise ValueError(tf8_variant)
        super().__init__(*args,**kwargs)
        if self.tf2_stage!='lr_match':raise ValueError('TF8 requires frozen-HR LR training')
        width=kwargs.get('tf5_width',128)
        self.lr_anchor_encoder=ContinuousAnchor(kwargs['in_channels'],width)
        if tf8_variant=='continuous_dual':
            self.lr_gated_residual=nn.ModuleList([DualGate(self.model_channels,width) for _ in self.blocks])
        self.tf8_variant=tf8_variant
        self.assert_trainable_contract()
