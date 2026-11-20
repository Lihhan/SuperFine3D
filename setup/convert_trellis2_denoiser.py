"""Write the TRELLIS.2-4B 1024 shape flow model as the fp32 state dict the refiner training starts from
($SF3D_CKPT/trellis2/official_1024_denoiser.pt). usage: convert_trellis2_denoiser.py"""
import os, torch
from safetensors.torch import load_file
src = os.path.join(os.environ['SF3D_TRELLIS2_WEIGHTS'], 'ckpts', 'slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors')
out = os.path.join(os.environ['SF3D_CKPT'], 'trellis2', 'official_1024_denoiser.pt'); os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({k: v.float() for k, v in load_file(src).items()}, out); print(out)
