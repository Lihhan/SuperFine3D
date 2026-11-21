"""AOTInductor packages of the detail-estimator networks that the pipeline runs at every level (the mesh-conditioned
re-estimation): condition VAE encode, UNet at t_start (with the cached ControlNet residuals), VAE decode. Static batch 8, tile
768 px, weights of the mesh-conditioned (RGB) estimator baked in. Each package is checked against the eager network.
Writes $SF3D_CACHE/aoti/{enc,unet,dec}.pt2 and meta.json (GPU name, torch version, checkpoint fingerprint); the pipeline uses
them only on a matching GPU / torch / checkpoint. usage: python setup/build_estimator_aot.py   (after source setup/env.sh)"""
import sys, os, json, time, torch, torch.nn as nn
sys.path.insert(0, f'{os.environ["SF3D_ROOT"]}/superfine3d/detail')
from hfdetail import estimator
OUT = f'{os.environ["SF3D_CACHE"]}/aoti'; CK = f'{os.environ["SF3D_CKPT"]}/estimator/rgb/ckpt_last.pt'; os.makedirs(OUT, exist_ok=True)
torch.backends.cudnn.benchmark = True
os.path.exists(f'{OUT}/meta.json') and os.remove(f'{OUT}/meta.json')        # packages are used only once a complete build passed
m = estimator.NiRNE(CK); p = m.pipe; dev, dt = p._execution_device, p.dtype; vae = p.vae; T0 = p.t_start
assert vae.config.latent_channels == 4
class Enc(nn.Module):                     # = vae.encode(x).latent_dist.mode()
    def __init__(s): super().__init__(); s.e, s.q = vae.encoder, vae.quant_conv
    def forward(s, x): return s.q(s.e(x))[:, :4]
class Dec(nn.Module):                     # = vae.decode(z).sample
    def __init__(s): super().__init__(); s.pq, s.d = vae.post_quant_conv, vae.decoder
    def forward(s, z): return s.d(s.pq(z))
class UNet(nn.Module):
    def __init__(s): super().__init__(); s.u = p.unet
    def forward(s, sample, emb, mid, *down):
        return s.u(sample, T0, encoder_hidden_states=emb, down_block_additional_residuals=down, mid_block_additional_residual=mid, return_dict=False)[0]
torch.manual_seed(0); B = 8
x = torch.randn(B, 3, 768, 768, device=dev, dtype=dt); x = x / x.norm(dim=1, keepdim=True)   # unit normal maps, as the conditions
x[:2] = 0; x[:2, 2] = 1                                                                      # and flat (background-like) ones
emb = torch.zeros(B, 257, 1024, device=dev, dtype=dt); sc = vae.config.scaling_factor
with torch.no_grad():
    dn, mid = p.controlnet(torch.randn(B, 4, 96, 96, device=dev, dtype=dt), T0, encoder_hidden_states=emb, conditioning_scale=1.0, guess_mode=False, return_dict=False)
    z = Enc()(x); u = UNet()(z * sc, emb, mid, *dn); y = Dec()(u / sc)                       # eager chain (reference)
# ATen convolutions and matmuls only: Triton templates chosen by autotuning overflowed in fp16 on real inputs (gains come from fusion)
cfg = {'max_autotune': True, 'max_autotune_gemm_backends': 'ATEN', 'max_autotune_conv_backends': 'ATEN'}
mods = (('enc', Enc(), (x,)), ('unet', UNet(), (z * sc, emb, mid, *dn)), ('dec', Dec(), (u / sc,)))
F = {}
for name, mod, args in mods:
    path = f'{OUT}/{name}.pt2'; t = time.time()
    with torch.no_grad():
        ep = torch.export.export(mod.eval(), args, strict=False)
        torch._inductor.aoti_compile_and_package(ep, package_path=path, inductor_configs=cfg)
    F[name] = torch._inductor.aoti_load_package(path); print(f'{name}: compiled in {time.time() - t:.0f} s', flush=True)
one = lambda r: r[0] if isinstance(r, (list, tuple)) else r
with torch.no_grad():                                                                        # the AOT chain against the eager chain
    za = one(F['enc'](x)); ua = one(F['unet'](za * sc, emb, mid, *dn)); ya = one(F['dec'](ua / sc))
ok = True
for name, a_, r_ in (('enc', za, z), ('unet', ua, u), ('dec', ya, y)):
    fin = bool(torch.isfinite(a_).all()); d = (a_.float() - r_.float()).abs()
    rel = d.mean().item() / max(r_.float().abs().mean().item(), 1e-6); ok &= fin and rel < 0.02
    print(f'{name}: finite {fin}; max |diff| {d.max().item():.3g}, mean {d.mean().item():.3g} (relative {rel:.2%}) vs the eager chain', flush=True)
if not ok:
    for f in ('enc.pt2', 'unet.pt2', 'dec.pt2', 'meta.json'): os.path.exists(f'{OUT}/{f}') and os.remove(f'{OUT}/{f}')
    sys.exit('AOT packages rejected (non-finite or inaccurate); the pipeline will use the eager networks')
json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, unet_ckpt=estimator.ckpt_fingerprint(CK), batch=8, tile=768),
          open(f'{OUT}/meta.json', 'w'), indent=1)
print('ESTIMATOR_AOT_READY', OUT)
