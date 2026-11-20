"""Exact speed-up of portable_pipeline.decode (native1024_ours_20261026/decode_code) for an in-process pipeline:
  * the shape decoder is built and set up once (same named_pipe + setup_decoder + hooks) and reused at every level
    (parked on the CPU between levels), instead of being reloaded from disk per level;
  * the decoded mesh is returned in memory instead of being exported to mesh_{res}.ply by trimesh and read back.
The decoder forward is the same call on the same inputs (same dtype cast, set_resolution, low_vram, inference_mode).
usage: decfast.decode(pp, res, coords_np, feats_np) -> (vertices float32 cuda, faces int64 cuda)"""
import time, numpy as np, torch

_DEC = {}


def _build(pp):
    import importlib
    pipe = pp.named_pipe(['shape_slat_decoder']); decoder = pipe.models['shape_slat_decoder']; dtype = pp.setup_decoder(decoder)
    pp.install_decoder_lowmem_patch()
    import trellis2.models.sc_vaes.sparse_unet_vae as vae
    assert vae.SparseUnetVaeDecoder.forward.__globals__['F'].__class__.__name__ == 'FunctionalProxy'

    def final_norm_gate(module, args):
        x = args[0]; n = min(8192, len(x.feats)); idx = torch.arange(n, device=x.feats.device, dtype=torch.int64) * (len(x.feats) - 1) // max(n - 1, 1)
        f = x.feats[idx].float(); rms = f.square().mean(-1).sqrt()
        assert torch.isfinite(f).all() and bool((rms > .85).all()) and bool((rms < 1.15).all()), 'Invalid final normalized features'
    decoder.output_layer.register_forward_pre_hook(final_norm_gate)
    decoder.eval().requires_grad_(False)
    conv_impl = importlib.import_module('flex_gemm.ops.spconv.submanifold_conv3d')
    original_cache = conv_impl.SubMConv3dFunction._compute_neighbor_cache
    if not getattr(original_cache, '_decfast_checked', False):
        def checked_cache(coords, shape, kernel_size, dilation, needs_grad):
            assert not needs_grad, 'Inference unexpectedly requested backward cache'
            return original_cache(coords, shape, kernel_size, dilation, False)
        checked_cache._decfast_checked = True
        conv_impl.SubMConv3dFunction._compute_neighbor_cache = staticmethod(checked_cache)
    _DEC['decoder'] = decoder; _DEC['dtype'] = dtype


def decode(pp, res, coords, feats):
    if 'decoder' not in _DEC:
        _build(pp)
    decoder, dtype = _DEC['decoder'], _DEC['dtype']
    from trellis2.modules.sparse import SparseTensor
    z = SparseTensor(feats=torch.from_numpy(np.ascontiguousarray(feats)).cuda(), coords=torch.from_numpy(np.ascontiguousarray(coords)).cuda())
    z = z.replace(z.feats.to(dtype)); decoder.set_resolution(res); decoder.low_vram = True; decoder.cuda()
    pp.clean()
    z = z.replace(z.feats.detach())
    with torch.inference_mode():
        decoded = decoder(z, return_subs=False)[0]
    torch.cuda.synchronize()
    v = decoded.vertices.float().cuda().clone(); F = decoded.faces.long().cuda().clone()   # mesh may be assembled on the CPU (DECODE_CPU_MESH_ASSEMBLY)
    del decoded, z
    decoder.cpu(); pp.clean()
    return v, F
