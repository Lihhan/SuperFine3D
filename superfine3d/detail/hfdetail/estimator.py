"""Tiled fine-detail normal estimation on the input image (port of est_normals.py, MODEL=nirne TILE=3 TILE_SIZE=768).
Spend compute on resolution: upsample the image `up` x (Lanczos), predict overlapping ts x ts tiles at the model's
native processing size (stride 3/4 ts), blend with a separable tent window, renormalise, area-downsample back.
Camera-frame predictions are mapped to the object frame with the camera axes and the model's sign convention
(NiRNE on these cameras: n_obj = -x r + y u + z (-fwd), found with sign_search.py).

Mesh-conditioned models (ft_nirne/ft_train.py COND=1) also take the current mesh's rendered normals (camera frame,
model convention) as a second input: the map is upsampled like the image (bilinear) and cut into the same tiles.

`batch` > 1 predicts several tiles per model call (same tiles, same blending): a speed-up whose numerical effect is
bounded by the fp16 network itself (tests/test_estimator.py)."""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os, sys, numpy as np, torch, torch.nn.functional as F_
from PIL import Image

NIRNE_SIGNS = (-1., 1., 1.)

def tile_positions(W, ts, offset=0):
    # est_normals.py produced a negative tile offset when the upsampled image was smaller than a tile (never hit
    # there: 3 x >= 256 px); refuse instead of silently mis-tiling.
    if W < ts: raise ValueError(f'upsampled image ({W}px) smaller than a tile ({ts}px): raise `up` or lower `ts`')
    st = ts * 3 // 4
    if offset:                                                        # shifted grid (TTA): seams fall elsewhere
        return sorted(set([0] + list(range(offset % st, W - ts, st)) + [W - ts]))
    pos = list(range(0, max(W - ts, 0) + 1, st))
    return pos if pos[-1] == W - ts else pos + [W - ts]

def tent_window(ts):
    t = np.minimum(np.arange(ts) + .5, ts - np.arange(ts) - .5); t = (t / t.max()).astype(np.float32)
    return (t[:, None] * t[None, :])[..., None] + 1e-3

def tiled_predict(pil, predict, up=3., ts=768, batch=1, cond=None, offset=0, fg=None):
    """pil: square RGB image. predict(list of PIL ts x ts) -> list of (ts,ts,3) float arrays (camera frame, [-1,1]);
    with cond (W0,W0,3 camera-frame normals) predict(tiles, cond_tiles) gets (3,ts,ts) float tensors as well.
    fg: optional (W0,W0) bool foreground; tiles whose footprint (+1 px) holds no foreground pixel are not predicted.
    Every image pixel a foreground pixel's value depends on lies only in tiles that are kept, so foreground values are
    unchanged; pixels covered by no kept tile get (0,0,1).
    Returns (W0,W0,3) float32 unit normals (camera frame)."""
    W0 = pil.size[0]; assert pil.size[1] == W0, 'square images only'
    W = int(round(W0 * up)); big = pil.resize((W, W), Image.Resampling.LANCZOS); pos = tile_positions(W, ts, offset)
    if cond is not None:
        assert cond.shape[:2] == (W0, W0), cond.shape
        cb = F_.interpolate(torch.as_tensor(cond).float().permute(2, 0, 1)[None], size=(W, W), mode='bilinear', align_corners=False)[0]
        cb = cb / cb.norm(dim=0, keepdim=True).clamp_min(1e-6)
    acc = np.zeros((W, W, 3), np.float32); wsum = np.zeros((W, W, 1), np.float32); win = tent_window(ts)
    xy = [(y, x) for y in pos for x in pos]
    if fg is not None:
        fg = np.asarray(fg, bool); assert fg.shape == (W0, W0), fg.shape
        def has_fg(y, x):
            a, b = max(int(np.floor(y / up)) - 1, 0), min(int(np.ceil((y + ts) / up)) + 1, W0)
            c, d = max(int(np.floor(x / up)) - 1, 0), min(int(np.ceil((x + ts) / up)) + 1, W0)
            return bool(fg[a:b, c:d].any())
        # The batched network's output depends on the batch size (fp16 kernel choice), not on the other tiles in the
        # batch (tests/test_estimator.py, real-model test). Each kept tile is therefore predicted in a batch of the
        # size it had in the full pass (full batches, then the remainder), padded with repeats whose output is dropped.
        n = len(xy); sizes = [batch if i < n - n % batch else n % batch for i in range(n)]
        groups = {}
        for p_, sz in zip(xy, sizes):
            if has_fg(*p_): groups.setdefault(sz, []).append(p_)
        plan = []
        for sz, ps in groups.items():
            for i in range(0, len(ps), sz):
                ch = ps[i:i + sz]; plan.append((ch, ch + [ch[0]] * (sz - len(ch))))
    else:
        plan = [(xy[i:i + batch], xy[i:i + batch]) for i in range(0, len(xy), batch)]
    for keep, ch in plan:
        tiles = [big.crop((x, y, x + ts, y + ts)) for y, x in ch]
        preds = predict(tiles) if cond is None else predict(tiles, [cb[:, y:y + ts, x:x + ts] for y, x in ch])
        for (y, x), p_ in zip(keep, preds[:len(keep)]): acc[y:y + ts, x:x + ts] += p_ * win; wsum[y:y + ts, x:x + ts] += win
    if fg is not None:
        empty = wsum[..., 0] == 0; acc[empty] = (0., 0., 1.); wsum[empty] = 1.
    n = acc / wsum; n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)
    return F_.interpolate(torch.from_numpy(n).permute(2, 0, 1)[None], size=(W0, W0), mode='area')[0].permute(1, 2, 0).numpy()

def tiled_predict_tta(pil, predict, up=3., ts=768, batch=1, cond=None, shifts=(0,), flip=False, fg=None):
    """Test-time averaging of tiled_predict over tile-grid offsets (pixels of the upsampled image) and an optional
    horizontal mirror (image and conditioning mirrored, camera-frame x negated, prediction mirrored back).
    Unit normals are averaged and renormalised. shifts=(0,), flip=False is tiled_predict itself."""
    outs = []
    for fl in ([False, True] if flip else [False]):
        im = pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if fl else pil; cd = cond
        g = None if fg is None else (np.asarray(fg)[:, ::-1].copy() if fl else fg)
        if fl and cond is not None:
            cd = torch.as_tensor(cond).flip(1).clone(); cd[..., 0] = -cd[..., 0]
        for o in shifts:
            n = tiled_predict(im, predict, up, ts, batch, cond=cd, offset=o, fg=g)
            if fl: n = n[:, ::-1].copy(); n[..., 0] = -n[..., 0]
            outs.append(n)
    if len(outs) == 1: return outs[0]
    n = np.mean(outs, 0); return (n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)).astype(np.float32)

def to_object(nc, cam, signs=NIRNE_SIGNS, device='cuda'):
    """Camera-frame normals (H,W,3) -> object frame with cam axes r, u, fwd (cam.json keys)."""
    nc = torch.from_numpy(np.ascontiguousarray(nc)).float().to(device)
    R = torch.tensor(cam['r'], device=device).float(); U = torch.tensor(cam['u'], device=device).float(); B = -torch.tensor(cam['fwd'], device=device).float()
    return signs[0] * nc[..., 0:1] * R + signs[1] * nc[..., 1:2] * U + signs[2] * nc[..., 2:3] * B

def to_camera(n_obj, cam, signs=NIRNE_SIGNS):
    """Inverse of to_object: object-frame normals (H,W,3) torch -> model camera frame."""
    R = torch.tensor(cam['r'], device=n_obj.device).float(); U = torch.tensor(cam['u'], device=n_obj.device).float(); B = -torch.tensor(cam['fwd'], device=n_obj.device).float()
    return torch.stack([signs[0] * (n_obj @ R), signs[1] * (n_obj @ U), signs[2] * (n_obj @ B)], -1)

def cond_map(n_obj, m, cam, signs=NIRNE_SIGNS):
    """Mesh normals (object frame) + coverage -> conditioning map (camera frame), uncovered pixels face the camera
    (as in training)."""
    c = to_camera(n_obj, cam, signs); c[~m] = torch.tensor([0., 0., 1.], device=c.device); return c

def target_map(n_obj, image_rgb, res):
    """Resize (area) to the height-field resolution, renormalise; mask = image foreground (premultiplied on black)."""
    n = F_.interpolate(n_obj.permute(2, 0, 1)[None], size=(res, res), mode='area')[0].permute(1, 2, 0); n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    fg = torch.from_numpy(image_rgb.max(-1) > 0.02).float().to(n.device)
    m = F_.interpolate(fg[None, None], size=(res, res), mode='area')[0, 0] > 0.99
    return n * m[..., None], m

def ckpt_fingerprint(path):
    """Identity of an estimator checkpoint for the AOT packages: size and SHA-256 of its first and last MiB."""
    import hashlib; n = os.path.getsize(path); h = hashlib.sha256(str(n).encode())
    with open(path, "rb") as f: h.update(f.read(1 << 20)); f.seek(max(n - (1 << 20), 0)); h.update(f.read())
    return h.hexdigest()[:32]
class NiRNE:
    """NiRNE (Hi3DGen yoso-normal-v1-8-1) with optional fine-tuned UNet/controlnet weights (ft_nirne/ft_train.py)."""
    def __init__(self, ckpt=None, weights=f'{_E["SF3D_NIRNE_WEIGHTS"]}', device='cuda'):
        import diffusers.models.controlnets.controlnet as _cn; sys.modules.setdefault('diffusers.models.controlnet', _cn)
        sys.path.insert(0, f'{_E["SF3D_STABLENORMAL"]}'); from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
        self.pipe = YOSONormalsPipeline.from_pretrained(weights, variant='fp16', torch_dtype=torch.float16, trust_remote_code=True, safety_checker=None).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if ckpt:
            sd = torch.load(ckpt, map_location=device)
            for k, mdl in (('unet', self.pipe.unet), ('controlnet', self.pipe.controlnet)):
                if k in sd: mdl.load_state_dict({a: b.half() for a, b in sd[k].items()})
        self.ckpt = ckpt
        if os.environ.get('EST_CUDNN') == '1': torch.backends.cudnn.benchmark = True   # quality profile: autotuned convolutions
    @torch.no_grad()
    def __call__(self, tiles, conds=None):
        """The YOSONormalsPipeline.__call__ steps (preprocess, VAE encode, controlnet + UNet at t_start, decode, unpad,
        resize back, normalise), run on a batch of tiles; the pipeline itself hard-codes a batch-1 text embedding."""
        p = self.pipe; dev, dt = p._execution_device, p.dtype
        img, padding, orig = p.image_processor.preprocess(tiles, p.default_processing_resolution, 'bilinear', dev, dt)
        lat, pred, _ = p.prepare_latents(img, None, None, 1, len(tiles))
        if conds is not None:                                         # mesh-conditioned: UNet starts from the coarse normals
            c = torch.stack([torch.as_tensor(x) for x in conds]).to(dev, dt)
            pred = p.vae.encode(c).latent_dist.mode() * p.vae.config.scaling_factor
        emb = torch.zeros(len(tiles), 257, 1024, device=dev, dtype=dt)
        dn, mid = p.controlnet(lat.detach(), p.t_start, encoder_hidden_states=emb, conditioning_scale=1.0, guess_mode=False, return_dict=False)
        out = p.unet(pred, p.t_start, encoder_hidden_states=emb, down_block_additional_residuals=dn, mid_block_additional_residual=mid, return_dict=False)[0]
        y = p.decode_prediction(out); y = p.image_processor.unpad_image(y, padding)
        y = p.image_processor.resize_antialias(y, orig, 'bilinear', is_aa=False); y = p.normalize_normals(y)
        y = p.image_processor.pt_to_numpy(y)
        return [y[i] for i in range(len(tiles))]
    def call_pipeline(self, tile):
        """The unmodified pipeline call (as est_normals.py), for the equivalence test."""
        torch.manual_seed(0); return self.pipe(tile, match_input_resolution=True).prediction[0]

def foreground(image_rgb_uint8):
    """Foreground as target_map defines it (premultiplied image, max channel > 0.02)."""
    return (image_rgb_uint8.astype(np.float32) / 255).max(-1) > 0.02

def estimate(image_rgb_uint8, cam, res, model, up=3., ts=768, batch=1, signs=NIRNE_SIGNS, cond=None, shifts=(0,), flip=False,
             skip_background=True):
    """Full step: image (H,W,3 uint8, premultiplied on black) + cam -> target normals (res,res,3) object frame, mask.
    cond: optional (W0,W0,3) camera-frame conditioning map (cond_map) for mesh-conditioned models.
    shifts / flip: test-time averaging (tiled_predict_tta); defaults = a single pass.
    skip_background: do not predict tiles without foreground (target values inside the mask are unchanged)."""
    pil = Image.fromarray(image_rgb_uint8); fg = foreground(image_rgb_uint8) if skip_background else None
    nc = tiled_predict_tta(pil, model, up, ts, batch, cond=None if cond is None else cond.cpu(), shifts=shifts, flip=flip, fg=fg)
    return target_map(to_object(nc, cam, signs), image_rgb_uint8.astype(np.float32) / 255, res)
