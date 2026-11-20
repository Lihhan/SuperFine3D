"""Image-space normal fidelity on Toys4k with each method's OWN image-only camera (the supp protocol, Table-1 column).
GT normals: the GT mesh rendered with the TRUE input camera of the Toys4k render (prepare.json: orthographic, yaw/pitch,
ortho_scale 1.55, 1024 px; toys_benchmark_prepare.py conventions) on the pixel grid of the input crop (the fitted preprocess box of
the input, W0 = crop_input.py size). Each result: rendered with its own fitted camera (cam_refined.json P on the same box) on the
same grid. Both as camera-frame, camera-facing face normals (x right, y up, z towards the camera). On pixels covered by both: mean
angle, 8-px low-passed angle, band [a,b] in {[1,4],[4,16],[16,64]} px correlation and amplitude ratio of L_a(n)-L_b(n)
(relief24/eval_gt2.py definitions). Also the IoU of each render's coverage with the input alpha (camera check).
usage: fid_t4k_img.py KEY OUT.json LABEL=MESH@CAMDIR [...]   (CAMDIR: cam_fit.json with box/size + cam_refined.json)
Opt-in env (defaults unchanged): T4K_ROOT = root of the Toys4k render (prepare.json, input.png, GT; e.g. the 4096 re-render);
FID_BOX = cam_fit.json that defines the common crop box (e.g. one fitted on the 4096 input). With FID_BOX each result's camera is
exactly re-expressed on that box (P rows 0/1 += box offset/scale; a fit on the 1024 render of the same view is the same camera up to
resolution rounding of its box), instead of the 1-px box-equality assert."""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys, os, json, math, time, numpy as np, torch, torch.nn.functional as Fn
E = f'{_E["SF3D_TOYS4K_EVAL"]}'
T4R = os.environ.get('T4K_ROOT', E)
sys.path[:0] = [f'{_E["SF3D_ROOT"]}/superfine3d/detail', f'{_E["SF3D_ROOT"]}/superfine3d/detail/whole', E + '/code', f'{_E["SF3D_ROOT"]}/third_party/site-packages', f'{_E["SF3D_ROOT"]}/third_party/site-packages']
import toys_metrics_v4 as v4
from hfdetail import raster, camera
from PIL import Image
torch.set_num_threads(4); dev = 'cuda'
K, OUT = sys.argv[1:3]; items = []
for a in sys.argv[3:]:
    lab, rest = a.split('=', 1); mesh, cd = rest.rsplit('@', 1); items.append((lab, mesh, cd))
prep = json.load(open(f'{T4R}/{K}/input_data/prepare.json')); tc = prep['camera']; SZ = int(prep.get('input_pixels', 1024)); assert tc['type'] == 'ORTHO'
y, p = math.radians(tc['yaw_deg']), math.radians(tc['pitch_deg'])
d = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)]); f = -d
up = np.array([0., 0., 1.]); ut = up - (up @ f) * f; ut /= np.linalg.norm(ut); rt = np.cross(f, ut); W = tc['scale_in_object_units']
FB = os.environ.get('FID_BOX'); fit0 = json.load(open(FB or f'{items[0][2]}/cam_fit.json')); sc = SZ / max(fit0['size']); box = [b * sc for b in fit0['box']]; side = box[2] - box[0]
W0 = int(min(4096, round(side))); W0 -= W0 % 8
def gt_render():
    g = v4.load(f'{T4R}/{K}/input_data/GT_normalized.ply'); V = torch.as_tensor(np.asarray(g.vertices), dtype=torch.float64, device=dev)
    F = torch.as_tensor(np.asarray(g.faces), dtype=torch.long, device=dev)
    xi = ((V @ torch.as_tensor(rt, device=dev)) / W + .5) * SZ; yi = (.5 - (V @ torch.as_tensor(ut, device=dev)) / W) * SZ   # input pixels
    uu = (xi - box[0]) / side; vv = (yi - box[1]) / side; z = V @ torch.as_tensor(f, device=dev)
    clip = torch.stack([2 * uu - 1, 2 * vv - 1, z / 4., torch.ones_like(z)], 1)[None].float().contiguous()
    tri = raster.tri_ids(raster.rasterize(clip, F, W0)); N, m = raster.normal_map(V.float(), F, tri, f)
    R = torch.as_tensor(np.stack([rt, ut, -f]), dtype=torch.float32, device=dev); return N @ R.T, m
def lp(N, m, k):
    a = Fn.avg_pool2d((N * m[..., None]).permute(2, 0, 1)[None], 2 * (k // 2) + 1, 1, k // 2, count_include_pad=False)[0].permute(1, 2, 0)
    w = Fn.avg_pool2d(m[None, None].float(), 2 * (k // 2) + 1, 1, k // 2, count_include_pad=False)[0, 0]
    return a / w.clamp_min(1e-6)[..., None]
G, Gm = gt_render()
alpha = Image.open(f'{T4R}/{K}/input_data/input.png').convert('RGBA').crop(tuple(int(round(b)) for b in box)).resize((W0, W0), Image.Resampling.NEAREST)
At = torch.as_tensor(np.asarray(alpha)[..., 3] > 127, device=dev); iou = lambda m: float((At & m).sum() / (At | m).sum())
res = dict(asset=K, protocol="own image-only fitted camera per result; GT with the true input camera; input-crop grid", box=box, W0=W0,
           true_camera=tc, gt_iou_vs_alpha=iou(Gm), methods={})
if T4R != E or FB: res.update(t4k_root=T4R, box_from=FB, input_pixels=SZ)
print(K, 'W0', W0, 'GT coverage IoU vs input alpha %.4f' % iou(Gm), flush=True)
for lab, mesh, cd in items:
    t0 = time.time(); fit = json.load(open(f'{cd}/cam_fit.json')); cam = json.load(open(f'{cd}/cam_refined.json'))
    bi = [b * SZ / max(fit['size']) for b in fit['box']]; P_ = cam['P']
    if FB:   # same input pixel, re-expressed on the common box: u' = (bi0 - box0)/side + u * side_i/side
        a_ = (bi[2] - bi[0]) / side; P0 = np.asarray(cam['P'], dtype=np.float64)
        assert abs(a_ - 1) < .02 and max(abs(bi[0] - box[0]), abs(bi[1] - box[1])) < .02 * side, f'{lab}: crop box differs beyond resolution rounding'
        P_ = np.stack([a_ * P0[0] + (bi[0] - box[0]) / side * P0[2], a_ * P0[1] + (bi[1] - box[1]) / side * P0[2], P0[2]]).tolist()
    else: assert np.allclose(bi, box, atol=1.), f'{lab}: crop box differs'
    m_ = v4.load(mesh); v = torch.as_tensor(np.asarray(m_.vertices), dtype=torch.float32, device=dev); F = torch.as_tensor(np.asarray(m_.faces), dtype=torch.long, device=dev); del m_
    tri = raster.tri_ids(raster.rasterize(camera.clip_coords(v, P_), F, W0)); N, m = raster.normal_map(v, F, tri, cam['fwd']); del v, F, tri
    r_, u_, f_ = [torch.tensor(cam[k_], device=dev).float() for k_ in ('r', 'u', 'fwd')]; Nc = torch.stack([N @ r_, N @ u_, N @ (-f_)], -1)
    b = m & Gm; ang = lambda a_, c_: float(torch.rad2deg(torch.acos((a_ * c_).sum(-1).clamp(-1, 1)))[b].mean())
    A = lp(Nc, m, 8); A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-9); B = lp(G, Gm, 8); B = B / B.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    row = dict(mesh=mesh, camdir=cd, box_item=bi, ang=ang(Nc, G), ang_lo=ang(A, B), iou_vs_alpha=iou(m), cam_iou=cam.get('iou'))
    dg = torch.rad2deg(torch.acos((Nc * G).sum(-1).clamp(-1, 1)))[b]   # per-pixel angle on the common pixels (Pixal3D Table-1 style fields, additive)
    row.update(ang_med=float(dg.median()), acc11_25=float((dg < 11.25).float().mean()), acc22_5=float((dg < 22.5).float().mean()), acc30=float((dg < 30).float().mean())); del dg
    for lo, hi in ((1, 4), (4, 16), (16, 64)):
        dA = lp(Nc, m, lo) - lp(Nc, m, hi); dB = lp(G, Gm, lo) - lp(G, Gm, hi); x = dA[b]; yv = dB[b]
        row[f'corr{lo}-{hi}'] = float((x * yv).sum() / (x.norm() * yv.norm()).clamp_min(1e-12)); row[f'E{lo}-{hi}'] = float(x.norm() / yv.norm().clamp_min(1e-12))
    row['seconds'] = time.time() - t0; res['methods'][lab] = row
    print('%-12s' % lab, '  '.join('%s %.3f' % (k_, v_) for k_, v_ in row.items() if isinstance(v_, float)), flush=True)
    del N, m, Nc; torch.cuda.empty_cache()
json.dump(res, open(OUT, 'w'), indent=1)
