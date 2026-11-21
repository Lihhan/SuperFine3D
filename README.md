<div align="center">

# SuperFine3D

### Efficient Global-Local Recursive Refinement for Extremely Detailed 3D Geometry Generation

<a href="https://lihhan.github.io/SuperFine3D-page/"><img alt="Project page" src="https://img.shields.io/badge/Project-Page-ff9a4a?style=for-the-badge"></a>
<img alt="Paper" src="https://img.shields.io/badge/Paper-coming%20soon-555?style=for-the-badge">

**A single image &rarr; 4096³ geometry in about three minutes on one GPU.**

</div>

<p align="center"><img src="docs/static/img/fig1_teaser.jpg" width="100%" alt="High-resolution geometry with fine detail and efficient inference"></p>

<p align="center"><sub><b>High-resolution geometry with fine detail and efficient inference.</b> Surface normals of GT, TRELLIS.2,
Hunyuan3D-2.1, Pixal3D, Direct3D-S2, UltraShape and ours, with zooms A and B; detail quality vs. end-to-end time; latency of
one attention layer and one DiT block vs. high-resolution tokens.</sub></p>

| <h3>4096³</h3>decoder grid (sparse 256³ latent) | <h3>176 s</h3>image &rarr; 4096³ mesh | <h3>9.4&times;</h3>faster than full-attention recursion, 23.0% less memory | <h3>525&times;</h3>faster attention layer at 708k tokens | <h3>28&times;</h3>faster DiT block at 708k tokens |
|:---:|:---:|:---:|:---:|:---:|

## Overview

SuperFine3D learns a **shared refinement operator** from paired coarse-to-fine representations and **recursively reuses it**
to synthesize geometry beyond the training resolutions: trained only on 512³/1024³ TRELLIS.2 latents, it refines
1024³ &rarr; 2048³ &rarr; 4096³, a sparse 256³ latent lattice.

<p align="center"><img src="docs/static/img/fig2_pipeline.jpg" width="100%" alt="Overview of SuperFine3D"></p>

<p align="center"><sub><b>Overview.</b> A pretrained generator produces a coarse 512³ shape, from which the input camera is
estimated. At every level, detail normals estimated from the input image are injected into the current mesh; the corrected
mesh is encoded at twice the resolution, partially noised, and refined by the shared refinement network conditioned on the
preceding latent and the image (bottom).</sub></p>

<p align="center"><img src="docs/static/img/fig4_beyond_resolution.jpg" width="100%" alt="Generation beyond training resolutions"></p>

<p align="center"><sub><b>Recursive refinement, level by level.</b> Generation beyond training resolutions: the same operator, trained on 512³–1024³ pairs, refines
1024³ &rarr; 2048³ &rarr; 4096³.</sub></p>

- **Spatial-cell attention** restricts high-resolution interactions to local spatial neighborhoods while keeping bounded,
  boundary-prioritized communication across neighboring cells. All cells run in one indexed launch of a FlashAttention-2 varlen
  kernel with fused 3D RoPE (`third_party/flash_attn_3d_rope_fa2`).
- **Spatially anchored cross-scale guidance** uses globally contextualized coarse geometry to guide local detail synthesis
  and preserve structural consistency.
- **Geometry-aware training objectives** keep stable structures unchanged and introduce fine-grained surface detail.
- **Image-faithful detail:** a fine-tuned one-step diffusion normal estimator, run on tiles of the upsampled input, drives a
  band-limited height field; a mesh-conditioned variant keeps albedo out of the geometry.

## Results

<p align="center"><img src="docs/static/img/fig5_comparison.jpg" width="100%" alt="Comparison of fine geometric details"></p>

<p align="center"><sub><b>Comparison of fine geometric details.</b> Inputs are 4096 px images generated with Nano Banana Pro,
identical for all methods.</sub></p>

End-to-end inference at 4096³-class resolution on the same input (paper Table 2):

| Method | Time (s) ↓ | Peak memory (GiB) ↓ |
|---|---|---|
| TRELLIS.2-Rec. (full attention) | 1654.14 | 63.07 |
| UltraShape | 2214.71 | – |
| **SuperFine3D** | **176.19** | **48.54** |

SuperFine3D is 9.4&times; faster than TRELLIS.2-Rec. with 23.0% less peak GPU memory, and about 12.6&times; faster than
UltraShape.

<details>
<summary><b>More results</b> (off-view detail, textured inputs, input resolution, two views, ablations)</summary>

<p align="center"><img src="docs/static/img/supp_more_comparisons.jpg" width="100%" alt="Additional comparisons"><br>
<sub><b>Additional comparisons</b>: carved censer, guardian lion and pangolin (generated images), and ornate wooden door.</sub></p>

<p align="center"><img src="docs/static/img/supp_offview.jpg" width="80%" alt="Detail holds off the input view"><br>
<sub><b>Detail holds off the input view</b>: close-ups rendered with the object turned by 30&deg;.</sub></p>

<p align="center"><img src="docs/static/img/supp_textured.jpg" width="100%" alt="Image appearance is not carved into the surface"><br>
<sub><b>Image appearance is not carved into the surface</b>: carved feathers, chased medallions and carved scales are recovered;
grain stripes and painted checks do not become relief.</sub></p>

<p align="center"><img src="docs/static/img/fig6_two_view.jpg" width="100%" alt="Novel-view comparison"><br>
<sub><b>Novel-view comparison</b>: a single view stretches the dragon sideways; two views recover its pose.
Photos: The Cleveland Museum of Art, Open Access (CC0).</sub></p>

<p align="center"><img src="docs/static/img/fig3_training.jpg" width="100%" alt="Training objective"><br>
<sub><b>Training objective</b>: L<sub>keep</sub>, L<sub>detail</sub> (render space) and L<sub>edge</sub>, L<sub>plane</sub>
(decoder space) are added to the flow-matching loss.</sub></p>

</details>

The [project page](https://lihhan.github.io/SuperFine3D-page/) (`docs/`) has an interactive 3D viewer: SuperFine3D side by
side with each baseline under one camera, click-to-zoom, and hotspots that load full-resolution geometry; plus the
supplementary video.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/infer.sh` | image → mesh, Table-1 configuration |
| `assets/examples/` | 4096² RGBA example inputs (pangolin, nautilus) |
| `superfine3d/detail/bench/bench_t1_body.py` | the end-to-end pipeline (stage list in its header) |
| `superfine3d/refiner/code/` | refiner model (`ablation_model.py`, `structure.py`, stage packages `tf3`–`tf8`), spatial-cell attention session (`session.py`, `kernel_session.py`, `tf6_2/boundary_attention.py`), LR guidance (`custom_attention/lr_guidance.py`) |
| `superfine3d/refiner/generate.py` | refinement driver (support construction, SDEdit sampling per level) |
| `superfine3d/refiner/decode_code/` | low-memory decoding of 2048³/4096³ latents |
| `superfine3d/detail/bench/ours_fast.py`, `ours_triton.py` | fused DiT kernels: LoRA merged into QKV/out, Triton RMSNorm+RoPE, LayerNorm+adaLN, indexed LR reads |
| `third_party/flash_attn_3d_rope_fa2/` | FlashAttention-2 varlen fork with fused 3D RoPE and row-indexed K/V |
| `superfine3d/detail/hfdetail/`, `superfine3d/detail/whole/inject_lib.py` | detail estimation on tiles, band-limited targets, evidence gate, height-field optimisation, hole capping |
| `superfine3d/detail/fit_input_camera.py`, `superfine3d/detail/whole/camera_refine.py` | input-camera fit (silhouette) and appearance refinement |
| `superfine3d/detail/fig5/pixal512_worker.py`, `superfine3d/start/cand_start.py` | Pixal3D 512³ start and its placement in the input frame |
| `superfine3d/encode/pairs.py`, `superfine3d/detail/hfdetail/latent_io.py`, `parvox.py` | mesh → latent encoding (parallel voxelisation) |
| `training/refiner/` | the five refiner training stages (`run_stage.sh`, `configs/`) |
| `training/estimator/` | detail-estimator fine-tuning (`train.sh`, `ft_train.py`) and its data builders |
| `evaluation/` | CD-L1 / F-score / DQ (`score_geometry.py`) and image-space normal fidelity (`fid_t4k_img.py`) |
| `setup/` | environment (`env.sh`), CUDA extension and kernel builds, TRELLIS.2 fork preparation, config rendering |

## Installation

Tested with Python 3.10, CUDA 12.8 and PyTorch 2.9.0 on RTX PRO 6000 Blackwell GPUs (sm_120, 96 GB).

1. **Python environment** (a fresh Python 3.10 venv or conda env)
   ```bash
   python -m pip install -U pip        # an old pip (e.g. 23.0) rejects some wheels' metadata
   pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
2. **TRELLIS.2** at commit `75fbf01` and its CUDA extensions (`nvdiffrast` v0.4.0, `CuMesh`, `FlexGEMM`, `o-voxel`,
   `flash-attn==2.7.3`). The extensions are built as wheels into `third_party/site-packages`, which comes first on the
   path of SuperFine3D and of the Pixal3D worker, so both environments share them.
   ```bash
   git clone https://github.com/microsoft/TRELLIS.2.git third_party/TRELLIS.2
   git -C third_party/TRELLIS.2 checkout 75fbf01
   bash setup/build_extensions.sh      # sources in .cache/ext_src (pinned revisions), wheels -> third_party/site-packages
   ```
   Build notes (also for step 3):
   - CUDA 12.8's `nvcc` accepts host compilers up to gcc 14. With a newer system gcc, point `CC`, `CXX` and
     `CUDAHOSTCXX` to a gcc &le; 14 (e.g. conda's `gxx_linux-64=14`).
   - The extensions include cuSPARSE / cuBLAS headers through PyTorch. If your CUDA toolkit does not ship them, use the
     headers of PyTorch's `nvidia-*` wheels:
     `export CPATH=$(python -c "import nvidia, glob, os; print(':'.join(glob.glob(os.path.join(nvidia.__path__[0], '*', 'include'))))")`
   - The default target is sm_120; set `TORCH_CUDA_ARCH_LIST` / `FLASH_ATTN_CUDA_ARCHS` for other GPUs.
   - `setup/flex_autotune.json` holds the FlexGEMM kernel choices of our runs (RTX PRO 6000 Blackwell). `setup/env.sh`
     copies it into `.cache/` on first use, so on that GPU the 512³ start and the camera fit match ours exactly; other GPUs
     autotune on their first run (slower first run).
3. **Our attention kernel**
   ```bash
   bash setup/build_kernels.sh
   ```
4. **Pixal3D** (512³ start; only needed for the default `pixal3d` start): clone [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) at commit `f7cf384`
   into `third_party/Pixal3D/repo`. Install its Python requirements in its own Python 3.10 virtual environment
   (`third_party/Pixal3D/venv`) following its README; its CUDA extensions come from `third_party/site-packages` (step 2).
   Put its weights in `third_party/Pixal3D/weights`.
5. **Detail estimator code and base weights**
   - Clone [hugoycj/StableNormal](https://github.com/hugoycj/StableNormal) at commit `1b7203c` into `third_party/StableNormal`
     (its YOSO pipeline differs from the Stable-X/StableNormal head):
     ```bash
     git clone https://github.com/hugoycj/StableNormal.git third_party/StableNormal
     git -C third_party/StableNormal checkout 1b7203c
     ```
   - Download `Stable-X/yoso-normal-v1-8-1` (NiRNE) and point `SF3D_NIRNE_WEIGHTS` to it.
6. **TRELLIS.2-4B weights**
   - Download `microsoft/TRELLIS.2-4B` and point `SF3D_TRELLIS2_WEIGHTS` to the snapshot.
   - The image condition model `facebook/dinov3-vitl16-pretrain-lvd1689m` must be in the Hugging Face cache
     (`HF_HOME`). It is a gated model, so accept its license on Hugging Face first.
7. **SuperFine3D weights**
   ```
   checkpoints/refiner/denoiser_step0002000.pt   (116 MB; trainable refiner parameters; train.json is in the repository)
   checkpoints/estimator/grey/ckpt_04000.pt      (2.3 GB; detail estimator, grey route)
   checkpoints/estimator/rgb/ckpt_last.pt        (2.3 GB; mesh-conditioned detail estimator, textured inputs)
   ```
   The download link will be added here.
8. **Locations:** edit the variables marked `set me` in `setup/env.sh`. Every script reads its locations from there.

## Inference

```bash
# Pixal3D 512^3 start (Table 1) -> outputs/pangolin/final_4096.ply
bash scripts/infer.sh assets/examples/pangolin.png outputs/pangolin
# 2048^3 output
bash scripts/infer.sh assets/examples/pangolin.png outputs/pangolin_2048 2048
# TRELLIS.2 512^3 start
bash scripts/infer.sh assets/examples/nautilus.png outputs/nautilus 4096 trellis2
```

`assets/examples/` holds the two 4096&times;4096 inputs of the project page's 3D viewer (AI-generated images).
An input must be an RGBA image whose alpha channel masks the object. Background removal is not run.

**512³ start.** Set the fourth argument, or `SF3D_START`:

| `START` | 512³ start | Notes |
|---|---|---|
| `pixal3d` (default) | Pixal3D native 512³ output | Placed in the TRELLIS.2 reference frame and re-fitted to the input image. Runs concurrently with the reference frame. This is the configuration of Table 1. |
| `trellis2` | the TRELLIS.2 512³ shape of the reference frame, with its fitted camera | Pixal3D is not run and does not need to be installed. |

Everything after the start is identical for both: detail injection, encodes, recursive refinement, decoding and export.
- **`report.json`** records the time and peak GPU memory of every stage.
- **Stages:**
  1. TRELLIS.2 reference frame and input camera.
  2. 512³ start: Pixal3D, run concurrently and placed in the input frame (`pixal3d`), or the reference frame itself
     (`trellis2`).
  3. Detail injection at 512³ and 1024³.
  4. Encode 2048³ → refine → decode.
  5. Inject → encode 4096³ → refine → decode.
  6. Final band-limited injection and hole capping.
- **Exactness:** all speed switches set by `scripts/infer.sh` give the same output as the reference path.
  `FUSED=0` falls back to the unfused DiT.

## Training

### Refiner

1. **Data.** `SF3D_DATA` holds the TRELLIS.2-encoded latents of the training assets:
   - `shape_latent_1024/` and `shape_latent_512/`, which are paired.
   - `data256/` (256³ latents).
   - `renders_cond/` (condition renders).
   - `metadata/` (instance lists).

   These are produced with TRELLIS.2's `data_toolkit`. Then prepare the fork and the starting weights:
   ```bash
   source setup/env.sh
   bash setup/prepare_trellis2_variants.sh        # TRELLIS.2 fork used by the trainers
   python setup/convert_trellis2_denoiser.py      # TRELLIS.2 1024 flow model -> checkpoints/trellis2/official_1024_denoiser.pt
   ```
2. **Stages.** Each stage starts from the previous one:

   | Stage | Trains | Steps |
   |---|---|---|
   | 1 | spatial-cell attention (HR LoRA, boundary-prioritized cross-cell routing), from the TRELLIS.2 1024 flow model | 5000 |
   | 2 | cross-scale guidance (LR adapters reading the 512³ parent latent) | 3000 |
   | 3 | joint HR/LR fine-tuning | 3000 |
   | 4 | refinement objective on decoded geometry (keep / detail) | 3000 |
   | 5 | geometry objective (edge / plane terms): the released refiner | 2000 |

   ```bash
   for s in 1 2 3 4 5; do bash training/refiner/run_stage.sh $s; done
   ```
   - Each stage writes to `runs/stageN_*` and links its checkpoints into `checkpoints/stages/`.
   - Stage 5 also links the final weights to `checkpoints/refiner/`.
   - Configuration overrides go after the stage number, for example `trainer.args.max_steps=10`.
   - One GPU per stage (about 75 GB).

### Detail estimator

The estimator is NiRNE, initialised from `yoso-normal-v1-8-1`. It is fine-tuned to predict normals of zoomed-in crops
at the model's native 768 px tile size. The loss combines cosine error with a masked high-pass term.

```bash
bash training/estimator/train.sh grey     # grey-shaded crops                    -> checkpoints/estimator/grey
bash training/estimator/train.sh rgb      # textured, lit, mesh-conditioned      -> checkpoints/estimator/rgb
```

The RGB route proceeds in steps:
1. Start from the grey model.
2. Train on 50% textured and lit views.
3. Switch to mesh conditioning: the UNet starts from the latent of the coarse mesh normals, and the image drives the
   ControlNet.
4. Add v2 relief and scan crops, with conditioning misalignment and dropout plus albedo, photometric and exposure
   augmentation.

**Data** goes under `$SF3D_DATA/estimator`:
- `training/estimator/build_v2.sh`: Objaverse download → Blender dump → crop records → grey views → GT normal targets →
  detail filter → textured renders.
- `make_targets.py` and `make_cond.py`: targets and coarse-normal conditions for the crop views.

## Evaluation

```bash
source setup/env.sh
python evaluation/score_geometry.py KEY MESH.ply OUT.json      # CD-L1, P95, F-scores, DQ (MeshyBench quality_4096)
bash setup/python.sh evaluation/fid_t4k_img.py KEY OUT.json LABEL=MESH@CAMDIR [...]   # Ang. and band correlations
bash evaluation/cam_t4k.sh LIST                                # baseline cameras (image-only fit) + fidelity
```

- **Geometry:** each mesh is normalised and rigidly aligned to the ground truth (100k samples). CD-L1 and F-scores are
  then computed point-to-triangle with 2M samples per direction.
- **DQ** uses [MeshyBench](https://github.com/meshy-dev/meshybench) at commit `5cce370`; point `SF3D_MESHYBENCH` to that
  checkout (default `third_party/meshybench`).
- **Image-space fidelity:** each method is rendered with its own image-only fitted camera on the pixel grid of the
  input crop and compared with ground-truth normals. The 4–16 px and 16–64 px band correlations are normal-map
  correlations after band-pass filtering.

## Acknowledgements and licenses

SuperFine3D builds on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2),
[Pixal3D](https://github.com/TencentARC/Pixal3D), [StableNormal / NiRNE](https://github.com/Stable-X/StableNormal),
[FlashAttention](https://github.com/Dao-AILab/flash-attention) and [MeshyBench](https://github.com/meshy-dev/meshybench).
See `THIRD_PARTY_NOTICES.md` for the files derived from them and their licenses.

## Citation

```bibtex
@misc{li2026superfine3d,
  title  = {SuperFine3D: Efficient Global-Local Recursive Refinement for Extremely Detailed 3D Geometry Generation},
  author = {Li, Lihan and Zhou, Junwei and Tai, Yu-Wing},
  year   = {2026}
}
```
