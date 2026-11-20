# SuperFine3D: Efficient Global-Local Recursive Refinement for Extremely Detailed 3D Geometry Generation

**Lihan Li, Junwei Zhou, Yu-Wing Tai** · Dartmouth College

SuperFine3D turns a single image into **4096³ geometry** (a sparse 256³ latent lattice) in about three minutes on one GPU.
One refinement operator, trained only on paired 512³/1024³ TRELLIS.2 latents, is applied recursively
(1024³ → 2048³ → 4096³). Spatial-cell attention replaces full self-attention over the hundreds of thousands of
fine-grid tokens. Image-faithful relief comes from a fine-tuned one-step diffusion normal estimator.

- **Recursive refinement.** Each output latent supplies the guidance and the sparse support of the next stage.
- **Spatial-cell attention.** Tokens attend within local cells, with bounded, boundary-prioritized cross-cell
  communication. Cells are processed by index in one launch, and a FlashAttention-2 varlen kernel reads K/V by row index.
- **Cross-scale guidance.** Spatially anchored coarse (parent) features and geometry-aware objectives keep the shape
  while detail is added.
- **Image-faithful detail.** A fine-tuned NiRNE estimator runs on tiles of the upsampled input and drives a band-limited
  height field. A mesh-conditioned variant keeps albedo out of the geometry on textured inputs.

## Results

Geometric quality on Toys4K, from the paper's Table 1. CD-L1 is ×10³ and F-scores are percentages. Corr is the
image-space normal correlation in the 4–16 px and 16–64 px bands, with each method's own fitted camera.

| Method | Res. | CD-L1 ↓ | Ang. (°) ↓ | F@0.002 ↑ | DQ ↑ | Corr 4–16 ↑ | Corr 16–64 ↑ |
|---|---|---|---|---|---|---|---|
| TRELLIS.2 | 1536 | 15.711 | 35.40 | 17.075 | 0.5444 | 0.002 | 0.105 |
| Hunyuan3D-2.1 | 1024 | 16.115 | 38.42 | 14.204 | 0.3680 | 0.006 | 0.110 |
| Pixal3D | 1536 | 17.761 | 33.70 | 14.197 | 0.5319 | 0.038 | 0.268 |
| UltraShape | 4096 | 17.381 | 35.76 | 12.083 | 0.1708 | 0.014 | 0.166 |
| **SuperFine3D** | 2048 | **15.639** | **27.44** | 17.107 | 0.5957 | 0.364 | 0.501 |
| **SuperFine3D** | 4096 | 15.660 | 27.46 | **17.123** | **0.6474** | **0.368** | **0.518** |

Efficiency was measured on one NVIDIA RTX PRO 6000 Blackwell:
- **End to end:** an image becomes a 4096³ mesh in 176.2 s, with 48.5 GiB peak GPU memory. Interpreter start-up and
  imports are excluded.
- **Against a full-attention recursive baseline:** 9.4× faster, with 23.1% less memory.
- **Against UltraShape:** 12.6× faster.
- **Per layer, at 708k high-resolution tokens:** one attention layer is 525× faster and one DiT block 28× faster than
  full attention (FlashAttention-2).

## Repository layout

| Path | Contents |
|---|---|
| `scripts/infer.sh` | image → mesh, Table-1 configuration |
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
| `setup/` | environment (`env.sh`), kernel build, TRELLIS.2 fork preparation, config rendering |

## Installation

Tested with Python 3.10, CUDA 12.8 and PyTorch 2.9.0 on RTX PRO 6000 Blackwell GPUs (sm_120, 96 GB).

1. **Python environment**
   ```bash
   pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
2. **TRELLIS.2** at commit `75fbf01`, with its CUDA extensions (`o-voxel`, `FlexGEMM`, `CuMesh`, `nvdiffrast`,
   `flash-attn==2.7.3`), built with TRELLIS.2's `setup.sh`.
   ```bash
   git clone https://github.com/microsoft/TRELLIS.2.git third_party/TRELLIS.2
   git -C third_party/TRELLIS.2 checkout 75fbf01
   ```
   Wheels built for a GPU architecture that the stock wheels lack can be installed with
   `pip install --target third_party/site-packages`; that directory comes first on the path.
3. **Our attention kernel**
   ```bash
   bash setup/build_kernels.sh
   ```
4. **Pixal3D** (512³ start; only needed for the default `pixal3d` start): clone [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) at commit `f7cf384`
   into `third_party/Pixal3D/repo`. Install it in its own virtual environment (`third_party/Pixal3D/venv`) following
   its README, and put its weights in `third_party/Pixal3D/weights`.
5. **Detail estimator code and base weights**
   - Clone [Stable-X/StableNormal](https://github.com/Stable-X/StableNormal) into `third_party/StableNormal`.
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
bash scripts/infer.sh input.png outputs/example                  # -> outputs/example/final_4096.ply (Pixal3D start)
bash scripts/infer.sh input.png outputs/example 2048             # 2048^3 output
bash scripts/infer.sh input.png outputs/example 4096 trellis2    # TRELLIS.2 512^3 start
```

The input must be an RGBA image whose alpha channel masks the object. Background removal is not run.

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
- **DQ** uses [MeshyBench](https://github.com/meshy-dev/meshybench); point `SF3D_MESHYBENCH` to a checkout.
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
