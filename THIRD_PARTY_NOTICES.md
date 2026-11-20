# Third-party notices

| Location | Origin | License |
|---|---|---|
| `training/trellis2_variants/` | modified and added files of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (commit 75fbf01); `setup/prepare_trellis2_variants.sh` adds the unmodified upstream files | MIT (`training/trellis2_variants/LICENSE.TRELLIS.2`) |
| `superfine3d/refiner/code/custom_models/` | derived from TRELLIS.2 model files | MIT |
| `third_party/flash_attn_3d_rope_fa2/` | fork of [FlashAttention](https://github.com/Dao-AILab/flash-attention) v2.8.3 (varlen forward/backward) with fused 3D RoPE and row-indexed K/V reads, from the Pointelligence runtime | BSD-3-Clause (`LICENSE.flash-attention`, `AUTHORS.flash-attention`) for the FlashAttention parts; Apache-2.0 (`LICENSE`) for the additions |
| `superfine3d/refiner/code/pointelligence_grid/`, `training/trellis2_variants/trellis2/third_party/pointelligence/` | Pointelligence grid attention and RoPE kernels | Apache-2.0 |
| `superfine3d/detail/hfdetail/estimator.py`, `training/estimator/ft_train.py` | use the NiRNE pipeline of [StableNormal](https://github.com/Stable-X/StableNormal) and the `Stable-X/yoso-normal-v1-8-1` weights (not included) | see upstream |
| `superfine3d/detail/fig5/pixal512_worker.py` | drives [Pixal3D](https://github.com/TencentARC/Pixal3D) inference (not included) | see upstream |
| `evaluation/score_geometry.py` | calls the `mesh_details` metric of [MeshyBench](https://github.com/meshy-dev/meshybench) (not included) | see upstream |
