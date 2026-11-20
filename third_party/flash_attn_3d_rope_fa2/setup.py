"""Build script for flash_attn_3d_rope — vendored fork of flash_attn 2.8.3 with
3D-RoPE-fused FlashAttention varlen kernel (Lever B).

Phase 0 status: namespace-renamed verbatim port. RoPE fusion lands in Phase 1+.

Compilation scope:
  - sm_89 (5880 Ada) + sm_90 (H200) — matches deployed targets.
  - fp16 + bf16, headdim ∈ {32, 64, 96, 128, 192, 256}, both causal and non-causal
    instantiations included for binary completeness even though grid_sdpa only
    uses non-causal hdim=32.
  - Drops: ROCm/HIP path, prebuilt-wheel download, multi-arch sm_75 / sm_100 / sm_120.

Namespace isolation: -DFLASH_NAMESPACE=flash_3d_rope so symbols don't collide with
the installed flash_attn==2.8.3 wheel that ships in the cluster image.

Local install (dev):
    pip install -e .

Build a wheel:
    python setup.py bdist_wheel
    # produces dist/*.whl
"""
import os
import sys
import warnings
from pathlib import Path

# Auto-default CUDA_HOME to the ccache-shim home if it exists, so devs and
# CI get warm rebuilds (~90s vs ~30min) without having to remember the env
# var. Gated on shim existence so envs without ccache wiring fall through
# to the real /usr/local/cuda.
_default_cuda_ccache = Path.home() / ".local/cuda-ccache-home/bin/nvcc"
if "CUDA_HOME" not in os.environ and _default_cuda_ccache.exists():
    os.environ["CUDA_HOME"] = str(_default_cuda_ccache.parent.parent)

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


PACKAGE_NAME = "flash_attn_3d_rope_fa2"
NAMESPACE = "flash_3d_rope_fa2"  # avoid symbol collision with installed flash_attn AND fa3 sibling

THIS_DIR = Path(__file__).parent.resolve()
SRC_DIR = THIS_DIR / "src"
CUTLASS_INC = THIS_DIR / "cutlass" / "include"


# ─── Architecture flags ───────────────────────────────────────────────────────
CUDA_ARCHS = os.environ.get("FA3D_CUDA_ARCHS", "89;90").split(";")


def get_cuda_arch_flags():
    flags = []
    for arch in CUDA_ARCHS:
        flags += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    return flags


# ─── Source list ──────────────────────────────────────────────────────────────
def collect_sources():
    """All .cu kernel instantiations + flash_api.cpp."""
    cpp_sources = [str(SRC_DIR / "flash_api.cpp")]
    cu_sources = sorted(str(p) for p in SRC_DIR.glob("flash_*_hdim*.cu"))
    return cpp_sources + cu_sources


# ─── nvcc / cxx flags ─────────────────────────────────────────────────────────
def torch_cxx_abi_flag():
    """Match torch's _GLIBCXX_USE_CXX11_ABI to avoid link errors."""
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    return [f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]


CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    f"-DFLASH_NAMESPACE={NAMESPACE}",
    "-DFLASHATTENTION_DISABLE_DROPOUT",       # we don't use dropout
    "-DFLASHATTENTION_DISABLE_ALIBI",         # we don't use alibi
    "-DFLASHATTENTION_DISABLE_SOFTCAP",       # we don't use softcap
    "-DFLASHATTENTION_DISABLE_LOCAL",         # we don't use local/window attention
] + torch_cxx_abi_flag()

NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    f"-DFLASH_NAMESPACE={NAMESPACE}",
    "-DFLASHATTENTION_DISABLE_DROPOUT",
    "-DFLASHATTENTION_DISABLE_ALIBI",
    "-DFLASHATTENTION_DISABLE_SOFTCAP",
    "-DFLASHATTENTION_DISABLE_LOCAL",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-Xcompiler=-O3",
    # fp16/bf16 support required for newer CUTLASS
    "--threads", "4",
] + torch_cxx_abi_flag() + get_cuda_arch_flags()


# ─── Extension ────────────────────────────────────────────────────────────────
ext_modules = [
    CUDAExtension(
        name=f"{PACKAGE_NAME}_C",
        sources=collect_sources(),
        include_dirs=[
            str(SRC_DIR),
            str(CUTLASS_INC),
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
]


setup(
    name=PACKAGE_NAME,
    version="0.1.0.dev0",
    description="3D-RoPE-fused FlashAttention varlen (Lever B). "
                "Vendored fork of flash_attn 2.8.3.",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.6"],
    python_requires=">=3.10",
)
