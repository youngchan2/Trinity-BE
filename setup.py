"""Dynamic PyTorch extension configuration for the PEP 517 build backend.

Package metadata, discovery and the Rust extension live in pyproject.toml.
Build through uv; this file is not a command-line entry point.
"""

import os
from pathlib import Path

from setuptools import setup

root = Path(__file__).parent.resolve()
extensions = []
commands = {}

if os.environ.get("TRINITY_BUILD_CUDA", "0") == "1":
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    if torch.__version__.split("+")[0] != "2.9.1" or torch.version.cuda != "13.0":
        raise RuntimeError("native build requires torch 2.9.1+cu130")

    if CUDA_HOME is None:
        raise RuntimeError("native build requires a CUDA Toolkit; set CUDA_HOME")

    include_dirs = [str(root / "src/native"), str(root / "src/emit/cuda")]
    defines = []

    if nvshmem := os.environ.get("NVSHMEM_HOME"):
        include_dirs.append(str(Path(nvshmem) / "include"))
        defines.append(("TRINITY_NVSHMEM", "1"))

    extensions.append(
        CUDAExtension(
            "trinity_lowering._native",
            [
                str(Path("src/native") / name)
                for name in ("bindings.cpp", "runtime.cpp", "world.cpp")
            ],
            include_dirs=include_dirs,
            define_macros=defines,
            libraries=["cuda", "dl"],
            library_dirs=[str(Path(CUDA_HOME) / "lib64/stubs")],
            extra_compile_args={"cxx": ["-std=c++17", "-O2", "-g0", "-fvisibility=hidden"]},
        )
    )
    commands["build_ext"] = BuildExtension

setup(ext_modules=extensions, cmdclass=commands)
