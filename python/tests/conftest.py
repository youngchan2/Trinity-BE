import os
from pathlib import Path
import pytest


@pytest.fixture
def sdk(tmp_path):
    root = tmp_path / "sdk with spaces"

    def write(path, text=""):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    fixture = Path(__file__).resolve().parents[2] / "src/compile/cuda/tests/fake_nvcc.py"
    script = write("cuda/bin/nvcc", fixture.read_text())
    script.chmod(0o700)
    write("cuda/lib64/libcudart.so")
    write("cuda/lib64/stubs/libcuda.so")

    write("cutlass/include/cute/tensor.hpp")
    write(
        "cutlass/include/cutlass/version.h",
        "#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 5\n#define CUTLASS_PATCH 1\n",
    )

    write("mode", "success")
    write("fixture.cpp", 'extern "C" int trinity_abi() {return 0;}')

    import trinity_lowering as tl

    return root, tl.CompileConfig(cuda_root=root / "cuda", cutlass_root=root / "cutlass")


def pytest_collection_modifyitems(items):
    import torch

    for item in items:
        if "gpu" in item.keywords and (
            not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0)
        ):
            item.add_marker(pytest.mark.skip(reason="requires an accessible Hopper GPU"))
        if "cuda_runtime" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="requires an accessible CUDA GPU"))
        if "distributed" in item.keywords and os.environ.get("TRINITY_DISTRIBUTED_TESTS") != "1":
            item.add_marker(
                pytest.mark.skip(reason="set TRINITY_DISTRIBUTED_TESTS=1 on a Hopper/NVLink node")
            )
