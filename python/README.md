# Trinity Python/PyTorch bindings

Build concrete tensor program plans, compile them into CUDA artifacts, and run
them with PyTorch Tensors.

## Getting started

This walkthrough runs a BF16 matrix multiplication on one Hopper GPU.

### Prerequisites

- Linux x86-64 with a Hopper GPU (`sm_90a`).
- CUDA Toolkit 13.0 and a compatible NVIDIA driver.
- uv, Python 3.12, Rust, and a C++17 compiler.

The installation below selects PyTorch 2.9.1+cu130 and initializes the pinned
CUTLASS dependency.

### Install

From the Trinity repository root:

```sh
cd crates/trinity-lowering

git submodule update --init --recursive third_party/cutlass
export CUDA_HOME=/usr/local/cuda-13.0
export CUTLASS_HOME="$PWD/third_party/cutlass"

uv sync --no-install-project
TRINITY_BUILD_CUDA=1 MAX_JOBS=1 uv pip install --no-build-isolation --no-deps -e .
```

Adjust `CUDA_HOME` to your CUDA Toolkit location. Run the following commands from
this same directory.

### Run your first program

```sh
uv run --no-sync python python/examples/streamed.py
```

The example compiles a `(128, 64) @ (64, 128)` GEMM, binds PyTorch input Tensors,
runs it twice on a CUDA stream, and checks a saved result against a PyTorch
reference. It then closes the execution and module. Successful completion means
the numerical comparison passed.

Read [plans.py](examples/plans.py) to see how the physical plan is built, then
[streamed.py](examples/streamed.py) for the
`compile → load → prepare → run → wait → close` flow.

`run()` submits work asynchronously and reuses the output Tensor. The example
clones the output before submitting again and calls `wait()` before checking
the result.
