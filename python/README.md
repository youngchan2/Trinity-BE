# Trinity Python/PyTorch bindings

Build concrete tensor program plans, compile them into CUDA artifacts, and run
them with PyTorch Tensors.

## Getting started

This walkthrough runs a BF16 matrix multiplication with bias addition on one Hopper GPU.

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

The example compiles a `(128, 64) @ (64, 128)` GEMM followed by a `(128, 128)` bias
addition, binds the `X`, `W` and `bias` Tensors, runs it twice on a CUDA stream,
and checks a saved result against a PyTorch reference. It then closes the
execution and module. Successful completion means the numerical comparison passed.

Read [plans.py](examples/plans.py) to see how the physical plan is built, then
[streamed.py](examples/streamed.py) for the
`compile → load → prepare → run → wait → close` flow.

`run()` submits work asynchronously and reuses the output Tensor. The example
clones the output before submitting again and calls `wait()` before checking
the result.

### CUDA Graph with bias addition and ReLU

```sh
uv run --no-sync python python/examples/graph.py
```

The Graph example uses `gemm_relu()` to compile GEMM, a full `[M, N]` bias addition
and ReLU into one physical plan, preserving BF16 intermediates between operations.
The bias is a named Tensor input retained by the prepared execution. CUDA Graph
captures all three generated kernels through `execution.run`; no PyTorch
arithmetic fallback is needed. The streamed example continues to use `gemm()`
for GEMM followed by bias addition.

### Pointwise operations

`tl.pointwise_implementations()` enumerates Add, Mul, Div, ScalarDiv, Square, Sqrt,
Sigmoid and ReLU definitions for contiguous 1D/2D BF16 and FP32 tensors.
Each definition's `enumerate(dtypes, shapes)` takes inputs followed by one output;
all operand shapes must match. ScalarDiv additionally requires `scalar=...`.
Arithmetic uses FP32 and converts to the declared output dtype at the store.

For example, obtain a BF16-to-FP32 ReLU implementation with:

```python
definition = next(d for d in tl.pointwise_implementations() if d.id == "cuda.relu")
implementation = definition.enumerate(["bf16", "fp32"], [[129], [129]])[0]
```

ReLU maps negative values, including negative infinity, to positive zero.
NaN, positive infinity and signed zero are preserved. It uses the same 128-thread
CUDA body for streamed and persistent execution, with bounds checks on tail chunks.
