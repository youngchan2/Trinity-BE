# trinity-lowering

Build concrete tensor program plans, compile them into CUDA artifacts, and run
them with PyTorch Tensors.

Rust handles implementation selection, physical plan validation, CUDA emission,
and compilation. The Python API binds Tensors to a C++ runtime that owns native
loading and execution.

The CUDA backend targets NVIDIA Hopper (`sm_90a`), with streamed execution for
single-GPU programs and persistent execution using NVSHMEM for multi-GPU programs.

Contiguous BF16/FP32 vectors and matrices support pointwise arithmetic, ReLU,
row sums and explicit row broadcasting. See the
[Python operator guide](python/README.md#pointwise-operations) for enumeration
and dtype contracts.

## Getting started

Follow the [Python/PyTorch guide](python/README.md) for prerequisites,
installation, and a complete single-GPU BF16 matrix multiplication example.

The uv project is rooted in this directory, alongside `Cargo.toml`. Python
sources, examples, and tests live under `python/`.

## Source layout

| Location | Purpose |
| --- | --- |
| [src/physical](src/physical/) | Physical plan construction and validation |
| [src/implementation](src/implementation/) | Concrete operation implementations |
| [src/emit/cuda](src/emit/cuda/) | CUDA source generation and host ABI |
| [src/compile/cuda](src/compile/cuda/) | NVCC compilation and artifact ownership |
| [src/python.rs](src/python.rs) | Python compiler bindings |
| [src/native](src/native/) | Tensor execution, NVSHMEM, and CUDA Graph runtime |
| [python](python/) | Python API, examples, and integration tests |

## Development

After setting up the Python environment, run these commands from this directory:

```sh
uv run --no-sync cargo fmt -p trinity-lowering -- --check
uv run --no-sync cargo clippy -p trinity-lowering --all-targets --locked -- -D warnings
uv run --no-sync cargo test -p trinity-lowering --locked
uv run --no-sync pytest
```

The default tests do not require a GPU. Generated-kernel execution tests require
Hopper hardware; multi-GPU integration tests additionally require NVLink and
NVSHMEM.
