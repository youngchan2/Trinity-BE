# Trinity Lowering

Trinity Lowering provides physical planning, CUDA code generation, and compilation for explicit tensor programs, with a runtime for PyTorch Tensor execution.

The compiler selects kernel implementations for a physical plan and produces
CUDA artifacts. The runtime loads these artifacts and executes them with
PyTorch Tensors bound through the Python API.

## Getting started

### Prerequisites

- Linux x86-64
- NVIDIA GPU listed in the [supported targets](docs/architecture/emission.md#대상-하드웨어).
- CUDA Toolkit 13.0 and a compatible NVIDIA driver.
- uv, Python 3.12, Rust, and a C++17 compiler.

The examples below use the default Hopper target.

### Installation

Run the following commands from the `trinity-lowering` source directory
containing `pyproject.toml` and `Cargo.toml`:

```sh
git submodule update --init --recursive third_party/cutlass
export CUDA_HOME=/usr/local/cuda-13.0
export CUTLASS_HOME="$PWD/third_party/cutlass"

uv sync --no-install-project
TRINITY_BUILD_CUDA=1 MAX_JOBS=1 uv pip install --no-build-isolation --no-deps -e .
```

Adjust `CUDA_HOME` to the local CUDA Toolkit path. The uv project in this
directory pins PyTorch 2.9.1+cu130. Run subsequent commands from this directory.

### Examples

Run BF16 matrix multiplication with bias addition:

```sh
uv run --no-sync python python/examples/streamed.py
```

The example compiles the program, binds PyTorch Tensors, executes it on a CUDA
stream, and checks the result against a PyTorch reference.

For CUDA Graph capture and replay with an additional ReLU operation:

```sh
uv run --no-sync python python/examples/graph.py
```

## How it works

![Trinity Lowering architecture](docs/images/trinity-lowering-architecture.svg)

- **Planning** establishes a validated physical plan for the input tensor program.
- **Emission** translates the plan into CUDA code for the target hardware.
- **Compilation** makes the generated code executable by the runtime.

See the [architecture guide](docs/architecture/README.md) for the public API and details of each stage.

## Source layout

| Location | Purpose |
| --- | --- |
| [src/plan](src/plan/) | Physical plan construction and validation |
| [src/implementation/definitions](src/implementation/definitions/) | Implementation definitions and applicability enumeration |
| [src/emit](src/emit/) | Kernel selection, execution placement, and CUDA source generation |
| [src/compile/cuda](src/compile/cuda/) | NVCC compilation and artifact ownership |
| [src/python.rs](src/python.rs) | Python compiler bindings |
| [src/native](src/native/) | Native loading, Tensor execution, and CUDA Graph runtime |
| [python](python/) | Python API, examples, and integration tests |

## Development

After setting up the Python environment, run these commands from this directory:

```sh
uv run --no-sync cargo fmt -p trinity-lowering -- --check
uv run --no-sync cargo clippy -p trinity-lowering --all-targets --locked -- -D warnings
uv run --no-sync cargo test -p trinity-lowering --locked
uv run --no-sync pytest
```
