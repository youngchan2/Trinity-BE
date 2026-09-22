# Trinity Lowering

Trinity Lowering provides physical planning, CUDA code generation, and compilation for explicit tensor programs, with a runtime for PyTorch Tensor execution.

The compiler selects kernel implementations for a physical plan and produces
CUDA artifacts. The runtime loads these artifacts and executes them with
PyTorch Tensors bound through the Python API.

## Compiler entry points

Generate Triton kernels from extracted, scheduled Trinity IR, or construct explicit
tensor program plans for the CUDA provider pipeline.

| Input | Entry point | Current result |
| --- | --- | --- |
| Scheduled Trinity IR with named views, keyed indices and split loops | `triton::compile(text, options)` | Python source containing Triton kernels and `forward(...)` |
| Explicit IR with symbol and dtype bindings | `lower_ir(text, config)` | `PhysicalPlan` values for the CUDA provider pipeline |
| Explicit values, operations and loops | `PhysicalPlanBuilder::build(...)` | A validated `PhysicalPlan` |
| A `PhysicalPlan` | `emit::kernel_candidates(&plan)` | CuTe/Triton/Quack candidates and unsupported reasons per operation |
| Single-GPU computation `PhysicalPlan`, including scheduled loops/regions | `emit::emit_triton(&plan, options)` | Triton kernels and ordered `forward(...)` launches |
| Single-GPU `PhysicalPlan` | `emit::emit_python(&plan)` | Python `prepare(inputs)` and executable; compares independent operations or runs scheduled regions through Triton |

The fallback now follows `scheduled IR → PhysicalPlan → TritonKernelProvider →
TritonPlan → source + launches`. Views, symbolic dimensions, ordered regions,
loops, outputs and input mutations belong to the common plan. Triton chooses
padding, local representation, numerical precision and launch configurations.
It consumes typed plan expressions directly, without reparsing a saved source AST.
Independent full-tensor operations can still compare Triton with Quack. Whole
scheduled regions currently execute Triton directly; cross-provider benchmarking
of those regions and native CUDA implementations remains future work.
Unannotated source IR defaults to FP16; typed Triton candidates preserve explicit
FP16/BF16/FP32 storage. Native CUDA continues to require BF16/FP32.
The former CUDA `lower_loop_ir` API is now named `lower_ir`.

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
| [src/analysis](src/analysis/) | Shared shape, access, scope and dataflow facts; storage inference; source collection and PhysicalPlan projection |
| [src/emit/provider/triton/plan.rs](src/emit/provider/triton/plan.rs) | Triton program and per-kernel plans |
| [src/emit/provider/triton/lowering](src/emit/provider/triton/lowering/) | Triton padding, SSA initialization, numerical precision, indexing and launch planning |
| [src/emit/provider/triton/codegen](src/emit/provider/triton/codegen/) | Triton kernel bodies and Python launch wrappers |
| [src/analysis/plan](src/analysis/plan/) | Physical plan construction, scheduled-IR import, symbolic binding and validation |
| [src/emit/implementation/definitions](src/emit/implementation/definitions/) | Implementation identities and candidate enumeration; no program generation |
| [src/emit](src/emit/) | Provider candidates, backend lowering/codegen, native composition and Python program assembly |
| [src/emit/candidate.rs](src/emit/candidate.rs) | Candidate discovery without choosing the first supported provider |
| [src/emit/provider/quack](src/emit/provider/quack/) | Optional opaque GEMM/epilogue specifications and Python call wrappers |
| [src/emit/provider/triton](src/emit/provider/triton/) | Whole-program fallback provider and operation adapter using the same path |
| [src/emit/wrapper](src/emit/wrapper/) | Python program assembly, correctness checks, benchmarking, selection and execution |
| [src/compile/cuda](src/compile/cuda/) | NVCC compilation and artifact ownership |
| [src/python.rs](src/python.rs) | Python compiler bindings |
| [src/native](src/native/) | Tensor execution, NVSHMEM, and CUDA Graph runtime |
| [python](python/) | Python API, examples, and integration tests |

`PhysicalPlanBuilder` accepts an explicit ordered Loop/Operation program. Compute
operations require a supplied store expression; `build()` never infers loops,
tiles, or computation bodies. It normalizes names/operands, checks structural
invariants, and canonicalizes IDs while preserving execution order.

The removed automatic expansion and `ImplementationDefinition::schedule()` contract
are preserved under [old/plan-rewrite](old/plan-rewrite/README.md). Implementation
identities and applicability enumeration remain under `definitions`; these do not
construct programs. Python stays in place and supports explicit body expressions
and `add_loop()` nodes. See [plans.py](python/examples/plans.py).

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

Python IR reader tests use the checked-in `tests/fixtures/ir` fixtures. The optional
FFN GPU benchmark requires an enclosing workspace's `examples/ffn_v3/run.py` and
skips when that harness is unavailable.

## Triton source generation

Use `triton::compile(text, options)` or `analysis::analyze_text(text)` followed by
`triton::lower(analysis, options)` and `TritonPlan::emit()`. Both enter the common
PhysicalPlan and Triton provider. The IR's computation
graph and loop schedule are preserved. The wrapper allocates intermediate
tensors and returns outputs through one emission path, including `mloop` programs.

For an inspectable common plan, use `TritonKernelProvider.lower_source(...)` in
Rust or `lower_triton(text, **options)` in Python. The result exposes the common
plan and generated source. See [the provider contract and usage](docs/architecture/triton-provider.md).

Both `lower_ir` and `PhysicalPlanBuilder::from_scheduled` use `analysis::storage`
to assign `ValueInstance.storage`:
ABI inputs/outputs are `External`, intermediates that need materialization or
cross a kernel boundary are `Global`, and compatible local intermediates are
`Register`. Triton honors these backing-storage decisions; a global output can
still be accumulated locally before its final store.
`lower_ir` retains unbound tile symbols when their access relationships suffice
to prove the storage classification; it never supplies a hidden sample tile.

Emission uses FP32 register computation and accumulation while retaining
logical storage and GEMM operand dtypes. Raw-exp overflow is an IR numerical
limitation; emission does not silently insert stabilization. It validates multiple symbolic tile assignments and
benchmarks up to 64 configurations per kernel by default. See
[precision policy](docs/architecture/triton-precision.md) and
[autotuning policy](docs/architecture/triton-autotuning.md) for controls and limits.

```sh
cargo test --locked --test batched_mla_emit -- --nocapture
```

These tests read the checked-in stage 14/16/20 fixtures and write `stage14.py`,
`stage16.py` and `stage20.py` under `target/tests/batched_mla/`. They check Python
syntax, kernel counts and the launch wrapper without importing PyTorch or Triton
or executing GPU work. Python 3 is required. `generated_kernels/` and
`reference_kernels/` remain ignored artifacts.

## Optional Quack candidates

`emit::kernel_candidates` enumerates CuTe, Triton and applicable Quack implementations.
`emit::emit_python` composes independent candidates into a standalone module. Its
`prepare(inputs)` checks outputs against a PyTorch reference, benchmarks passing
candidates, and returns an executable with per-operation selections and reports.
Quack is imported lazily; a missing optional package leaves Triton available.
This selection path currently supports single-GPU, loop-free full-tensor operations.
Native CUDA emission is available separately through `emit()`; its implementations
are not yet included in the Python candidate comparison.

## Explicit plan inputs

`lower_ir` reads the loops, accesses and expressions already present in the input.
Its `mloop` normalization translates the specified split into parallel/sequential
loops; it does not select a new tile or schedule. Direct Rust Builder calls register
operations with `add_operation(inflows, outflows, expression)`, then pass the top-level
`Vec<Statement>` to `build(statements, output_name, output)`.

Operations own their expressions directly. Neither the Reader nor the Builder selects
an implementation; selection belongs to Emit. Recognized reductions implicitly start
at zero, with initialization generated by Emit. Builder communication uses an explicit
`all_gather` expression; textual communication IR remains deferred. See
[Planning](docs/architecture/planning.md) and [Emit design](docs/architecture/emission.md).

The producer of the IR or direct Builder input is responsible for validating loop
ranges and memory accesses. Plan construction validates structural consistency.

`TensorAccess` holds a value ID, an optional contiguous `view_shape`, and indices
in view-axis order. `ValueInstance::shape()` remains the allocation/ABI shape;
different views of one value share storage and must preserve the element count.
The text reader uses the first view as the base shape, and retains later views
on their accesses. The Builder can declare the base shape separately. The explicit
`lower_ir` reader requires concrete view extents. The scheduled frontend also
retains `view_dimensions` expressions and symbolic base dimensions alongside
sample shapes, so split scratch allocations and accesses specialize together.

`AccessIndex::Tile` and `ClippedTile` use `TileWidth::Constant(32)` or
`TileWidth::Symbol("BLOCK".into())`. Unbound tile/loop parameters remain in the plan;
symbols supplied to `lower_ir` are resolved immediately. Before provider selection
or emission, call `plan.bind_symbols(&bindings)` in Rust, or
`plan.bind_symbols({"BLOCK": 32})` in Python. This returns a new plan with both
tile widths and loop ranges bound consistently, leaving literal widths unchanged.
`plan.symbols()` (Python: `plan.symbols`) lists symbolic configuration names,
including those with sample defaults in `plan.bindings()`.
Binding is not autotuning: candidate enumeration and timing remain separate.

CuTe uses access views for tile shapes and addressing. Native coverage checks
compare physical storage intervals across views. The loop-free Triton candidate
adapter also preserves views; the Quack host adapter currently rejects changed
views until its argument binding supports them. The scheduled-IR Triton provider
now consumes the same common plan, including multiple outputs and input cache
updates. This does not expand native/Quack coverage automatically.

The current `trinity::lower(candidate, ...)` input contains a whole-tensor DAG, not
an explicit loop program. Its automatic expansion path has been retired and now
returns `LoweringError::ExplicitProgramRequired`. Candidate extraction itself is
unchanged; reconnecting it requires a separate explicit-program contract.

Emission supports independent constant parallel domains and constant sequential
loops within a CTA. Dependent bounds and parallel work inside a sequential loop
are rejected. Execution placement checks region coverage and inter-CTA hazards,
then renders a shared device body, Streamed wrappers and the existing host ABI.
Kernel launch thread counts and scratch sizes are resolved individually.

Compiler/runtime work follows [PLAN.md](../../docs/draft/PLAN.md). Public fusion
candidate generation remains unavailable. Structural plan validation is not a
proof of device execution legality; CUDA compilation and GPU accuracy require
separate verification.
