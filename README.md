# trinity-lowering

> Native Streamed emission is connected for Hopper, sm_89 and sm_120. Explicit
> IR/Builder plans can generate CUDA artifacts using pointwise, reduction, GEMM
> and Register continuations. Persistent, communication, Shared tensor transport
> and Opaque emission remain unsupported. The previous emitter is reference-only
> under [old/emit-rewrite](old/emit-rewrite/README.md).

Generate Triton kernels from extracted, scheduled Trinity IR, or construct explicit
tensor program plans for the CUDA provider pipeline.

| Input | Entry point | Current result |
| --- | --- | --- |
| Scheduled Trinity IR with named views, keyed indices and split loops | `triton::compile(text, options)` | Python source containing Triton kernels and `forward(...)` |
| Explicit IR with symbol and dtype bindings | `lower_ir(text, config)` | `PhysicalPlan` values for the CUDA provider pipeline |
| Explicit values, operations and loops | `PhysicalPlanBuilder::build(...)` | A validated `PhysicalPlan` |

The Triton fallback remains available independently of the native CUDA emitter.
It uses `analysis::ProgramAnalysis` and `triton::ProgramPlan`; the updated CUDA path
uses `plan::PhysicalPlan`. Automatic conversion and cross-backend selection are not
connected yet. Triton currently stores tensors as FP16; CUDA plans carry explicit
dtypes. The former CUDA `lower_loop_ir` API is now named `lower_ir`.

Rust handles implementation selection, physical plan validation, CUDA emission,
and compilation. The Python API binds Tensors to a C++ runtime that owns native
loading and execution.

The CUDA backend emits single-GPU Streamed programs for `sm_90a`, `sm_89` and
`sm_120`. The retained NVSHMEM runtime is not yet connected to the new emitter.

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
| [src/analysis](src/analysis/) | Trinity IR parsing, ordered accesses, lexical scopes and dependency queries for Triton |
| [src/triton/plan.rs](src/triton/plan.rs) | Triton program and per-kernel plans |
| [src/triton/lowering](src/triton/lowering/) | Triton storage, initialization, indexing and launch planning |
| [src/triton/codegen](src/triton/codegen/) | Triton kernel bodies and Python launch wrappers |
| [src/plan](src/plan/) | Physical plan construction and validation |
| [src/implementation/definitions](src/implementation/definitions/) | Implementation identities and candidate enumeration; no program generation |
| [src/emit](src/emit/) | CUDA scope collection, provider selection, kernel composition, execution placement and rendering |
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
`triton::lower(analysis, options)` and `ProgramPlan::emit()`. The IR's computation
graph and loop schedule are preserved. Managed mode allocates intermediate
tensors and returns outputs; it is enabled automatically for programs with `mloop`.
See [the Triton emitter guide](docs/TRITON_EMITTER.md) for the analysis and plan contract.

```sh
cargo test --locked --test batched_mla_emit -- --nocapture
```

These tests read the checked-in stage 14/16/20 fixtures and write `stage14.py`,
`stage16.py` and `stage20.py` under `target/tests/batched_mla/`. They check Python
syntax, kernel counts and the launch wrapper without importing PyTorch or Triton
or executing GPU work. Python 3 is required. `generated_kernels/` and
`reference_kernels/` remain ignored artifacts.

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
