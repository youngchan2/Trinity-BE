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

CUDA implementations are grouped by logical operation under `src/implementation/cuda/`:
`gemm`, `all_gather`, `pointwise`, `reduce_sum`, and `broadcast`. Each concrete
implementation owns its candidate enumeration, validation, schedule, Body creation,
and CUDA templates. Cross-operation rules live in `fusion`; shared scalar support
lives in `expression.rs`.

The common emitter resolves loop scopes and tensor accesses, then calls the selected
implementation's `accumulation()` hook for fragment traversal and pipeline binding.
It connects output bindings and pointwise consumers, reserves implementation scratch
before shared intermediate tiles, and supplies the same Body to both runtimes.

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

## Unified lowering

Candidate and Builder operations are normalized into the same ordered Loop/Statement
program as `lower_loop_ir`. Compute operations have tile expressions; communication
operations use the selected implementation and explicit loop coordinates. Both can
appear in one plan. Invalid builder shapes or attributes may now fail in `finalize`.

CUDA backends implement `schedule`, symbolic `phases`, and optional communication
`work`/scalar hooks. Each task invokes one `Body { prologue, mainloop: Option<Phase>,
epilogue }`. Its mainloop contains the IR's existing sequential loops; pointwise
bodies can omit it. Bindings and internal symbols are renamed across the entire
body and connected by symbol identity before CUDA rendering.

A `ploop` describes a collection of tasks. `Execution.tasks` distinguishes Statement
provenance, task-set ID, body ID, argument index and task slot. Streamed execution
launches each task set as a grid, including specialized tail bodies; Persistent
Workers dispatch the same bodies. Nested Split-K partial and reduction sets retain
their dependency boundary. `Execution.work` exposes entry/stage reads and completion
writes, and `CudaSource::bodies()` exposes the composed phases.

`fuse(plan, fusion_rules(plan.target()))` returns the original and compatible adjacent
fusion candidates. GEMM→pointwise and pointwise chains forward registers after the
original dtype conversion. Compatible GEMM→GEMM uses a shared A tile adapter. Local
values need no launch-buffer allocation. Outside consumers prevent promotion.
Dependencies, stages, output tokens and resources are rebuilt for each fused plan.
The current Hopper rules require M/N multiples of 128 and K multiples of 64;
GEMM→GEMM additionally requires producer N=128 and consumer K=N=128.
Communication promotion, fusion of two loop-free root stores, and additional
fragment adapters remain unsupported.

Stage readiness is separate from Phase and pipeline buffer slots. Streamed bodies
rely on stream order for external readiness; internal CUDA synchronization remains.
Persistent Workers publish one completion token after the entire body succeeds.

Regression coverage is in `tests/fusion.rs`, `tests/loop_ir.rs`, and
`tests/unified_lowering.rs`. Their opt-in CUDA tests compile and link fused bodies,
FFN/Split-K, and mixed compute/communication programs. They do not execute GPU work.
