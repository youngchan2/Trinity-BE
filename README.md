# trinity-lowering

`trinity-lowering` provides concrete implementation candidates and validated
physical tensor program graphs.

## Target selection

`TargetCapability` selects a platform and contains its concrete capability:

```rust
use trinity_lowering::{CudaTargetCapability, LoweringConfig, TargetCapability};

let target = TargetCapability::Cuda(CudaTargetCapability::Hopper);
let config = LoweringConfig::new(target);
```

The platform enum lives in `src/platform/mod.rs`; CUDA capabilities live in
`src/platform/cuda.rs`. Planning and emission share these types. Both enums are
non-exhaustive, and the default configuration selects CUDA Hopper. They are
re-exported by `trinity-lowering` and `trinity`.

Implementation selection dispatches first by platform, then by capability inside
the CUDA backend. `PhysicalPlan.target()` retains the platform wrapper, while
`CudaRequirements.target` contains only `CudaTargetCapability`. Callers that used
the former `TargetCapability::Hopper` variant must use the nested form above.

## CUDA emission

`emit(&PhysicalPlan)` returns CUDA source, binding/workspace requirements, and
execution metadata. It does not access the filesystem, compiler, or GPU. CUDA
helpers and templates are embedded in the generated source. Compilation
and execution will be exposed through the future compile/runtime and
Python/PyTorch bindings.

Generated CUDA requires CUDA **13.0+**, the repository's pinned CUTLASS submodule,
and Hopper (`sm_90a`). Multi-GPU execution additionally requires **NVSHMEM 3.7.2**,
direct NVLink peer mappings, cooperative kernel launch, and a world team with
NVLS multicast support for NVLS operations. Cooperative launch is required so
the admission limit is based on a resident Worker grid; NVSHMEM's ordinary-launch
fallback is rejected. Single-GPU sources do not include or link NVSHMEM.
NVLS BF16 matrices require an even column extent (the API transfers packed BF16
pairs). Each matrix is limited to `i32::MAX` elements to keep CuTe global
indexing in range.

### Execution code layout

`src/emit/mod.rs` exposes the existing API through re-exports from `emit::cuda`.
CUDA source, requirements, backend contracts, execution metadata and rendering
live under `src/emit/cuda/`, with `source.rs`, `requirements.rs`, `backend.rs`,
`graph.rs` and `render.rs` separating their responsibilities.

`streamed` submits Action kernels to one CUDA stream and relies on stream order
for dependencies between kernels. `persistent` schedules work inside a resident
Worker kernel. Both execution paths use CUDA streams.

- `src/emit/cuda/streamed/` owns Single-GPU rendering, coordinate tables, kernels,
  input hooks and the host ABI. Kernels run in topological order on one stream.
  It has no task queue, readiness tokens, epoch, initialization kernel, completion
  atomics, or control workspace (`workspace_bytes = 0`, `workspace_alignment = 1`).
- `src/emit/cuda/persistent/` owns Multi-GPU rendering, Worker dispatch, task/readiness
  tables, admission, symmetric control storage, epoch handling and the host ABI.
- `src/emit/cuda/types.cuh` and `src/emit/cuda/templates/` contain shared CUDA
  operand/coordinate types and headers. Backend operation templates remain under
  `src/implementation/cuda/templates/`.

Both paths share the WGMMA tile math, layouts, two-stage load/compute pipeline,
and BF16 output conversion. Operation bodies receive `Bindings`, `Tile`, shared
scratch memory and a runtime hook object, without a scheduler `Context` or
`Task`. Streamed input hooks permit every in-bounds prefetch without polling or
additional CTA barriers. Persistent hooks check readiness and broadcast the
decision across the CTA. Missing future inputs wait only after the current
stage computes. Execution-specific scheduling can evolve behind these hooks.
Rust execution metadata retains dependencies for both paths; only the persistent
renderer emits dependency tables into CUDA.

### Generated launch ABI

Each source contains `trinity::generated::LaunchParams` and three C entry points.
The generated ABI is selected by `CudaRequirements.world_size`:

| Entry point / argument | Single GPU (`world_size == 1`) | Multi GPU |
| --- | --- | --- |
| `LaunchParams` fields | `bindings`, `binding_count`, `stream` | Those fields plus `workspace`, `workspace_bytes`, `worker_count`, `epoch`, verification delay fields |
| `trinity_prepare` | `int(unsigned* maximum_workers)`, reports 1 | Same signature, reports resident Worker limit |
| `trinity_launch` | `int(LaunchParams const*)` | Same signature, different parameter layout |
| `trinity_status` | `int(void* stream)` | `int(void const* workspace, void* stream)` |

The Single-GPU parameter layout and status signature replace the earlier shared
ABI; callers must use the declarations in their newly generated source.

1. `trinity_prepare` validates Hopper and configures shared-memory opt-in. Call
   before stream capture. Multi-GPU preparation also queries occupancy and
   collective-launch limits before the caller chooses a Worker count.
2. `trinity_launch` copies a host array of device pointers in
   `CudaRequirements.buffers` order into kernel arguments. It enqueues work and
   returns immediate launch/precondition errors. Multi-GPU callers must supply
   the same plan, Worker count and monotonically increasing nonzero epoch on
   every rank; the launch is collective.
3. `trinity_status` synchronizes the stream and reports CUDA errors. Multi-GPU
   status also reads the workspace for dispatcher/backend errors and completion.
   Call outside stream capture.

Allocate every binding at its stated byte size/alignment and do not alias
different value IDs. An identity plan can share its input/output value ID.
Symmetric buffers and the multi-GPU workspace must be allocated with NVSHMEM in
identical order/size on every rank; all control flags use CUDA system-scope
atomics, never NVSHMEM atomics. Zero the Multi-GPU workspace **once** before its
first invocation. Single-GPU execution does not allocate or initialize workspace.
Initialize inputs before launch. All allocations must outlive stream completion,
including the launch's final rank barrier. Do not overlap invocations sharing
allocations. Multi-GPU invocations using this world team must also be serialized,
even with different allocations; concurrent invocation/stream scheduling is
outside this initial contract. No intermediate value is overwritten within an
invocation. A Single-GPU graph replays the captured kernels and bindings without
initialization or an epoch parameter.

Positive return codes are CUDA errors. Negative codes are generated runtime
errors (`kInvalidEpoch`, `kPeerUnavailable`, `kCollectiveFailed`, `kInvalidLaunch`,
`kUnsupportedDevice` in the source's `Error` / `PersistentError` enums). The caller
must coordinate preflight failures across ranks before attempting another
collective launch.

### Fusion handoff contract

The first emitter rejects fused Actions and Shared/Register storage. The
`wgmma.cu.j2` template exposes `input_views`, `input_load` and `output_store`
blocks. The shared body calls `runtime.await_stage(stage)` before a required
load and `runtime.prefetch_stage(stage, stage_count)` for nonblocking lookahead.
Both hooks must return a CTA-uniform decision; the streamed implementation is
stateless, and the persistent implementation owns readiness and failure checks.
The two stage slots live until their outstanding cp.async/WGMMA accesses finish.
The FP32 accumulator belongs to one output tile and remains live for all K
stages. `output_store` converts to BF16 before handing the result to any following
operation. Future Shared/Register hooks must retain this rounding boundary,
correct fragment layout and tile lifetime; a value's full logical shape is not
the size of a promoted tile allocation. Fusion owns rewrite/storage decisions and
backend geometry; emit owns these execution and lifetime decisions.

### Testing

Rust tests cover source/binding generation, dependency regions, unsupported plans,
and scheduler interleavings. CUDA compilation and GPU integration tests will use
the future Python/PyTorch bindings, covering numerical correctness, bitwise
transfers, Graph replay, repeated epochs, and communication/GEMM stage overlap.
GPU execution remains unverified in the current environment because the driver
is unavailable.

## Getting Started

```shell
git clone --recursive https://github.com/kaist-ina/trinity-lowering.git
```

## Contribution

```shell
# Install the Git hook and check all tracked files.
pre-commit install
pre-commit run --all-files

# Format changes and run the Rust checks directly.
cargo fmt -p trinity-lowering
cargo clippy -p trinity-lowering --all-targets --locked -- -D warnings
```
