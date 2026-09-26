# Triton fallback provider

Implementation reference as of 2026-09-22. The canonical common
value/access/storage model is [Planning](planning.md); precision is defined in
[the dtype policy](triton-precision.md). Source, compilation and numerical
validation have separate tests described below.

The full scheduled fallback and the operation candidate adapter now meet at a
common `PhysicalPlan`. This connects source import, implementation planning,
kernel emission, buffer allocation and ordered launches. It does not change the
optimizer's computation graph, loop schedule or selected region boundaries.

```text
Scheduled IR text
  → analysis::analyze_text                 source collection
  → PhysicalPlanBuilder::from_scheduled    common typed program
      → ProgramFacts + analysis::storage   infer External / Global / Register
  → TritonKernelProvider::lower_program    backend implementation planning
      → analysis::from_physical            shared occurrence/scope tables
      → ProgramFacts                       logical access regions and dataflow
      → analysis::storage::for_values     retain and validate storage contracts
      → Triton lowering                    TritonPlan + KernelPlans
  → Triton codegen                         kernel_N + forward(...)
```

`lower_source` supplies sample shape/tile bindings before common construction.
Its source convenience entry uses the existing Hopper target default. A caller
that owns the target and bindings can call `from_scheduled` with `ScheduledConfig`
or construct a plan directly. `lower_program` preserves that plan's target.
Neither source generation entry queries a GPU.

The provider and its source backend now share `src/emit/provider/triton/`:
`candidate.rs` owns operation candidates, `program.rs` accepts whole programs,
and `plan.rs`, `lowering/`, `codegen/` retain their existing responsibilities.
The public Rust path `trinity_lowering::triton` remains a re-export of this module.

## Information ownership

| Layer | Information |
| --- | --- |
| `analysis/plan/` | Value identity, allocation/ABI shape, backing storage class, finalized logical/storage dtype, per-access view shape and symbolic dimensions, indices, expressions, ordered regions and loops, inputs/outputs, input mutation and initial recurrence seeds |
| `analysis/ir/` | Common syntax parsing, scheduled occurrence/scope representation and collection, projection from PhysicalPlan |
| `analysis/facts/` | Logical dtype inference, access ranges, shapes, loop/dependency/flow and coverage facts |
| `analysis/storage/` | Common backing-storage inference, materialization requirements, local read bindings, recurrence initialization and publication scope |
| `emit/provider/triton/` | Accept/reject the plan's storage/dtype contracts; pass shared facts into Triton lowering; preserve ABI output order |
| `emit/provider/triton/lowering/` | Padded tile shapes, Triton SSA incoming-value initialization, accumulator representation, FP32 opmath/cast implementation, grids and autotuning configurations |
| `emit/provider/triton/codegen/` | Triton expressions, load/store offsets and masks, loop bodies, kernels, scratch allocation and launch wrapper |

Unannotated source intermediates have `dtype_is_explicit() == false`, but their
`dtype()` is already finalized by common analysis. This flag records provenance,
not unresolved state. Triton consumes all plan dtypes and rejects conflicting
overrides; only register/opmath/cast implementation remains provider-specific
(see [precision](triton-precision.md)).

The common plan stores typed expressions, not a duplicate source AST or a
Triton plan. `analysis::from_physical` builds scope/access tables directly.
`ScheduledIr::ir()` is `None` for this projection. IDs are canonicalized during
common construction; consumers must use the resulting plan's IDs.

## Consuming common storage

[Common storage APIs and semantics](planning.md#공통-분석과-storage) are the source
of truth. `PhysicalPlanBuilder::from_scheduled` and `lower_ir` both use them;
there is no remaining Triton-only source storage classifier.

`TritonKernelProvider::lower_program` projects the common plan and passes its
`ValueInstance.storage` map into `triton::lowering::lower_projected`.
`analysis::storage::for_values` validates those contracts and derives local reads,
initialization and publication facts. `emit/provider/triton/lowering/storage.rs` consumes them
and adds Triton grid/SSA requirements. It does not silently reassign backing
storage. A Global/External value can still have a local accumulator and publish
at the common export scope; this is distinct from making the value Register.

The remaining Triton-specific incoming-value initialization covers a local
assignment inside a loop that is used outside it. The code generator separately
tracks pending global stores and emits CTA barriers before dependent loads when
lane mappings can differ. This is not a new common allocation decision or
cross-CTA synchronization.

## Regions, mutations and symbolic shapes

`Statement::Region` preserves a scheduled source kernel region. Its multiple
stores and sequential loops are offered together, so an accumulator producer
and its consumers are not accidentally split into independent launches.
`LoopKind::Split` retains the parallel part of a normalized mloop, with the
original serial range expressions in its child loop.

Multiple outputs and writes to input cache values remain explicit in the common
plan. The wrapper returns outputs in binding order, does not duplicate an input
argument when it is also an output, and restores mutated global arguments during
Triton autotuning trials.

`ValueInstance::shape()` and `TensorAccess::view_shape` describe sample concrete
shapes. Optional `dimensions` / `view_dimensions` retain the symbolic expressions
needed for runtime dimensions and split scratch capacity. `bind_symbols` creates
a concrete plan, specializing allocation shapes, access views, widths and loop
bounds together. Tile widths remain constants or single configuration symbols;
parameterized arithmetic widths are rejected instead of silently frozen.

## Entry points

Existing `triton::compile`, `compile_analysis` and `lower` now use the provider.
The new Rust entry points also accept the common plan directly:

```rust,ignore
let program = TritonKernelProvider.lower_source(analysis, options)?;
let common = program.physical_plan();
let source = program.emit();
let source_again = emit::emit_triton(common, options_for_this_emission)?;
```

Python, after rebuilding the extension:

```python
from pathlib import Path
import trinity_lowering as tl

ir = Path("tests/fixtures/batched_mla/batched_mla_postprocessed_stage20.txt").read_text()
program = tl.lower_triton(
    ir,
    shapes={"Q": [2, 128, 1, 128], "CKV_cache": [2, 64, 512],
            "W_DK": [512, 16384], "W_DV": [512, 16384]},
    autotune={"max_configs": 1},
)
Path("stage20.py").write_text(program.source)
common = program.physical_plan
source_again = tl.emit_triton(common, autotune={"max_configs": 1})
Path("program.py").write_text(tl.emit_python(common))
```

Import `stage20.py` and call `forward(...)` with its named CUDA tensor arguments,
or import `program.py`, call `executable = prepare(inputs)`, then
`result = executable(inputs)`. The latter accepts the common plan's input names.
`emit_triton` generates source; it does not launch or benchmark during generation.

## Selection boundary and current limits

These are separate implemented entry paths, not one integrated autotuner:

| Entry | Scope and output | Selection |
| --- | --- | --- |
| `TritonKernelProvider::lower_source/lower_program`, `emit_triton` | Whole scheduled program → `TritonPlan` → kernel source + ordered `forward` | Direct fallback, no other provider comparison |
| `kernel_candidates` | One `OperationId` in its loop context → alternatives/rejections | Discovery only; no import, compilation or benchmark |
| `emit_python` independent-operation path | Loop-free, single-output, resolved-dtype full-memory operations with supported reference | Python source can compare executable Triton/Quack candidates |
| Native `emit` | Supported CuTe operations → combined CUDA bodies → `CudaSource` | Current native priority selection; no Triton/Quack timing comparison |

`emit_python` switches to the whole-program Triton path for regions/loops,
multiple outputs, input mutation, or expressions
unsupported by its independent PyTorch reference. Its `comparison: not_performed`
report is intentional; `prepare` in that path does not certify accuracy.


- Independent loop-free full-tensor operations retain the existing
  CuTe/Triton/Quack candidate enumeration. Python execution compares supported
  Triton/Quack calls against a PyTorch reference and benchmarks them.
- Scheduled regions, multiple outputs, input mutations and extended expressions
  use the complete Triton fallback in `emit_python`. `prepare` binds these inputs;
  calling the executable compiles/tunes/launches through Triton. Its reports say
  `comparison: not_performed`: it does not claim cross-provider comparison or an
  independent accuracy check.
- `emit(plan)` remains the native CUDA entry, not an automatic Python fallback.
  Native split-loop composition, communication and explicit Shared transport are
  not enabled by this integration. Triton rejects communication and Shared
  transport; explicit Register values cannot silently escape into global memory.
- Triton retains its existing operator/grid restrictions. This integration does
  not claim that every semantically valid optimizer output is supported.

`tests/triton_provider.rs` covers direct Builder input, views, symbolic binding,
MLA access/loop preservation, cache mutation, multiple outputs, initialization
dependencies and Python syntax. The existing MLA, precision and autotuning tests
also run through this provider path. CUDA execution is a separate validation step.

`tests/common_storage.rs` checks local values, cross-region values, materialized
views, local subtile reads, accumulator initialization, explicit storage
contracts and MLA specialization. A shared inferred Register plan is emitted
through both CuTe and Triton, checking that neither allocates its scratch buffer.

`tests/triton_provider_python.py --offline` checks the Python API and captures
the generated launch arguments using CPU tensors, then compiles their actual
Triton signatures to CUDA PTX/cubin for an explicit target. It needs PyTorch,
Triton and the built extension, but no GPU driver. This checks compilation and
launch ABI, not numerical execution. With an uninstalled Rust build:

```sh
TRITON_CACHE_DIR=/tmp/trinity-provider-cache python tests/triton_provider_python.py \
  --compiler-path target/debug/libtrinity_lowering.so --offline --target 90
```

## Concrete fallback restrictions

- Single GPU only. Explicit Shared transport and communication are rejected.
- At most three parallel grid axes in one enclosing chain per source kernel.
  Sibling ploop regions inside one kernel and ploop nested inside sloop require
  an explicit source/adapter split; the provider does not invent one.
- Parallel grid bounds must be program-wide scalar parameters. Sequential
  arithmetic bounds can reference valid enclosing loop coordinates.
- View capacity, access coverage and writer ownership must be provable by the
  current common analysis. This is not a full SSA/alias/liveness solver.
- Parameterized tile widths are single symbols or constants. Binding and tuning
  cannot make an unsupported access/shape valid by changing its meaning.
- Triton tensor element/rank/operator constraints remain. Dot uses padded
  supported shapes; small dimensions use the existing multiply/reduce path.
- FP16/BF16/FP32 storage is supported. Raw-exp overflow is an IR/precision-domain
  limitation; see the accepted assumptions in [precision](triton-precision.md).
- Source/target compilation is not proof of finite outputs, numerical agreement,
  GPU resource feasibility on another target, or good performance.

Sources: [program provider](../../src/emit/provider/triton/program.rs),
[grid validation](../../src/emit/provider/triton/lowering/loops.rs),
[kernel specialization](../../src/emit/provider/triton/lowering/storage.rs),
[expression lowering](../../src/emit/provider/triton/lowering/expression.rs).

## Reuse by Quack and CuTe

Reuse common `PhysicalPlan` expressions/accesses and the APIs in
[Planning](planning.md#공통-분석과-storage). The original source text is not needed
once the plan owns those expressions. Backend-specific layout/code is not
available from `ValueInstance.storage` alone.

The in-crate `KernelProvider`/`KernelContext` in
[provider/mod.rs](../../src/emit/provider/mod.rs) provide candidate discovery without
native rendering requirements. `NativeKernelProvider`, `SpecifiedKernel`,
phase code, CTA requirements and register bindings are owned by
[native/mod.rs](../../src/emit/native/mod.rs) and
[native/interface.rs](../../src/emit/native/interface.rs). Triton/Quack opaque
providers implement only the common discovery trait. These are not a public
third-party plugin ABI. Adding provider variants/registration still modifies the
crate's emission pipeline. `KernelCandidate` currently covers exactly one
operation; it does not yet express a multi-operation GEMM+epilogue region.

[KernelRequest](../../src/emit/request.rs) is the independent host-call boundary:
it retains the operation expression and tensor IDs/types/shapes, but currently
accepts only single-GPU, loop-free, positive full-tensor External/Global accesses
and non-in-place outputs. Common preparation checks bound configuration symbols;
it does not build CUDA buffers or impose native output/mutation limits. Native
ABI restrictions and buffer slots are prepared only in the CUDA emission path.
General region/loop coverage and CTA-local register
ports require further integration, not a parallel copy of Triton's analyzer.

The current [Quack adapter](../../src/emit/provider/quack/mod.rs) recognizes one
store of GEMM with optional residual/column bias and ReLU. Its checks restrict it
to Hopper/SM120, BF16 matrix inputs, BF16/FP32 storage, K/N multiples of 8,
full-matrix accesses without changed views, and no output/input alias.
It emits lazy `gemm`/`gemm_add`/`gemm_act` host calls. This describes source support
checks; installed Quack API compatibility and GPU correctness remain unverified
by these source-level contracts.

CuTe owns thread/register layout, instruction choice, pipeline and scratch/barrier
requirements. Native composition handles compatible ports, iteration ownership
and resource placement. It is not automatically launchable through
`KernelCandidate::emit_python`, which returns no wrapper for Native fragments.
See [Emission](emission.md) for that path.

**Open integration decisions:** region/multiple-operation candidate coverage; Native-versus-Triton rounding and
reference tolerances; library stride/layout/workspace adaptation; combined
Native/Opaque execution and benchmark selection. These are not completed APIs.


## Register views and concatenation

Common `LocalRead.split_last` proves ownership of a full last axis before a
two-factor view is read. Triton emits reshape/gather and preserves local FP32
values; non-power-of-two factors are rejected to preserve padding positions.

Equal-width unpadded concat uses ordered `tl.cat`. Unequal or padded concat
retains the generic gather path. This avoids manufacturing repeated gather users
in the partial-RoPE case that triggered Triton 3.8's thread-locality assertion.
IR concatenation order and the original view semantics remain unchanged.
