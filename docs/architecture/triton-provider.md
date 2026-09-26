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

These remain distinct paths; Native CUDA is not in the Python autotuner:

| Entry | Scope and output | Selection |
| --- | --- | --- |
| `TritonKernelProvider::lower_source/lower_program`, `emit_triton` | Whole scheduled program → `TritonPlan` → source + ordered `forward` | Direct fallback |
| `kernel_candidates` | One operation → implementations/rejections | Discovery only |
| `region_candidates` | Complete region → original Triton kernel or Quack API call | Source/discovery only |
| `emit_python` independent-operation path | Supported loop-free memory operations | Correctness and timing comparison |
| `emit_python` region path | Eligible scheduled regions, fixed shapes, one output, no mutation | Triton/Quack comparison per complete region |
| Native `emit` | CuTe operations → combined CUDA bodies → `CudaSource` | Native priority selection |

`TritonPlan::emit_region(index)` emits the existing kernel with `run(values)` for
caller-owned global buffers. It reuses ordinary `kernel_launch`, preserving the
original schedule, tile policy and source body. Cross-kernel split tuning is not
supported by this adapter; `emit_triton` retains its existing full-program path.

Scheduled `emit_python` tries the region comparison path when at least one Quack
candidate exists and each region has an implementation plus a reference (or
direct Triton fallback without comparison). Providers receive original
`RegionFacts`; common classification/summary success is optional. A Triton
lowering rejection does not prevent a Quack-only region with a reference from
being used. Quack never substitutes a fragment inside a larger region.
The [Quack provider](quack-provider.md) documents matching, preparation cost,
reference, selection and remaining restrictions. A region lacking a reference
executes Triton directly with `comparison: not_performed`.

Multiple outputs, input mutation, cross-kernel split tuning or no eligible
Quack region retain the whole-program `triton_program` path. Its `prepare` binds
inputs, and invoking the executable compiles/tunes/launches through Triton; it
does not claim cross-provider comparison or independent correctness validation.
The manifest retains region pattern/candidate rejection information.

`emit(plan)` remains the Native CUDA entry. Native split-loop composition,
communication and explicit Shared transport are not enabled by this integration.
Triton rejects communication and Shared transport; explicit Register values
cannot silently escape into global memory. Existing Triton grid/operator
restrictions remain; not every valid optimizer output is supported.

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
- Common `LocalRead.split_last` proves a full last-axis tile can be read through
  a two-factor view. Triton currently requires power-of-two factors, emits the
  corresponding reshape/gather and preserves FP32 local values. Non-power-of-two
  views can remain meaningful for other providers despite this Triton restriction.
- Equal-width, unpadded concat uses ordered `tl.cat`; unequal/padded concat keeps
  the generic gather path. This avoids the Triton 3.8 gather-layout assertion
  observed in nested partial-RoPE concat without changing the IR or compiler passes.
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
it records shared operation scopes without classifying computations. Quack
discovery supplies its own optional `emit::QuackPatternAnalysis` diagnostics. It does
not build CUDA buffers or impose native output/mutation limits. Native
ABI restrictions and buffer slots are prepared only in the CUDA emission path.
General region/loop coverage and CTA-local register
ports require further integration, not a parallel copy of Triton's analyzer.

The Quack-owned [pattern analysis](quack-provider.md#quack-연산-패턴-인식) classifies complete
scheduled kernel regions as `SingleGemm`, `GemmEpilogue`, or `Other`, with a
reason for Other. Leading nested ploops share one region result. Ordered stores,
K-loop accumulation and post-loop pointwise epilogues are analyzed together;
no part of a larger region is advertised as its complete implementation.
`OperationCandidates.region_pattern` is a Quack diagnostic; `scope` is the shared original region location;
independent-operation manifests use `region_pattern` and `scope`. Scheduled
fallback manifests use `region_patterns`, including all covered operation IDs,
GEMM/accumulation operations, reduction loop, initialization and epilogue stores.
Region classification is not region-wide candidate comparison: candidate
execution coverage still follows the existing operation contract.

The current [Quack adapter](../../src/emit/provider/quack/mod.rs) consumes that
classification only when the entire region is one non-recurrent store, and
lowers the supported subset: identity, optional residual/column
bias and ReLU. Other classified pointwise epilogues retain their classification
but receive a provider rejection. Its checks restrict it
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
