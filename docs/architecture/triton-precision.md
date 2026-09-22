# Triton fallback precision

## Current policy (2026-09-22)

Logical/storage dtype, register computation, GEMM operands and accumulation are
separate decisions. The accepted policy assumes values narrowed to FP16/BF16 at
GEMM/storage boundaries fit the requested type. Missing numerical stabilization
in source IR is not repaired by widening every downstream GEMM or inserting max
reductions. Raw exponentials can therefore overflow in FP16.

```rust
use trinity_lowering::triton::{Options, TensorDType};

let options = Options {
    default_dtype: TensorDType::Bf16,
    ..Options::default()
};
```

`default_dtype` selects unannotated boundary values and is the fallback type for
unconstrained intermediates; it defaults to FP16. `Options::dtypes` supplies
explicit logical/storage contracts, including typed PhysicalPlan values.
`TritonPlan::tensor_dtype(id)` reports the resolved logical/storage dtype, not
the dtype of its temporary register representation.

| Value/use | Policy |
| --- | --- |
| Input, output, mutable input cache | Declared dtype, otherwise `default_dtype` |
| Unannotated intermediate | Propagate producer's logical operand types; FP32 opmath alone does not promote storage |
| Register load and ordinary assignment | FP32; no implicit half rounding just because a value is named |
| Pointwise / exp / division | FP32 computation for half inputs |
| Reduction and loop-carried accumulator | FP32 computation; materialization still uses the logical/storage dtype |
| GEMM inputs | Logical operand dtype, including inline exp/reduction/division/dot results |
| Explicit cast | Preserve the rounding; then use FP32 opmath. Cast type controls a subsequent GEMM |
| Differing GEMM operand types | Existing fallback promotion to FP32; this is not a claim that PyTorch matmul accepts mixed input types |
| GEMM accumulation/result in registers | Explicit `out_dtype=tl.float32` |
| Small dot dimensions below 16 | Existing FP32 multiply/reduce implementation |
| Explicit/logically FP32 dot inputs | Existing `input_precision='ieee'` |
| Global store and wrapper allocation | Resolved logical/storage dtype, including inter-kernel scratch |

`analysis::dtype::resolve` in [analysis/facts/dtype.rs](../../src/analysis/facts/dtype.rs)
propagates logical types forward through
unannotated intermediate definitions to a fixed point. Unary operations,
reductions and views preserve their operand's logical type; binary operations
and dots merge operand types. Recurrence inference begins without assuming a
half type for unknown intermediates, preventing false mixed-type promotions.
Explicit tensor contracts remain fixed. There is no backward request to widen
logits or other producers merely because exp or reduction consumes them.

For example, FP16 `X -> exp -> E -> GEMM` computes exp in FP32, keeps an in-kernel
E register in FP32, and converts E to FP16 at the dot boundary. If E is
materialized, its allocation/store is also FP16. An explicit FP32 X, E contract,
or operand cast instead preserves FP32 through the corresponding logical path.
A register accumulator being FP32 does not, by itself, make it an FP32 input to
a later GEMM. Ordinary GEMMs use 16-bit operands and FP32 accumulation.

## Cast locations and ownership

| Boundary | Emitted behavior | Reason / source |
| --- | --- | --- |
| Global/local load | Floating values enter FP32 registers | Half opmath; `codegen/indexing.rs`, `local.rs` |
| Named register assignment | Retain FP32; no cast back to logical half storage type | A local SSA assignment is not global materialization; `codegen/kernel.rs` |
| Explicit IR cast | Round to requested type; FP16/BF16 casts then upcast for following opmath | Preserve requested rounding; `codegen/ops.rs` |
| `tl.dot` input | Cast both operands to the resolved common logical type | FP16/BF16 GEMM with FP32 accumulation; explicit FP32 and existing mixed-type promotion remain |
| Global store | Cast to `tensor_dtype`; wrapper allocates that same dtype | Preserve ABI/materialized-buffer contract |

`tl.dot` currently emits `out_dtype=tl.float32`; surrounding IR addition is still
emitted separately. Folding a recurrence into the third `acc` argument is not
implemented by this policy change. Small-dot multiply/reduce remains its existing
FP32 path; do not infer Tensor Core use merely from a logical FP16/BF16 type.
The supported storage type set is FP16/BF16/FP32, not a general PyTorch type system.

## Provider integration boundary

**Implemented in common planning:** logical tensor/storage types are resolved
by `analysis::dtype::resolve` before `PhysicalPlanBuilder::from_scheduled`
constructs the plan. `ValueInstance::dtype()` is final for every provider;
`dtype_is_explicit() == false` describes inference provenance only. Native and
independent candidate preparation no longer reject values merely for being inferred.

Triton's program provider imports every finalized dtype and rejects conflicting
`Options::dtypes` overrides, including inferred intermediates. Change source
configuration and rebuild the common plan to request different logical types.
[precision.rs](../../src/emit/provider/triton/lowering/precision.rs) only queries types of
lowered inline expressions from these contracts; shared merge/cast rules come
from `analysis::dtype`. It no longer runs tensor dtype inference. FP32 opmath,
accumulator representation and the actual cast locations remain in Triton.

**Open for cross-provider work:** CuTe Native currently emits declared-type
rounding at some register producer/consumer boundaries, while Triton keeps
ordinary register intermediates in FP32. See the Native test
`forwards_registers_in_each_producers_output_scope_with_rounding_and_fanout` in
[combine tests](../../src/emit/native/combine/tests.rs). Quack's actual rounding depends
on the selected library API and has not been verified against this policy.
Common `Storage::Register` does not establish bitwise equivalence between these
implementations. New candidate comparisons must explicitly decide logical dtype,
rounding boundaries and accuracy criteria before treating them as equivalent.
Sharing logical dtype resolution does not resolve those register-rounding
choices. This refactor preserves the existing Triton numerical policy; it does
not establish numerical equivalence with CuTe or Quack.

## Numerical limitations and previous fixes

The Python backend's `226068e` fix removed premature FP16 register assignments,
while retaining FP16 global storage and dot inputs. The Rust extended-IR path
had reintroduced that register narrowing. Register assignments now retain FP32
opmath, independently of storage dtype.

Earlier Rust fixes also promoted exp/reduction/division intermediates and their
producers to FP32 storage, and subsequently made their GEMM consumers FP32. That
was a stronger numerical policy than the original backend and increased memory
and GEMM costs. It is superseded by the explicit logical/compute separation above.
The BF16 dot fix remains: a BF16 operand must not silently become FP16.

`exp(scores) @ V / sum(exp(scores))` has an additional limitation: raw exp can
exceed FP16's 65504 maximum before the dot input cast. FP32 accumulation and
`tl.dot(..., acc)` cannot recover an operand that is already infinite. The
emitter preserves this IR and can produce nonfinite results; it does not claim
that the new policy makes QKNorm numerically safe. Tests retain these cases as
expected numerical limitations, not successful accuracy checks. BF16 has a
wider exponent range but also does not guarantee finite arbitrary exponentials.

Stable or online softmax must be represented by the optimizer's IR or an
explicit implementation choice. This change does not add max subtraction,
reorder division, change kernel/loop boundaries or modify autotuning.

## Inductor reference

The inspected reference is PyTorch **2.8.0+cu128**. Its Triton backend separates
`triton_compute_type`, `triton_store_type` and reduction accumulation types.
Half inputs use FP32 opmath by default, while graph types govern materialization
and GEMM operands. `mm_common.py::acc_type` selects FP32 GEMM accumulation for
FP16/BF16; TF32 selection is a separate precision setting.

Inductor receives a typed graph. It does not generally infer FP32 storage or an
FP32 GEMM just because an upstream exp/reduction computes in FP32. Trinity's
untyped source path propagates logical types as described above; it does not
claim complete equivalence with PyTorch's type system or fusion rounding.

## Validation

`tests/logical_dtype.rs` verifies common inference, recurrence anchoring, explicit
contracts/scalar defaults and provider rejection of conflicting inferred types.
`tests/triton_precision.rs` covers half storage versus FP32 opmath, register
assignments, cross-kernel stores, explicit FP32/cast contracts, BF16 inputs,
inline and materialized exp-to-dot, reductions, nested dots and accumulator reuse.
The ignored CUDA test uses stage-by-stage PyTorch references with declared
storage rounding. It also retains out-of-range raw-exp inputs, a BF16 GEMM
outside FP16's range, and register arithmetic where 40000 + 40000 exceeds FP16
before division returns the output to range. Run with `TRINITY_TEST_PYTHON` and
`CUDA_VISIBLE_DEVICES` set for the intended environment.

The tiled-store/full-tile-read barrier regression remains separate from precision;
it verifies publication between lane layouts within a CTA, not cross-CTA ordering.

Source generation, target compilation and numerical execution are different
checks. The tests above define reproducible checks; their presence does not
establish GPU correctness for every IR, shape or tuning configuration.
Historical runs using the superseded FP32-promotion policy do not validate
the current logical-dtype policy.

## References

- [PyTorch 2.8 Triton codegen](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/triton.py)
- [PyTorch 2.8 GEMM options](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/kernel/mm_common.py)
- [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

## Historical validation: superseded FP32-promotion policy

The v4 corpus regenerated and passed Python syntax checks: FP16 390 IRs / 872
kernels, plus BF16 QKNorm 100 IRs / 140 kernels. Of the FP16 kernels, 593 have 64
candidate configurations; the remaining kernels have 3–45, according to their
IR constraints. These counts describe emission, not full-corpus execution.

On GPU 0 (RTX PRO 5000 Blackwell), QKNorm expr_1, expr_7, expr_75 each passed
FP16 and BF16 execution, repeatability, and a canonical PyTorch reference
comparison (`rtol=0.05`, `atol=0.001`). Maximum absolute errors were about
0.0000882 (FP16) and 0.0004733 (BF16). The reference keeps normalization in FP32
and rounds external cache updates to the model dtype. These checks used the
baseline tile; the separate small-GEMM test exercises all 64 tuning candidates.

Workspace artifacts: `data_triton_validation/dtype_autotune_fp16/`,
`data_triton_validation/dtype_autotune_bf16/`, and
`data_triton_validation/dtype_autotune_reference_samples.json` under the enclosing
project root. An independent process was using GPU 0, so this run establishes
correctness and tuner operation, not isolated performance or a speedup.

### Full execution smoke check (2026-09-21)

The same generated sources were checked on physical GPU 0 without timed
autotuning or requiring exclusive GPU use. Each passing case completed cold and
warm executions with restored inputs, finite outputs, and exactly repeatable
outputs. This is not a full-corpus PyTorch reference comparison or a check of
every emitted configuration.

| Model dtype / corpus | Passed | Shared-memory limit | Compiler pass failure | Compile timeout |
| --- | ---: | ---: | ---: | ---: |
| FP16 vanilla (113) | 111 | 2 | 0 | 0 |
| FP16 RoCo (57) | 47 | 0 | 6 | 4 |
| FP16 QKNorm (100) | 100 | 0 | 0 | 0 |
| FP16 FFN (120) | 118 | 2 | 0 | 0 |
| BF16 QKNorm (100) | 100 | 0 | 0 | 0 |

RoCo expr_96/99/111/120/122/126 fail in Triton 3.4's
`OptimizeThreadLocality` pass (`loopResult.hasOneUse()`); four of those were
also checked with an emitted smaller tile and reproduced the failure. RoCo
expr_36/44/75/76 exceeded the 900-second limit in CUDA compilation. These
remain unresolved and must not be represented as successful execution.

FFN expr_55/95 hit the measured 2 MiB shared-memory requirement even with an
existing `tile_n=16, num_warps=8` candidate (device limit: 101,376 bytes).
Their baseline compile timeout/stopped attempts are preserved separately.
Resource search was bounded; this does not prove that every configuration fails.
The workspace `data_triton_validation/dtype_execution_report.json` and `.md`
contain source hashes, per-case results, failure logs, and the exact test scope.
