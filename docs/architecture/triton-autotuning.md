# Triton fallback autotuning

Triton lowering produces typed `TuningConfig` records alongside storage and
precision decisions. Codegen renders those records; it does not discover a new
schedule. `TritonPlan::tuning_configs(kernel_index)` exposes the selected search.

## Search policy

- Existing symbolic loop steps: baseline from `Options::symbols`, then
  16, 32, 64, 128, 256 where the sample loop extent permits. The baseline may be
  smaller, e.g. 1 for a batch axis. Fixed numeric IR steps/tiles remain fixed.
- `Options::tuning[symbol]` replaces that symbol's default choices exactly.
- `num_warps`: 4, 8, 2. `num_stages`: 1, 2, 3 for kernels containing GEMM;
  pointwise/reduction-only kernels use the first stage setting.
- Default maximum: 64 configurations per kernel, including launch parameters.
  Large Cartesian tile spaces are sampled (up to 512 joint assignments plus
  single-axis variations), then validated. The final launch combinations are
  sampled across the valid list, retaining its first and last entries.
- An mloop split parameter is tuned only by its owner. Scratch is allocated for
  the maximum legal candidate split; subsequent kernels receive the owner's
  actual selected split value.

```rust
use trinity_lowering::triton::{AutotuneOptions, Options, TensorDType};

let mut options = Options {
    default_dtype: TensorDType::Bf16,
    autotune: AutotuneOptions {
        max_configs: 128,
        num_warps: vec![4, 8, 2],
        num_stages: vec![1, 2, 3, 4],
    },
    ..Options::default()
};
options.tuning.insert("tile_k".into(), vec![32, 64, 128, 256]);
```

`max_configs = 1` validates only the first requested assignment and uses the first launch settings;
without explicit tuning overrides that is the baseline. The search finds the
fastest **tested** candidate on that device, not a global optimum. Compilation
and tuning cost increases with the search budget. Regenerate source files after
changing lowering options or the emitter; already generated Python is unchanged.

## Candidate validation and execution

`src/triton/lowering/tuning.rs` reuses lowering with each tile assignment, without
recursively planning another search. It rejects invalid broadcasts, dot shapes,
producer coverage, or ownership. Register/global storage, local-view bindings,
and allocation shapes must match the baseline plan. A variable step cannot
change independently of a fixed-width region it addresses: that could create
holes or overlap. This is intentionally conservative; it searches variants of
one implementation, not alternative storage strategies.

Launch-time pruning checks the actual shape bounds and mloop whole-chunk
contract. No surviving candidate gives an explicit error. Triton's autotuner
compiles/benchmarks the survivors and handles `OutOfResources` configurations;
if every configuration fails, the final launch still fails. There is no claim of
an exact shared-memory estimate before Triton compilation.

A globally read-and-written tensor is included in `restore_value`: Triton saves
and restores its contents around benchmark invocations. This covers mutable
inputs/caches and earlier-kernel partial sums. The final selected execution runs
once on the original state. Write-only output buffers need no restore, including outputs whose self-read
is satisfied by a zero-initialized register accumulator. Avoiding those copies
keeps unnecessary memory traffic out of the benchmark.

The tuning key includes shape parameters, strides, and incoming split parameters;
Triton also includes tensor dtypes in its key. An autotuner caches the selection
for repeated launches in that process. Regenerate or reload for a new benchmark
session; this implementation adds no persistent tuning database.

## Shared-memory limits

A symbolic tile can become smaller and reduce resource pressure. A full-tile K
axis with a fixed numeric extent cannot become a K loop as part of this search.
For example, a `16 x 16384 @ 16384 x 64` IR can remain unsupported on a 99 KiB
shared-memory device even with more configurations. Num stages and warp count
also affect compiler resource use. FP16 and BF16 both use two bytes of storage;
the dtype switch alone does not solve the original shared-memory problem.

## Tests

`tests/triton_tuning.rs` checks search diversity/budget, explicit overrides,
fixed-tile/shape constraints, and mutation restoration. The ignored GPU test
emits a tail-masked GEMM and checks **all 64 candidates** against PyTorch, then
checks that an in-place increment executes exactly once after both cold tuning
and a cached launch. A split-sum case also checks every legal mloop candidate,
scratch capacity, and the selected split count reaching the consumer kernel.
Results include the chosen configuration and tuning timings
in `target/tests/triton_tuning/gpu_results.json`.

Performance timing while another process uses the GPU is not evidence of the
best configuration under exclusive use. Correctness checks remain useful;
rerun tuning on an idle device for performance selection.

References: [Triton autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html),
[PyTorch GEMM template](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/kernel/mm.py).
