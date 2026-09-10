# trinity-lowering

Rust analysis and Triton source generation for extracted, scheduled Trinity IR.
The emitter preserves the IR's computation graph and loop schedule. Named views,
keyed indices, split loops, and expression-valued loop bounds are lowered into
Triton kernels and a Python `forward(...)` launch wrapper.

## Source layout

```text
src/analysis/          IR parsing, tensor accesses, lexical scopes, dependencies
src/triton/plan.rs     ProgramPlan and per-kernel lowering results
src/triton/shape.rs    Access shapes and index expressions
src/triton/lowering/  Storage, initialization, loop/grid and parameter planning
src/triton/codegen/   Kernel bodies, addresses, autotuning and launch wrappers
tests/batched_mla_emit.rs
                      Stage 14/16/20 file emission and Python syntax checks
tests/fixtures/batched_mla/
                      Supplied postprocessed IR files and input shapes
```

The separate CUDA implementation and physical-plan APIs remain in
`src/implementation/` and `src/physical/`, with their unit tests under `src/tests`.
They are independent of the current Triton emitter.

## Rust API

For Rust callers, use `triton::compile(text, options)` or
`analysis::analyze_text(text)` followed by `triton::lower(analysis, options)` and
`ProgramPlan::emit()`.

## Emission tests

```shell
cargo test --locked --test batched_mla_emit -- --nocapture
```

The three tests read the checked-in stage 14/16/20 `.txt` fixtures and write
`stage14.py`, `stage16.py`, and `stage20.py` to `target/tests/batched_mla/`.
They check Python syntax, the expected kernel count, and the launch wrapper.
Python 3 is required; PyTorch, Triton, and a GPU are not used by these tests.
These are source-generation checks, not numerical or performance tests.

`generated_kernels/` and `reference_kernels/` are ignored artifacts and are not
changed by the tests.

## Development checks

```shell
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

The repository also provides pre-commit hooks in `.pre-commit-config.yaml`.
