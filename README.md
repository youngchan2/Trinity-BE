# trinity-lowering

`trinity-lowering` provides concrete implementation candidates and validated
physical tensor program graphs.

The `analyzer` module also collects ordered tensor accesses and lexical loop
scopes from extracted Trinity programs. It retains the original syntax together
with the analysis result for subsequent passes. See
[Inductor architecture notes and implementation scope](docs/INDUCTOR_ANALYSIS.md).

```shell
cargo run --example analyze_ir -- tests/fixtures/analyzer/ffn_cases.txt
```

The example accepts either one S-expression or a numbered evaluation list. It
reports incomplete `dummydata` entries as errors. This first pass collects facts.
The `triton` module resolves concrete shapes and symbols into an immutable plan
for per-access register/global bindings, initialization and store placement.
The emitter follows `Trinity/backend/codegen`: tensor names, pointer/stride
arguments, `BLOCK_*`, autotune, `TENSOR_PARAMS`, `BLOCK_PARAMS`, and the named
`forward(...)` interface. The caller supplies all global fp16 tensors; local
arithmetic and accumulators use fp32. The original loop schedule is preserved.

See [the Triton fallback interface, limitations and validation](docs/TRITON_FALLBACK.md).

```shell
cargo run --example emit_triton -- program.ir shapes.txt generated.py tile_k=64 tile_n=128 tile_p=64
```

Regenerated Llama/Falcon FFN and vanilla sources are in `generated_kernels/`
(generated artifacts, ignored by Git). Original Python-backend reference outputs
for regression comparison are in `tests/fixtures/triton_reference/`.

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
