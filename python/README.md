# Trinity Python/PyTorch bindings

Installation and native execution examples are documented in the
[project README](../README.md#getting-started).
See [Planning](../docs/architecture/planning.md) for the shared plan contract and
[Triton provider](../docs/architecture/triton-provider.md) for source generation.

`lower_ir` preserves the input program's loop structure and uses the common
storage analyzer to choose `external`, `global`, or `register` for each value.
Local intermediates no longer automatically acquire global buffers.
