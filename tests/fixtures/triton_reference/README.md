# Original backend reference output

These files were produced by **Trinity/backend/codegen/TritonGen.py**, not by the
Rust emitter. Input IR is pinned in `../analyzer/ffn_cases.txt` (177) and
`../analyzer/vanilla_cases.txt` (591). Shapes are copied from the corresponding
`Trinity/backend/profile/shapes/*_llama.json`, without `constants` overrides.
Generation uses `IRParser().parse(ir)` and `TritonCodeGen().generate(ast, shapes)`.

Original snapshot SHA-256:

- `ffn_177.py`: `208821b82e8c8139b8dad57f93e91879fe1dd1b2304e83450d6f06a475782d18`
- `vanilla_591.py`: `b15fb9a1c1e345686d53769786008b9fa4fd716562ea5ab62864f809de69bf8d`

The FFN reference intentionally retains the old, invalid normalization keepalive
statements. Do not execute these fixtures as corrected kernels.

`tests/compare_triton_reference.py` checks exact ABI/autotune/wrapper launch ASTs
and compares kernel arithmetic, offsets, masks and loop nests after expanding
temporary aliases. It ignores comments, temporary naming and the order of
independent zero initializers in the same scope. Two explicit differences are
allowed: removing artificial `x = x + 0` keepalives and spelling out fp16 casts
at stores that already target fp16 pointers. It does not claim byte-for-byte
source identity or GPU correctness.

`vanilla_falcon_591.py` is also frozen from `Trinity/backend/codegen`, using
`vanilla_falcon_591.ir` and `vanilla_falcon.shapes`. Its 4544-wide contraction
exercises a tail for BLOCK_K=128. The same strict comparison rejects redundant
dot-input `tl.where` and mask reductions; the original emits neither.
