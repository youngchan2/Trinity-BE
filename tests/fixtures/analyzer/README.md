# Analyzer corpus fixtures

These are unchanged, numbered IR entries from Trinity-BE's evaluation corpus.
They are kept here so tests do not depend on a sibling checkout or a running
profiler. Python-generated kernels are evidence for regressions, not numerical
reference implementations.

- `ffn_cases.txt`: IDs 4, 12, 177, 244, 307, 388, 1019, 1230 from
  `backend/evaluation/ffn/llama_ffn_cost6_kern5_wo_scheduler2.txt`.
  Source SHA-256: `0feffe3052fee9726e9dbed71e23fb3c105ceef5645e6b0b423a4e86759f9644`.
- `vanilla_cases.txt`: IDs 543, 591, 3633 from
  `backend/evaluation/vanilla/vanilla_llama_cost6_kern1.txt`.
  Source SHA-256: `22eb5d6c86963bde07f93e19ffa101b29dda6a7ed18fcd5971001fdb7c2754d5`.

The four failing FFN IDs have all normalization accesses in one inner k-loop.
The legacy emitter counted its parent p-loop again and emitted an unnecessary
out-of-loop reference. Tests check exact occurrence scopes; storage and liveness
passes will build on these records in subsequent work.
