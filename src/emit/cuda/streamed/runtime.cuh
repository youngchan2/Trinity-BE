// All operands are available at kernel entry, including outputs of earlier
// kernels on the same stream. These hooks generate no polling or CTA barriers.
struct StreamedRuntime {
  __device__ bool await_stage(unsigned) const { return true; }
  __device__ bool prefetch_stage(unsigned stage, unsigned stage_count) const {
    return stage < stage_count;
  }
};
