// Standalone test wrapper, independent of the production emitter/runtime.
#include <algorithm>

static bool cuda_ok(cudaError_t status) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
  return false;
}

template <class Output>
int check(void (*kernel)(const cutlass::bfloat16_t*, const cutlass::bfloat16_t*, Output*),
          int id, int m, int n, int k, int rows, int columns, int m0, int n0,
          int k0, int step, int iterations, int width, int shared_bytes = 32768) {
  using Input = cutlass::bfloat16_t;
  std::vector<Input> a(m * k), b(k * n);
  std::vector<Output> c(m * n, Output(-123.0f));
  // Dyadic inputs make the FP32 dot products exact, including overlapping K accesses.
  for (int i = 0; i < m * k; ++i) a[i] = Input(float((i * 13 + i / k * 7) % 17 - 8) / 8.0f);
  for (int i = 0; i < k * n; ++i) b[i] = Input(float((i * 11 + i / n * 3) % 19 - 9) / 8.0f);
  Input *da = nullptr, *db = nullptr;
  Output* dc = nullptr;
  if (!cuda_ok(cudaMalloc(&da, a.size() * sizeof(Input))) ||
      !cuda_ok(cudaMalloc(&db, b.size() * sizeof(Input))) ||
      !cuda_ok(cudaMalloc(&dc, c.size() * sizeof(Output)))) return 1;
  if (!cuda_ok(cudaMemcpy(da, a.data(), a.size() * sizeof(Input), cudaMemcpyHostToDevice)) ||
      !cuda_ok(cudaMemcpy(db, b.data(), b.size() * sizeof(Input), cudaMemcpyHostToDevice))) return 1;
  if (shared_bytes > 48 * 1024 &&
      !cuda_ok(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes))) return 1;
  // Repeat with fresh output so accidental loading of C or leaked pipeline state is visible.
  for (int repeat = 0; repeat < 3; ++repeat) {
    std::fill(c.begin(), c.end(), Output(-123.0f));
    if (!cuda_ok(cudaMemcpy(dc, c.data(), c.size() * sizeof(Output), cudaMemcpyHostToDevice))) return 1;
    void* args[] = {&da, &db, &dc};
    if (!cuda_ok(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(1), dim3(128), args, shared_bytes, nullptr)) ||
        !cuda_ok(cudaDeviceSynchronize()) ||
        !cuda_ok(cudaMemcpy(c.data(), dc, c.size() * sizeof(Output), cudaMemcpyDeviceToHost))) return 1;
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        float expected = -123.0f;
        if (row >= m0 && row < m0 + rows && col >= n0 && col < n0 + columns) {
          float sum = 0;
          for (int it = 0; it < iterations; ++it) {
            for (int offset = 0; offset < width; ++offset) {
              const int inner = k0 + it * step + offset;
              sum += float(a[row * k + inner]) * float(b[inner * n + col]);
            }
          }
          expected = float(Output(sum));
        }
        const float actual = float(c[row * n + col]);
        if (actual != expected) {
          std::fprintf(stderr, "case %d [%d,%d]: got %g, expected %g\n", id, row, col, actual, expected);
          return 1;
        }
      }
    }
  }
  return !(cuda_ok(cudaFree(da)) && cuda_ok(cudaFree(db)) && cuda_ok(cudaFree(dc)));
}
