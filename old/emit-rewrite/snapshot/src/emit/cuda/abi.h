#ifndef TRINITY_CUDA_HOST_ABI_H
#define TRINITY_CUDA_HOST_ABI_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

// This header is embedded verbatim into artifacts and shared by the native loader.
// No CUDA, PyTorch, or Rust types cross this boundary.
namespace trinity::abi {
inline constexpr std::uint32_t version = 1;

enum Domain : std::uint32_t { none, generated, runtime, driver, transport };
enum Stage : std::uint32_t { idle, validate, prepare, configure, registration,
                            initialization, launch, completion, release };

struct ErrorInfo {
  std::uint32_t domain = none, stage = idle;
  std::int32_t code = 0;
  std::uint32_t submitted = 0;
  std::uint32_t cleanup_domain = none;
  std::int32_t cleanup_code = 0;
};

struct StreamedLaunch {
  void* const* bindings;
  std::size_t binding_count;
  void* stream;
};

struct PersistentLaunch {
  void* const* bindings;
  std::size_t binding_count;
  void* workspace;
  std::size_t workspace_bytes;
  void* stream;
  unsigned worker_count;
  int delay_rank = -1;
  int delay_task = -1;
  unsigned long long delay_cycles = 0;
};

struct Descriptor {
  std::uint32_t version, mode, pointer_size, error_size, error_alignment;
  std::uint32_t launch_size, launch_alignment, field_count;
  std::uint32_t offsets[9];
  std::uint32_t error_offsets[6];
  char const* requirements;
  std::size_t requirements_size;
};

inline Descriptor descriptor(unsigned mode, char const* metadata, std::size_t size) {
  Descriptor d{version, mode, sizeof(void*), sizeof(ErrorInfo), alignof(ErrorInfo),
               0, 0, 0, {}, {}, metadata, size};

  std::uint32_t errors[] = {offsetof(ErrorInfo, domain), offsetof(ErrorInfo, stage),
    offsetof(ErrorInfo, code), offsetof(ErrorInfo, submitted),
    offsetof(ErrorInfo, cleanup_domain), offsetof(ErrorInfo, cleanup_code)};
  for (unsigned i = 0; i < 6; ++i) d.error_offsets[i] = errors[i];

  if (mode == 1) {
    d.launch_size = sizeof(StreamedLaunch); d.launch_alignment = alignof(StreamedLaunch);
    d.field_count = 3;
    d.offsets[0] = offsetof(StreamedLaunch, bindings);
    d.offsets[1] = offsetof(StreamedLaunch, binding_count);
    d.offsets[2] = offsetof(StreamedLaunch, stream);
  } else {
    d.launch_size = sizeof(PersistentLaunch); d.launch_alignment = alignof(PersistentLaunch);
    d.field_count = 9;
    std::uint32_t offsets[] = {offsetof(PersistentLaunch, bindings),
      offsetof(PersistentLaunch, binding_count), offsetof(PersistentLaunch, workspace),
      offsetof(PersistentLaunch, workspace_bytes), offsetof(PersistentLaunch, stream),
      offsetof(PersistentLaunch, worker_count), offsetof(PersistentLaunch, delay_rank),
      offsetof(PersistentLaunch, delay_task), offsetof(PersistentLaunch, delay_cycles)};
    for (unsigned i = 0; i < 9; ++i) d.offsets[i] = offsets[i];
  }

  return d;
}

inline int report(ErrorInfo* out, Domain domain, int code, Stage stage,
                  bool submitted = false) {
  if (out) *out = {code ? domain : none, stage, code, unsigned(submitted), none, 0};
  return code;
}

static_assert(std::is_standard_layout_v<ErrorInfo> && std::is_standard_layout_v<Descriptor>);
} // namespace trinity::abi
#endif
