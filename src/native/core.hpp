#pragma once
#include "abi.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace trinity::native {
struct ResourceBusy : std::runtime_error { using std::runtime_error::runtime_error; };

struct RuntimeFailure : std::runtime_error {
  abi::ErrorInfo info;
  int rank;

  RuntimeFailure(std::string message, abi::ErrorInfo error = {}, int r = -1)
      : std::runtime_error(std::move(message)), info(error), rank(r) {}
};

inline void check(int result, abi::ErrorInfo error, std::string const& operation, int rank = -1) {
  if (result) throw RuntimeFailure(operation + " failed (domain=" + std::to_string(error.domain) +
      ", code=" + std::to_string(error.code) + ")", error, rank);
}

template<class T> T symbol(void* handle, char const* name) {
  dlerror();
  auto address = dlsym(handle, name);
  auto error = dlerror();
  if (error || !address) throw RuntimeFailure(std::string("missing symbol ") + name + ": " + (error ? error : "null"));

  return reinterpret_cast<T>(address);
}

// Does not unload implicitly once preparation has begun: a failed registration
// may have left device code reachable. Call release() before close().
class Library {
  void* handle_ = nullptr;
  std::string directory_;
  bool release_required_ = false;

public:
  abi::Descriptor descriptor{};
  std::string metadata;

  using Prepare = int(*)(unsigned*, abi::ErrorInfo*);
  using Release = int(*)(abi::ErrorInfo*);

  Prepare prepare_fn = nullptr;
  Release release_fn = nullptr;

  explicit Library(std::string const& path, unsigned mode) {
    char pattern[] = "/tmp/trinity-module-XXXXXX";
    auto directory = mkdtemp(pattern);
    if (!directory) throw RuntimeFailure("cannot create private module directory");
    directory_ = directory;

    try {
      auto copy = directory_ + "/program.so";
      // Avoid std::filesystem symbols exported by libtorch (some wheel builds
      // provide incomplete filesystem shims). This directory contains one file.
      std::ifstream input(path, std::ios::binary);
      std::ofstream output(copy, std::ios::binary | std::ios::trunc);
      if (!input || !output) throw RuntimeFailure("cannot copy artifact library");
      output << input.rdbuf();
      output.close();
      if (input.bad() || !output) throw RuntimeFailure("artifact copy failed");

      handle_ = dlopen(copy.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!handle_) throw RuntimeFailure(std::string("dlopen: ") + dlerror());

      auto query = symbol<abi::Descriptor const*(*)()>(handle_, "trinity_abi");
      auto d = query();
      if (!d || d->version != abi::version) throw RuntimeFailure("unsupported host ABI; recompile artifact");
      auto expected = abi::descriptor(mode, nullptr, 0);
      if (d->mode != expected.mode || d->pointer_size != expected.pointer_size ||
          d->error_size != expected.error_size || d->error_alignment != expected.error_alignment ||
          d->launch_size != expected.launch_size || d->launch_alignment != expected.launch_alignment ||
          d->field_count != expected.field_count ||
          std::memcmp(d->offsets, expected.offsets, sizeof(d->offsets)) ||
          std::memcmp(d->error_offsets, expected.error_offsets, sizeof(d->error_offsets)) ||
          !d->requirements || d->requirements_size > 16*1024*1024)
        throw RuntimeFailure("host ABI layout mismatch");

      descriptor = *d;
      metadata.assign(d->requirements, d->requirements_size);
      prepare_fn = symbol<Prepare>(handle_, "trinity_prepare");
      release_fn = symbol<Release>(handle_, "trinity_release");
      symbol<void*>(handle_, "trinity_launch"); symbol<void*>(handle_, "trinity_status");
    } catch (...) { close_unprepared(); throw; }
  }

  Library(Library const&) = delete;
  Library& operator=(Library const&) = delete;

  ~Library() { if (!release_required_) close_unprepared(); }

  void* handle() const { return handle_; }

  unsigned prepare() {
    release_required_ = true;
    unsigned maximum = 0; abi::ErrorInfo error{};
    int result = prepare_fn(&maximum, &error);
    check(result, error, "prepare");

    return maximum;
  }

  void close() {
    if (!handle_) return;
    if (release_required_) {
      abi::ErrorInfo error{};
      int result = release_fn(&error);
      check(result, error, "release");
      release_required_ = false;
    }

    close_unprepared();
  }

private:
  void close_unprepared() noexcept {
    if (handle_) dlclose(handle_);
    handle_ = nullptr;
    if (!directory_.empty()) {
      ::unlink((directory_ + "/program.so").c_str());
      ::rmdir(directory_.c_str());
    }
  }
};

struct BufferSpec {
  std::size_t value, bytes, alignment;
  std::vector<std::int64_t> shape, strides;
  std::string dtype;
  bool symmetric;
};

struct TensorView {
  std::uintptr_t address;
  std::size_t available_bytes;
  std::vector<std::int64_t> shape, strides;
  std::string dtype;
  int device;
  bool cuda, symmetric;
};

inline void validate_tensor(BufferSpec const& s, TensorView const& t, int device) {
  auto width = s.dtype == "bf16" ? 2u : s.dtype == "fp32" ? 4u : 0u;
  std::size_t count=1;
  bool valid=width && (s.shape.size()>=1 && s.shape.size()<=3) && s.strides.size()==s.shape.size();
  for (std::size_t axis=s.shape.size(); valid && axis-->0;) {
    auto extent=s.shape[axis];
    valid=extent>0 && s.strides[axis]==static_cast<std::int64_t>(count) && count<=2147483647u/static_cast<std::size_t>(extent);
    if (valid) count*=extent;
  }
  if (!valid || s.bytes!=count*width || s.alignment<width || (s.alignment&(s.alignment-1)) ||
      !t.cuda || t.dtype!=s.dtype || t.device != device || t.shape != s.shape || t.strides != s.strides || !t.address || !s.alignment ||
      t.address % s.alignment || t.available_bytes < s.bytes || (s.symmetric && !t.symmetric))
    throw std::invalid_argument("Tensor does not match binding " + std::to_string(s.value) +
        " (dtype/shape/stride/alignment/device/storage)");
}

inline void validate_aliases(std::vector<BufferSpec> const& specs, std::vector<TensorView> const& tensors) {
  for (std::size_t i=0; i<specs.size(); ++i) for (std::size_t j=0; j<i; ++j) {
    auto a=tensors[i].address, b=tensors[j].address;
    if (a<=b ? b-a<specs[i].bytes : a-b<specs[j].bytes)
      throw std::invalid_argument("different canonical values overlap");
  }
}

// GC enqueues native-only callbacks. The registry is deliberately process-lived;
// interpreter shutdown stops polling without invoking CUDA or a collective.
class RetireQueue {
  struct Entry { std::uint64_t world; std::function<bool()> reclaim; };
  std::mutex mutex_;
  std::vector<Entry> entries_;
  std::atomic<bool> running_{true};

public:
  RetireQueue() { std::thread([this] {
    while (running_.load()) { drain(0); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
  }).detach(); }

  static RetireQueue& instance() { static auto* queue = new RetireQueue; return *queue; }

  void stop() { running_ = false; }

  void retire(std::function<bool()> reclaim, std::uint64_t world=0) {
    std::lock_guard<std::mutex> lock(mutex_); entries_.push_back({world,std::move(reclaim)});
  }

  void drain(std::uint64_t world) {
    std::vector<Entry> work;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it=entries_.begin();
      while (it!=entries_.end()) {
        if (it->world==world) { work.push_back(std::move(*it)); it=entries_.erase(it); } else ++it;
      }
    }

    for (auto& entry:work) {
      bool done=false;
      try { done=entry.reclaim(); } catch (...) { /* retain uncertain resources */ }
      if (!done) retire(std::move(entry.reclaim),entry.world);
    }
  }
};
} // namespace trinity::native
