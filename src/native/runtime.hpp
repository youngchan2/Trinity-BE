#pragma once
#include "core.hpp"
#include <ATen/ATen.h>
#include <c10/core/GradMode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <map>
#include <set>

namespace trinity::native {
inline void cuda_check(cudaError_t code, abi::Stage stage=abi::validate, bool submitted=false) {
  if (code != cudaSuccess) throw RuntimeFailure(cudaGetErrorString(code),
      {abi::runtime,stage,static_cast<int>(code),unsigned(submitted),0,0});
}

inline void driver_check(CUresult code) {
  if (code != CUDA_SUCCESS) throw RuntimeFailure("CUDA driver failure",
      {abi::driver,abi::validate,static_cast<int>(code),0,0,0});
}

inline c10::cuda::CUDAStream cuda_stream(std::uint64_t stream, int device) {
  return c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream),device);
}

struct Event {
  int device;
  cudaEvent_t event=nullptr;
  bool recorded=false;

  explicit Event(int device);
  ~Event();

  void record(c10::cuda::CUDAStream stream);
  bool ready() const;
  void wait(double timeout=-1) const;
};

struct Allocation {
  std::uint64_t id;
  void* pointer;
  std::size_t bytes;
  std::atomic<bool> leased{true};
  std::set<std::uint64_t> streams;
  std::vector<std::unique_ptr<Event>> completion;
};

class World : public std::enable_shared_from_this<World> {
  void* library_=nullptr;
  bool initialized_=false, closed_=false;
  std::atomic<bool> poisoned_{false};
  std::string failure_;
  std::mutex failure_mutex_;
  std::map<std::uint64_t,std::shared_ptr<Allocation>> allocations_;
  std::uint64_t next_allocation_=0;
  std::uintptr_t active_=0;
  void* (*align_)(std::size_t,std::size_t)=nullptr;
  void (*free_)(void*)=nullptr;
  void (*finalize_)()=nullptr;
  void* (*peer_)(void*,int)=nullptr;

public:
  int device, rank, size;
  std::uint64_t id;
  mutable std::recursive_mutex mutex;
  std::atomic<unsigned> modules{0};

  World(std::string const& library, int device, int rank, int size);
  static std::shared_ptr<World> create(std::string const&,int,int,int);
  std::string unique_id();
  void initialize(std::string const& uid);

  void healthy() const;
  bool poisoned() const { return poisoned_.load(); }
  void check_idle() const;
  void poison(std::string const& reason);
  void activate(std::uintptr_t owner);
  void deactivate(std::uintptr_t owner);

  at::Tensor allocate(std::size_t bytes,std::size_t alignment,std::vector<std::int64_t> shape,bool bf16);
  bool owns(at::Tensor const& tensor,std::size_t bytes) const;
  void record(at::Tensor const& tensor,std::uint64_t stream);
  void check_multicast(at::Tensor const& tensor);

  std::vector<std::pair<std::uint64_t,bool>> collectable();
  void collect(std::vector<std::uint64_t> const& ids);
  void close();
  void* library() const { return library_; }
};

class Module : public std::enable_shared_from_this<Module> {
public:
  int device;
  unsigned mode;
  CUcontext context=nullptr;
  std::unique_ptr<Library> library;
  std::shared_ptr<World> world;
  mutable std::recursive_mutex mutex;
  unsigned executions=0;
  bool prepared=false, closed=false;
  unsigned maximum_workers=0;
  void* launch_fn=nullptr;
  void* status_fn=nullptr;

  Module(std::string const&,int,unsigned,std::shared_ptr<World>);
  static std::shared_ptr<Module> create(std::string const&,int,unsigned,std::shared_ptr<World>);
  void check_context();
  unsigned prepare();
  void close();
};

class Graph;
extern thread_local Graph* active_graph;

class Execution : public std::enable_shared_from_this<Execution> {
  std::vector<TensorView> snapshots_;
  std::vector<c10::Storage> storage_leases_;
  std::vector<void*> pointers_;
  bool pending_=false, uncertain_=false, closed_=false;
  bool has_stream_=false;
  std::uint64_t stream_=0;
  std::unique_ptr<Event> ready_,tail_;
  std::exception_ptr failure_;

public:
  std::shared_ptr<Module> module;
  std::vector<BufferSpec> specs;
  std::vector<at::Tensor> tensors;
  at::Tensor workspace;
  std::size_t output_index;
  unsigned workers;
  Graph* graph=nullptr;

  Execution(std::shared_ptr<Module>,std::vector<BufferSpec>,std::vector<at::Tensor>,at::Tensor,std::size_t,unsigned,std::uint64_t);
  static std::shared_ptr<Execution> create(std::shared_ptr<Module>,std::vector<BufferSpec>,std::vector<at::Tensor>,at::Tensor,std::size_t,unsigned,std::uint64_t);

  void validate();
  at::Tensor run(std::uint64_t stream);
  void wait(double timeout=-1);
  void close(bool wait=true);
  bool reclaim();
  void status();

  at::Tensor output() const;
  bool pending() const { return pending_; }
  bool closed() const { std::lock_guard<std::recursive_mutex> lock(module->mutex); return closed_; }

  void graph_submitted(std::uint64_t stream);
  void graph_finished();
  void graph_failed(std::exception_ptr error) { if (!failure_) failure_=error; }
};

class Graph {
  std::unique_ptr<at::cuda::CUDAGraph> graph_;
  std::unique_ptr<Event> tail_;
  mutable std::recursive_mutex mutex_;
  bool pending_=false, uncertain_=false, captured_=false, capturing_=false, closed_=false;
  bool has_stream_=false;
  std::uint64_t stream_=0;
  std::vector<at::Tensor> keepalive_;
  std::vector<c10::Storage> storage_leases_;
  std::exception_ptr failure_;
  std::vector<std::uintptr_t> addresses_;
  std::vector<std::vector<std::int64_t>> shapes_,strides_;
  std::vector<at::ScalarType> dtypes_;

public:
  int device;
  std::vector<std::shared_ptr<Execution>> executions;
  std::shared_ptr<World> world;
  at::Tensor output;
  std::vector<std::size_t> calls;

  Graph(std::vector<std::shared_ptr<Execution>>,std::vector<at::Tensor>);
  static std::shared_ptr<Graph> create(std::vector<std::shared_ptr<Execution>>,std::vector<at::Tensor>);

  void begin(std::uint64_t stream,bool capture);
  void end_warmup(double timeout);
  void end_capture(at::Tensor result);
  void abort();
  void note(Execution*,std::uint64_t stream);

  at::Tensor result() const;
  bool closed() const { std::lock_guard<std::recursive_mutex> lock(mutex_); return closed_; }

  void replay(std::uint64_t stream);
  void wait(double timeout=-1);
  void close(bool wait=true);
  bool reclaim();
};
} // namespace trinity::native
