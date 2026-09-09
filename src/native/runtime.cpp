#include "runtime.hpp"

namespace trinity::native {
thread_local Graph* active_graph=nullptr;
static std::mutex capture_failure_mutex;
static std::set<int> failed_capture_devices;

static void check_capture_health(int device) {
  std::lock_guard<std::mutex> lock(capture_failure_mutex);
  if (failed_capture_devices.count(device))
    throw RuntimeFailure("CUDA Graph cleanup was incomplete on this device; restart the process");
}

Event::Event(int d):device(d) {
  c10::cuda::CUDAGuard guard(device);
  cuda_check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
}

Event::~Event() {
  if (event) { try { c10::cuda::CUDAGuard guard(device); cudaEventDestroy(event); } catch (...) {} }
}

void Event::record(c10::cuda::CUDAStream stream) {
  cuda_check(cudaEventRecord(event,stream.stream()),abi::completion,true); recorded=true;
}

bool Event::ready() const {
  if (!recorded) return true;

  c10::cuda::CUDAGuard guard(device);
  auto status=cudaEventQuery(event);
  if (status==cudaErrorNotReady) return false;
  cuda_check(status,abi::completion,true); return true;
}

void Event::wait(double timeout) const {
  auto start=std::chrono::steady_clock::now();
  while (!ready()) {
    if (timeout>=0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>=timeout)
      throw RuntimeFailure("GPU completion timeout; resources remain pinned",{abi::generated,abi::completion,-1001,1,0,0});
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

Module::Module(std::string const& path,int d,unsigned m,std::shared_ptr<World> w):device(d),mode(m),world(std::move(w)) {
  check_capture_health(device);

  c10::cuda::CUDAGuard guard(device);
  cuda_check(cudaFree(nullptr));
  int runtime_version=0; cuda_check(cudaRuntimeGetVersion(&runtime_version));
  if (runtime_version/1000!=13) throw RuntimeFailure("CUDA runtime major must be 13");

  driver_check(cuDevicePrimaryCtxRetain(&context,device));

  try {
    check_context();
    library=std::make_unique<Library>(path,mode);
    launch_fn=symbol<void*>(library->handle(),"trinity_launch");
    status_fn=symbol<void*>(library->handle(),"trinity_status");

    // Check dependency identity, not just equal version numbers.
    if (symbol<void*>(library->handle(),"cudaGetDevice") != reinterpret_cast<void*>(&cudaGetDevice))
      throw RuntimeFailure("artifact loaded a different CUDA runtime");
    if (mode==2) {
      if (!world || world->device!=device) throw std::invalid_argument("persistent module requires its device's world");
      world->healthy();
      if (symbol<void*>(library->handle(),"nvshmemx_init_status") != symbol<void*>(world->library(),"nvshmemx_init_status"))
        throw RuntimeFailure("artifact loaded a different NVSHMEM host library");
      ++world->modules;
    } else if (world) throw std::invalid_argument("streamed module does not use an NVSHMEM world");
  } catch (...) { cuDevicePrimaryCtxRelease(device); context=nullptr; throw; }
}

std::shared_ptr<Module> Module::create(std::string const& p,int d,unsigned m,std::shared_ptr<World> w) {
  return {new Module(p,d,m,std::move(w)),[](Module* module) {
    auto world=module->world ? module->world->id : 0;
    RetireQueue::instance().retire([module] { module->close(); delete module; return true; },world);
  }};
}

void Module::check_context() {
  check_capture_health(device);
  if (closed) throw RuntimeFailure("module is closed");

  CUcontext current=nullptr; driver_check(cuCtxGetCurrent(&current));
  if (current!=context) throw RuntimeFailure("module requires its prepared CUDA primary context");
}

unsigned Module::prepare() {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) { wl=std::unique_lock<std::recursive_mutex>(world->mutex); world->healthy(); }
  std::lock_guard<std::recursive_mutex> lock(mutex);

  c10::cuda::CUDAGuard guard(device); check_context();
  if (!prepared) { maximum_workers=library->prepare(); prepared=true; }

  return maximum_workers;
}

void Module::close() {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (closed) return;
  if (executions) throw ResourceBusy("module has live prepared executions/Graphs");

  c10::cuda::CUDAGuard guard(device); check_context();

  library->close(); library.reset();
  if (world) --world->modules;
  closed=true;

  auto status=cuDevicePrimaryCtxRelease(device); context=nullptr; driver_check(status);
}

static TensorView view(at::Tensor const& t,BufferSpec const& spec,Module const& module) {
  if (!t.defined() || (t.dim()!=1 && t.dim()!=2) || !t.is_cuda() || t.layout()!=at::kStrided)
    throw std::invalid_argument("binding must be a strided CUDA vector or matrix");

  auto offset=static_cast<std::size_t>(t.storage_offset())*t.element_size();
  auto bytes=t.storage().nbytes();
  if (offset>bytes) throw std::invalid_argument("invalid Tensor storage offset");

  auto dtype=t.scalar_type()==at::kBFloat16 ? "bf16" : t.scalar_type()==at::kFloat ? "fp32" : "unsupported";
  return {reinterpret_cast<std::uintptr_t>(t.data_ptr()),bytes-offset,t.sizes().vec(),t.strides().vec(),dtype,
    t.get_device(),true,module.world && module.world->owns(t,spec.bytes)};
}

Execution::Execution(std::shared_ptr<Module> m,std::vector<BufferSpec> s,std::vector<at::Tensor> t,
    at::Tensor ws,std::size_t output,unsigned w,std::uint64_t stream)
    :module(std::move(m)),specs(std::move(s)),tensors(std::move(t)),workspace(std::move(ws)),output_index(output),workers(w) {
  std::unique_lock<std::recursive_mutex> wl;
  if (module->world) wl=std::unique_lock<std::recursive_mutex>(module->world->mutex);
  std::lock_guard<std::recursive_mutex> lock(module->mutex);

  c10::cuda::CUDAGuard guard(module->device); module->check_context();
  if (!module->prepared || specs.size()!=tensors.size() || output_index>=tensors.size()) throw std::invalid_argument("invalid prepared bindings");
  if (workers<1 || workers>module->maximum_workers) throw std::invalid_argument("invalid Worker count");

  for (std::size_t i=0;i<specs.size();++i) {
    if (specs[i].value!=i) throw std::invalid_argument("noncanonical binding order");
    auto v=view(tensors[i],specs[i],*module); validate_tensor(specs[i],v,module->device);
    snapshots_.push_back(v); pointers_.push_back(tensors[i].data_ptr());
    storage_leases_.push_back(tensors[i].storage());
  }

  validate_aliases(specs,snapshots_);

  ready_=std::make_unique<Event>(module->device); tail_=std::make_unique<Event>(module->device);
  auto st=cuda_stream(stream,module->device);
  if (module->mode==2) {
    if (!workspace.defined() || !module->world->owns(workspace,workspace.nbytes()))
      throw std::invalid_argument("workspace must belong to the module world");
    module->world->record(workspace,stream);
    cuda_check(cudaMemsetAsync(workspace.data_ptr(),0,workspace.nbytes(),st.stream()));
  }

  ready_->record(st);
  ++module->executions;
}

std::shared_ptr<Execution> Execution::create(std::shared_ptr<Module> m,std::vector<BufferSpec> s,std::vector<at::Tensor> t,at::Tensor ws,std::size_t o,unsigned w,std::uint64_t st) {
  return {new Execution(std::move(m),std::move(s),std::move(t),std::move(ws),o,w,st),[](Execution* execution) {
    RetireQueue::instance().retire([execution] {
      if (!execution->reclaim()) return false;
      delete execution; return true;
    });
  }};
}

void Execution::validate() {
  if (closed_) throw RuntimeFailure("execution is closed");
  if (failure_) std::rethrow_exception(failure_);
  module->check_context();
  if (module->world) module->world->healthy();

  for (std::size_t i=0;i<tensors.size();++i) {
    if (c10::GradMode::is_enabled() && tensors[i].requires_grad()) throw std::invalid_argument("forward execution requires no_grad/inference_mode for gradient Tensors");
    auto v=view(tensors[i],specs[i],*module); validate_tensor(specs[i],v,module->device);
    if (v.address!=snapshots_[i].address || tensors[i].storage().unsafeGetStorageImpl()!=storage_leases_[i].unsafeGetStorageImpl()) throw std::invalid_argument("prepared Tensor storage was replaced");
  }
}

at::Tensor Execution::run(std::uint64_t stream) {
  std::unique_lock<std::recursive_mutex> wl;
  if (module->world) wl=std::unique_lock<std::recursive_mutex>(module->world->mutex);
  std::lock_guard<std::recursive_mutex> lock(module->mutex);

  c10::cuda::CUDAGuard guard(module->device);
  validate();
  if (graph && graph!=active_graph) throw ResourceBusy("execution belongs to a Graph");

  auto st=cuda_stream(stream,module->device);
  cudaStreamCaptureStatus capture;
  cuda_check(cudaStreamIsCapturing(st.stream(),&capture));
  if (capture!=cudaStreamCaptureStatusNone && !active_graph) throw ResourceBusy("use Trinity Graph.capture to retain resources");
  if (has_stream_ && pending_ && stream!=stream_) throw ResourceBusy("wait before changing execution stream");
  if (active_graph) active_graph->note(this,stream);

  auto owner=reinterpret_cast<std::uintptr_t>(active_graph ? static_cast<void*>(active_graph) : this);
  if (module->world) module->world->activate(owner);

  c10::cuda::CUDAStreamGuard stream_guard(st);
  if (capture==cudaStreamCaptureStatusNone) cuda_check(cudaStreamWaitEvent(st.stream(),ready_->event,0));

  abi::ErrorInfo error{};
  int result=0;

  try {
    for (auto const& t:tensors) {
      if (module->world && module->world->owns(t,t.nbytes())) module->world->record(t,stream);
      else c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(),st);
    }

    if (module->mode==1) {
      abi::StreamedLaunch p{pointers_.data(),pointers_.size(),st.stream()};
      result=reinterpret_cast<int(*)(abi::StreamedLaunch const*,abi::ErrorInfo*)>(module->launch_fn)(&p,&error);
    } else {
      abi::PersistentLaunch p{pointers_.data(),pointers_.size(),workspace.data_ptr(),workspace.nbytes(),st.stream(),workers};
      result=reinterpret_cast<int(*)(abi::PersistentLaunch const*,abi::ErrorInfo*)>(module->launch_fn)(&p,&error);
    }

    if (capture==cudaStreamCaptureStatusNone) {
      pending_=true; stream_=stream; has_stream_=true;
      tail_->record(st);
    }
    check(result,error,"launch",module->world?module->world->rank:-1);
  } catch (...) {
    if (!failure_) failure_=std::current_exception();
    if (module->world) module->world->poison("persistent submission failed");

    // The tail must cover partial submission; absence of a trustworthy event pins forever.
    if (capture==cudaStreamCaptureStatusNone) {
      try { tail_->record(st); pending_=true; stream_=stream; has_stream_=true; }
      catch (...) { uncertain_=true; }
    }
    throw;
  }

  return tensors[output_index];
}

void Execution::status() {
  if (!pending_) return;

  abi::ErrorInfo error{}; int result;
  if (module->mode==1)
    result=reinterpret_cast<int(*)(void*,abi::ErrorInfo*)>(module->status_fn)(reinterpret_cast<void*>(stream_),&error);
  else
    result=reinterpret_cast<int(*)(void const*,void*,abi::ErrorInfo*)>(module->status_fn)(workspace.data_ptr(),reinterpret_cast<void*>(stream_),&error);

  check(result,error,"completion",module->world?module->world->rank:-1);
}

void Execution::wait(double timeout) {
  std::unique_lock<std::recursive_mutex> wl;
  if (module->world) wl=std::unique_lock<std::recursive_mutex>(module->world->mutex);
  std::lock_guard<std::recursive_mutex> lock(module->mutex);
  if (closed_) return;
  if (graph && graph!=active_graph) throw ResourceBusy("wait on the owning Graph");

  c10::cuda::CUDAGuard guard(module->device);
  if (uncertain_) throw RuntimeFailure("completion cannot be established; resources are pinned");

  try { ready_->wait(timeout); if (pending_) { tail_->wait(timeout); status(); } }
  catch (...) {
    if (!failure_) failure_=std::current_exception();
    if (module->world) module->world->poison("persistent completion failed");
    throw;
  }

  pending_=false;
  if (module->world && !graph) module->world->deactivate(reinterpret_cast<std::uintptr_t>(this));
  if (failure_) std::rethrow_exception(failure_);
}

at::Tensor Execution::output() const {
  std::lock_guard<std::recursive_mutex> lock(module->mutex);
  if (closed_) throw RuntimeFailure("execution is closed");
  if (failure_) std::rethrow_exception(failure_);

  return tensors[output_index];
}

void Execution::close(bool wait_for_completion) {
  std::unique_lock<std::recursive_mutex> wl;
  if (module->world) wl=std::unique_lock<std::recursive_mutex>(module->world->mutex);
  std::lock_guard<std::recursive_mutex> lock(module->mutex);
  if (closed_) return;
  if (graph) throw ResourceBusy("execution is retained by a Graph");
  if (wait_for_completion) { try { wait(); } catch (...) { if (!reclaim()) throw; throw; } }
  if (!reclaim()) throw ResourceBusy("execution is pending");
}

bool Execution::reclaim() {
  std::unique_lock<std::recursive_mutex> wl;
  if (module->world) wl=std::unique_lock<std::recursive_mutex>(module->world->mutex);
  std::lock_guard<std::recursive_mutex> lock(module->mutex);
  if (closed_) return true;
  if (graph || uncertain_) return false;

  c10::cuda::CUDAGuard guard(module->device);
  if (!ready_->ready() || (pending_ && !tail_->ready())) return false;

  pending_=false;
  if (module->world) module->world->deactivate(reinterpret_cast<std::uintptr_t>(this));

  tensors.clear(); storage_leases_.clear(); workspace=at::Tensor(); tail_.reset(); ready_.reset();
  --module->executions; closed_=true; return true;
}

void Execution::graph_submitted(std::uint64_t stream) { pending_=true;stream_=stream;has_stream_=true; }

void Execution::graph_finished() { pending_=false; }

Graph::Graph(std::vector<std::shared_ptr<Execution>> e,std::vector<at::Tensor> keep)
    :keepalive_(std::move(keep)),executions(std::move(e)) {
  if (executions.empty()) throw std::invalid_argument("Graph requires prepared executions");

  device=executions[0]->module->device;
  c10::cuda::CUDAGuard guard(device);
  for (auto const& e:executions) if (e->module->world) {
    if (world && world!=e->module->world) throw std::invalid_argument("Graph cannot mix worlds");
    world=e->module->world;
  }
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);

  std::set<Module*> modules;
  for (auto const& e:executions) modules.insert(e->module.get());
  std::vector<std::unique_lock<std::recursive_mutex>> locks;
  for (auto* m:modules) locks.emplace_back(m->mutex);

  std::set<Execution*> unique;
  for (auto const& execution:executions) {
    if (!unique.insert(execution.get()).second || execution->module->device!=device || execution->graph || execution->pending())
      throw ResourceBusy("Graph needs distinct idle executions on one device");
    execution->validate();
    if (execution->module->world) {
      if (world && world!=execution->module->world) throw std::invalid_argument("Graph cannot mix worlds");
      world=execution->module->world;
    }
  }

  for (auto const& t:keepalive_) {
    if (!t.is_cuda() || t.get_device()!=device) throw std::invalid_argument("Graph keepalive must be CUDA Tensors on its device");
    if (c10::GradMode::is_enabled() && t.requires_grad()) throw std::invalid_argument("Graph requires no_grad/inference_mode for gradient Tensors");
    storage_leases_.push_back(t.storage());
    addresses_.push_back(reinterpret_cast<std::uintptr_t>(t.data_ptr()));
    shapes_.push_back(t.sizes().vec()); strides_.push_back(t.strides().vec());
    dtypes_.push_back(t.scalar_type());
  }

  graph_=std::make_unique<at::cuda::CUDAGraph>(); tail_=std::make_unique<Event>(device);
  for (auto& execution:executions) execution->graph=this;
}

std::shared_ptr<Graph> Graph::create(std::vector<std::shared_ptr<Execution>> e,std::vector<at::Tensor> k) {
  return {new Graph(std::move(e),std::move(k)),[](Graph* graph) {
    RetireQueue::instance().retire([graph] { if (!graph->reclaim()) return false; delete graph; return true; });
  }};
}

void Graph::begin(std::uint64_t stream,bool capture) {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  check_capture_health(device);
  if (closed_ || active_graph || captured_) throw ResourceBusy("Graph cannot begin");

  stream_=stream;
  if (world) world->activate(reinterpret_cast<std::uintptr_t>(this));

  c10::cuda::CUDAStreamGuard guard(cuda_stream(stream,device));
  if (capture) { capturing_=true; graph_->capture_begin(); }
  active_graph=this; calls.clear();
}

void Graph::note(Execution* e,std::uint64_t stream) {
  if (stream!=stream_) throw std::invalid_argument("Trinity Graph callbacks must use the capture stream");

  auto it=std::find_if(executions.begin(),executions.end(),[e](auto const& value){return value.get()==e;});
  if (it==executions.end()) throw std::invalid_argument("unregistered Graph execution");

  calls.push_back(it-executions.begin());
}

void Graph::end_warmup(double timeout) {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  c10::cuda::CUDAStreamGuard guard(cuda_stream(stream_,device));

  try {
    pending_=true; tail_->record(cuda_stream(stream_,device)); tail_->wait(timeout);
    for (auto& e:executions) e->wait(timeout);

    pending_=false;
  } catch (...) { active_graph=nullptr; throw; }

  active_graph=nullptr;
  if (world) world->deactivate(reinterpret_cast<std::uintptr_t>(this));
}

void Graph::end_capture(at::Tensor result) {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  c10::cuda::CUDAStreamGuard guard(cuda_stream(stream_,device));

  graph_->capture_end(); capturing_=false; active_graph=nullptr;
  output=std::move(result); captured_=true;

  keepalive_.push_back(output); storage_leases_.push_back(output.storage());
  addresses_.push_back(reinterpret_cast<std::uintptr_t>(output.data_ptr()));
  shapes_.push_back(output.sizes().vec()); strides_.push_back(output.strides().vec());
  dtypes_.push_back(output.scalar_type());
  if (world) world->deactivate(reinterpret_cast<std::uintptr_t>(this));
}

void Graph::abort() {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  c10::cuda::CUDAStreamGuard guard(cuda_stream(stream_,device));
  if (capturing_) {
    try { graph_->capture_end(); }
    catch (std::exception const& error) {
      // PyTorch 2.9 cannot restore all allocator/generator states after an
      // invalidated CUDA capture. Pin the CUDAGraph, including its pool callback.
      uncertain_=true; capturing_=false; active_graph=nullptr;
      { std::lock_guard<std::mutex> lock(capture_failure_mutex); failed_capture_devices.insert(device); }
      if (world) world->poison("CUDA capture cleanup failed");
      throw RuntimeFailure(std::string("capture cleanup failed; restart the process: ")+error.what(),
          {abi::generated,abi::completion,-1002,0,0,0});
    }
    capturing_=false;
  }

  // A failed warmup may already have submitted PyTorch and Trinity kernels.
  // Record its full tail before any Graph/keepalive ownership can be returned.
  try { tail_->record(cuda_stream(stream_,device)); pending_=true; }
  catch (...) { uncertain_=true; }

  active_graph=nullptr;
  captured_=false;
  // Keep the world active until the aborted callback's tail is complete.
}

void Graph::replay(std::uint64_t stream) {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (closed_ || !captured_ || uncertain_) throw RuntimeFailure("Graph is not replayable");
  if (failure_) std::rethrow_exception(failure_);
  if (has_stream_ && pending_ && stream!=stream_) throw ResourceBusy("wait before changing Graph stream");

  c10::cuda::CUDAStreamGuard guard(cuda_stream(stream,device));
  for (auto& e:executions) { std::lock_guard<std::recursive_mutex> lock(e->module->mutex);e->validate(); }

  for (std::size_t i=0;i<keepalive_.size();++i) {
    auto& t=keepalive_[i];
    if (c10::GradMode::is_enabled() && t.requires_grad()) throw std::invalid_argument("Graph replay requires no_grad/inference_mode for gradient Tensors");
    if (!t.is_cuda() || t.get_device()!=device || t.scalar_type()!=dtypes_[i] || reinterpret_cast<std::uintptr_t>(t.data_ptr())!=addresses_[i] || t.sizes().vec()!=shapes_[i] || t.strides().vec()!=strides_[i] || t.storage().unsafeGetStorageImpl()!=storage_leases_[i].unsafeGetStorageImpl())
      throw std::invalid_argument("Graph keepalive metadata/storage changed");
  }

  if (world) world->activate(reinterpret_cast<std::uintptr_t>(this));

  try {
    graph_->replay(); pending_=true; has_stream_=true; stream_=stream;
    tail_->record(cuda_stream(stream,device));
    for (auto& e:executions) {
      e->graph_submitted(stream);
      for (auto const& t:e->tensors) {
        if (world && world->owns(t,t.nbytes())) world->record(t,stream);
        else c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(),cuda_stream(stream,device));
      }
    }

    for (auto& t:keepalive_) {
      if (world && world->owns(t,t.nbytes())) world->record(t,stream);
      else c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(),cuda_stream(stream,device));
    }
  } catch (...) {
    if (!failure_) failure_=std::current_exception();
    for (auto& e:executions) e->graph_failed(failure_);

    try {
      tail_->record(cuda_stream(stream,device));
      pending_=true; has_stream_=true; stream_=stream;
    } catch (...) { uncertain_=true; }
    if (world) world->poison("Graph replay failed");
    throw;
  }
}

void Graph::wait(double timeout) {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (uncertain_) throw RuntimeFailure("Graph completion is uncertain; resources remain pinned");
  if (closed_) return;

  c10::cuda::CUDAGuard guard(device);

  try {
    if (pending_) {
      tail_->wait(timeout);
      for (auto& e:executions) { std::lock_guard<std::recursive_mutex> lock(e->module->mutex); e->status(); }
      for (auto& e:executions) e->graph_finished();
      pending_=false;
    }
  } catch (...) {
    if (!failure_) failure_=std::current_exception();
    for (auto& e:executions) e->graph_failed(failure_);
    if (world) world->poison("Graph completion failed");
    throw;
  }
  if (failure_) std::rethrow_exception(failure_);
  if (world) world->deactivate(reinterpret_cast<std::uintptr_t>(this));
}

at::Tensor Graph::result() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (closed_) throw RuntimeFailure("Graph is closed");
  if (failure_) std::rethrow_exception(failure_);
  return output;
}

void Graph::close(bool wait_for_completion) {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (closed_) return;
  if (wait_for_completion) { try { wait(); } catch (...) { if (!reclaim()) throw; throw; } }
  if (!reclaim()) throw ResourceBusy("Graph is pending");
}

bool Graph::reclaim() {
  std::unique_lock<std::recursive_mutex> wl;
  if (world) wl=std::unique_lock<std::recursive_mutex>(world->mutex);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (closed_) return true;
  if (capturing_ || uncertain_) return false;

  c10::cuda::CUDAGuard guard(device);
  if (pending_ && !tail_->ready()) return false;

  graph_->reset(); graph_.reset();
  for (auto& e:executions) { std::lock_guard<std::recursive_mutex> lock(e->module->mutex); e->graph_finished();e->graph=nullptr; }
  if (world) world->deactivate(reinterpret_cast<std::uintptr_t>(this));

  executions.clear(); keepalive_.clear(); storage_leases_.clear(); output=at::Tensor(); tail_.reset(); closed_=true;

  return true;
}
} // namespace trinity::native
