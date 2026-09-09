#include "runtime.hpp"

#ifdef TRINITY_NVSHMEM
#define NVSHMEM_HOSTLIB_ONLY
#include <nvshmem.h>
#include <host/nvshmemx_api.h>
#endif

namespace trinity::native {
static std::mutex world_creation_mutex;
static bool world_claimed=false;
static std::atomic<std::uint64_t> world_ids{1};

World::World(std::string const& path,int d,int r,int n):device(d),rank(r),size(n),id(world_ids++) {
#ifndef TRINITY_NVSHMEM
  throw RuntimeFailure("native extension was built without NVSHMEM_HOME");
#else
  if (size<2 || rank<0 || rank>=size) throw std::invalid_argument("invalid NVSHMEM rank/world size");

  std::lock_guard<std::mutex> lock(world_creation_mutex);
  if (world_claimed) throw ResourceBusy("one NVSHMEM world per process; reinitialization is unsupported");

  library_=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
  if (!library_) throw RuntimeFailure(std::string("NVSHMEM dlopen: ")+dlerror());

  try {
    int major=0,minor=0,patch=0;
    symbol<decltype(&nvshmemx_vendor_get_version_info)>(library_,"nvshmemx_vendor_get_version_info")(&major,&minor,&patch);
    if (major!=3 || minor!=7 || patch!=2) throw RuntimeFailure("NVSHMEM 3.7.2 is required");
    if (symbol<decltype(&nvshmemx_init_status)>(library_,"nvshmemx_init_status")()!=NVSHMEM_STATUS_NOT_INITIALIZED)
      throw RuntimeFailure("external NVSHMEM world borrowing is unsupported");

    align_=symbol<decltype(align_)>(library_,"nvshmem_align");
    free_=symbol<decltype(free_)>(library_,"nvshmem_free");
    finalize_=symbol<decltype(finalize_)>(library_,"nvshmemx_hostlib_finalize");
    peer_=symbol<decltype(peer_)>(library_,"nvshmem_ptr");
  } catch (...) { dlclose(library_);library_=nullptr;throw; }

  world_claimed=true;
#endif
}

std::shared_ptr<World> World::create(std::string const& p,int d,int r,int n) {
  // No destructor may finalize a world, even at interpreter shutdown.
  return {new World(p,d,r,n),[](World* w) {
    if (w->closed_) delete w;
    else RetireQueue::instance().retire([w] { return false; },w->id);
  }};
}

std::string World::unique_id() {
#ifdef TRINITY_NVSHMEM
  std::lock_guard<std::recursive_mutex> lock(mutex);
  nvshmemx_uniqueid_t uid=NVSHMEMX_UNIQUEID_INITIALIZER;
  int result=symbol<decltype(&nvshmemx_get_uniqueid)>(library_,"nvshmemx_get_uniqueid")(&uid);
  check(result,{abi::transport,abi::initialization,result,0,0,0},"NVSHMEM unique ID",rank);

  return std::string(reinterpret_cast<char*>(&uid),sizeof(uid));
#else
  throw RuntimeFailure("NVSHMEM is unavailable");
#endif
}

void World::initialize(std::string const& data) {
#ifdef TRINITY_NVSHMEM
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (initialized_ || closed_) throw ResourceBusy("world cannot initialize twice");
  if (data.size()!=sizeof(nvshmemx_uniqueid_t)) throw std::invalid_argument("invalid unique ID size");

  c10::cuda::CUDAGuard guard(device);
  nvshmemx_uniqueid_t uid=NVSHMEMX_UNIQUEID_INITIALIZER;
  std::memcpy(&uid,data.data(),sizeof(uid));
  nvshmemx_init_attr_t attributes=NVSHMEMX_INIT_ATTR_INITIALIZER;
  int result=symbol<decltype(&nvshmemx_set_attr_uniqueid_args)>(library_,"nvshmemx_set_attr_uniqueid_args")(rank,size,&uid,&attributes);
  check(result,{abi::transport,abi::initialization,result,0,0,0},"NVSHMEM UID attributes",rank);

  result=symbol<decltype(&nvshmemx_hostlib_init_attr)>(library_,"nvshmemx_hostlib_init_attr")(NVSHMEMX_INIT_WITH_UNIQUEID,&attributes);
  if (result) poison("NVSHMEM bootstrap failed; world is retained");
  check(result,{abi::transport,abi::initialization,result,0,0,0},"NVSHMEM bootstrap",rank);

  // hostlib initialization is lazy: this explicit collective completes device initialization.
  symbol<void(*)()>(library_,"nvshmem_barrier_all")();
  if (symbol<decltype(&nvshmemx_init_status)>(library_,"nvshmemx_init_status")()<NVSHMEM_STATUS_IS_INITIALIZED)
    throw RuntimeFailure("NVSHMEM device initialization did not complete");

  int threading=0;
  symbol<decltype(&nvshmem_query_thread)>(library_,"nvshmem_query_thread")(&threading);
  if (threading<NVSHMEM_THREAD_SERIALIZED) throw RuntimeFailure("NVSHMEM serialized host threading is required");

  initialized_=true;
#else
  throw RuntimeFailure("NVSHMEM is unavailable");
#endif
}

void World::healthy() const {
  if (!initialized_ || closed_ || poisoned_.load()) throw RuntimeFailure("world is unavailable or poisoned",{},rank);
}

void World::check_idle() const {
  std::lock_guard<std::recursive_mutex> lock(mutex); healthy();
  if (active_) throw ResourceBusy("finish the active execution/Graph before preparing allocations");
}

void World::poison(std::string const& reason) {
  std::lock_guard<std::mutex> lock(failure_mutex_);
  if (!poisoned_.exchange(true)) failure_=reason;
}

void World::activate(std::uintptr_t owner) {
  std::lock_guard<std::recursive_mutex> lock(mutex); healthy();
  if (active_ && active_!=owner) throw ResourceBusy("wait on the active execution/Graph before using this world");

  active_=owner;
}

void World::deactivate(std::uintptr_t owner) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (active_==owner) active_=0;
}

at::Tensor World::allocate(std::size_t bytes,std::size_t alignment,std::vector<std::int64_t> shape,std::string dtype) {
  std::lock_guard<std::recursive_mutex> lock(mutex); healthy();
  if (active_) throw ResourceBusy("finish world execution before allocating symmetric memory");
  if (!bytes || !alignment || (alignment&(alignment-1))) throw std::invalid_argument("invalid symmetric allocation");

  auto width=dtype=="bf16" ? 2u : dtype=="fp32" ? 4u : dtype=="uint8" ? 1u : 0u;
  if (!width || alignment<width || shape.empty()) throw std::invalid_argument("invalid allocation dtype/alignment/shape");
  std::size_t count=1;
  for (auto extent:shape) {
    if (extent<=0 || count>bytes/static_cast<std::size_t>(extent)) throw std::invalid_argument("invalid allocation shape");
    count*=extent;
  }
  if (bytes%width || count!=bytes/width) throw std::invalid_argument("allocation byte/shape mismatch");

  c10::cuda::CUDAGuard guard(device);
  auto allocation=std::make_shared<Allocation>();
  allocation->id=next_allocation_++; allocation->bytes=bytes;
  allocation->pointer=align_(alignment,bytes);
  if (!allocation->pointer) { poison("symmetric allocation failed");throw RuntimeFailure("NVSHMEM allocation failed",{},rank); }
  allocations_.emplace(allocation->id,allocation);

  for (int pe=0;pe<size;++pe) if (!peer_(allocation->pointer,pe)) {
    allocation->leased=false;poison("NVLink peer mapping unavailable");throw RuntimeFailure("NVLink peer mapping unavailable",{},rank);
  }

  try {
    auto options=at::TensorOptions().device(at::kCUDA,device).dtype(dtype=="bf16"?at::kBFloat16:dtype=="fp32"?at::kFloat:at::kByte);
    return at::from_blob(allocation->pointer,shape,[allocation](void*) { allocation->leased.store(false); },options);
  } catch (...) { allocation->leased=false;throw; }
}

bool World::owns(at::Tensor const& tensor,std::size_t bytes) const {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (!tensor.defined() || !tensor.is_cuda() || tensor.get_device()!=device) return false;

  auto p=reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  for (auto const& item:allocations_) {
    auto const& a=item.second; auto base=reinterpret_cast<std::uintptr_t>(a->pointer);
    if (a->leased.load() && p>=base && p-base<=a->bytes && bytes<=a->bytes-(p-base)) return true;
  }

  return false;
}

void World::check_multicast(at::Tensor const& tensor) {
  std::lock_guard<std::recursive_mutex> lock(mutex); healthy();
  if (!owns(tensor,tensor.nbytes())) throw std::invalid_argument("multicast buffer must belong to this world");

  c10::cuda::CUDAGuard guard(device);
#ifdef TRINITY_NVSHMEM
  auto pointer=symbol<decltype(&nvshmemx_mc_ptr)>(library_,"nvshmemx_mc_ptr")(NVSHMEM_TEAM_WORLD,tensor.data_ptr());
  if (!pointer) throw std::invalid_argument("NVLS multicast mapping is unavailable for this buffer");
#else
  throw RuntimeFailure("NVSHMEM is unavailable");
#endif
}

void World::record(at::Tensor const& tensor,std::uint64_t stream) {
  std::lock_guard<std::recursive_mutex> lock(mutex); healthy();

  auto p=reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  for (auto const& item:allocations_) {
    auto& a=*item.second; auto base=reinterpret_cast<std::uintptr_t>(a.pointer);
    if (a.leased.load() && p>=base && p-base<a.bytes) { a.streams.insert(stream);return; }
  }

  throw std::invalid_argument("Tensor is not owned by this world");
}

std::vector<std::pair<std::uint64_t,bool>> World::collectable() {
  RetireQueue::instance().drain(0);
  RetireQueue::instance().drain(id);

  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (!initialized_ || closed_) throw RuntimeFailure("world is unavailable",{},rank);
  if (active_) throw ResourceBusy("world still has an active execution/Graph");

  c10::cuda::CUDAGuard guard(device);
  std::vector<std::pair<std::uint64_t,bool>> result;
  for (auto const& item:allocations_) {
    auto& a=*item.second;
    bool ready=!a.leased.load();
    if (ready && a.completion.empty()) for (auto stream:a.streams) {
      auto e=std::make_unique<Event>(device);e->record(cuda_stream(stream,device));a.completion.push_back(std::move(e));
    }
    if (ready) for (auto const& e:a.completion) ready=ready&&e->ready();
    result.emplace_back(a.id,ready);
  }

  return result;
}

void World::collect(std::vector<std::uint64_t> const& ids) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (!initialized_ || closed_) throw RuntimeFailure("world is unavailable",{},rank);

  c10::cuda::CUDAGuard guard(device);
  for (auto allocation_id:ids) {
    auto it=allocations_.find(allocation_id);
    if (it==allocations_.end() || it->second->leased.load()) throw ResourceBusy("symmetric allocation has live storage leases");
    for (auto const& e:it->second->completion) if (!e->ready()) throw ResourceBusy("symmetric allocation has pending consumers");

    free_(it->second->pointer); allocations_.erase(it);
  }
}

void World::close() {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (closed_) return;
  if (!initialized_) throw RuntimeFailure("world initialization was not completed",{},rank);
  if (active_ || modules.load() || !allocations_.empty()) throw ResourceBusy("world has live modules, allocations or submissions");

  c10::cuda::CUDAGuard guard(device);
  finalize_();closed_=true;initialized_=false;
  dlclose(library_);library_=nullptr;
}
} // namespace trinity::native
