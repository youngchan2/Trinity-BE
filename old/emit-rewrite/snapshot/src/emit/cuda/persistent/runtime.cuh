template <class T>
__device__ T *workspace_at(Context const &c, std::size_t offset) {
  return reinterpret_cast<T *>(c.workspace + offset);
}

__device__ Header *header(Context const &c) {
  return workspace_at<Header>(c, 0);
}

template <class T> __device__ auto device_atomic(T *value) {
  static_assert(
      alignof(T) %
          cuda::atomic_ref<T, cuda::thread_scope_device>::required_alignment ==
      0);
  return cuda::atomic_ref<T, cuda::thread_scope_device>(*value);
}

template <class T> __device__ auto system_atomic(T *value) {
  static_assert(
      alignof(T) %
          cuda::atomic_ref<T, cuda::thread_scope_system>::required_alignment ==
      0);
  return cuda::atomic_ref<T, cuda::thread_scope_system>(*value);
}

__device__ unsigned char *peer_workspace(Context const &c, unsigned rank) {
  if (rank == c.rank)
    return c.workspace;
  return static_cast<unsigned char *>(nvshmem_ptr(c.workspace, rank));
}

__device__ void fail(Context const &c, int code) {
  int expected = 0;
  system_atomic(&header(c)->error)
      .compare_exchange_strong(expected, code, cuda::memory_order_release,
                               cuda::memory_order_relaxed);
}

__device__ bool failed(Context const &c) {
  for (unsigned rank = 0; rank < kWorld; ++rank) {
    auto *base = peer_workspace(c, rank);
    if (base == nullptr) {
      fail(c, kPeerUnavailable);
      return true;
    }
    int code = system_atomic(&reinterpret_cast<Header *>(base)->error)
                   .load(cuda::memory_order_acquire);
    if (code != 0) {
      fail(c, code);
      return true;
    }
  }
  return false;
}

__device__ bool dependencies_ready(Context const &c, Range range) {
  for (unsigned i = 0; i < range.count; ++i) {
    auto dependency = kDependencies[range.begin + i];
    auto *base = peer_workspace(c, dependency.rank);
    if (base == nullptr) {
      fail(c, kPeerUnavailable);
      return false;
    }
    auto *token = reinterpret_cast<unsigned long long *>(base + kTokens) +
                  dependency.slot;
    if (system_atomic(token).load(cuda::memory_order_acquire) != c.epoch)
      return false;
  }
  return true;
}

__device__ bool stage_ready(Context const &c, Task const &task,
                            unsigned stage) {
  return dependencies_ready(c, kStages[task.stages.begin + stage]);
}

__device__ bool await_stage(Context const &c, Task const &task,
                            unsigned stage) {
  __shared__ bool success;
  if (threadIdx.x == 0) {
    while (!stage_ready(c, task, stage) && !failed(c))
      __nanosleep(64);
    success = !failed(c);
  }
  __syncthreads();
  return success;
}

// Hooks passed to a CTA operation. Queue/task state stays in this runtime.
struct PersistentRuntime {
  Context const &context;
  Task const &task;

  __device__ bool await_stage(unsigned stage) const {
    return trinity::generated::await_stage(context, task, stage);
  }
  __device__ bool prefetch_stage(unsigned stage, unsigned stage_count) const {
    __shared__ bool ready;
    if (threadIdx.x == 0)
      ready = stage < stage_count && stage_ready(context, task, stage);
    __syncthreads();
    return ready;
  }
  __device__ unsigned rank() const { return context.rank; }
  __device__ void *peer_ptr(void *pointer, unsigned rank) const {
    return nvshmem_ptr(pointer, rank);
  }
  __device__ void fail(int code) const {
    trinity::generated::fail(context, code);
  }
};

__device__ void before_task(Context const &c, Task const &task) {
  if (c.delay_rank == static_cast<int>(c.rank) &&
      c.delay_task == static_cast<int>(task.slot)) {
    if (threadIdx.x == 0) {
      unsigned long long start = clock64();
      while (clock64() - start < c.delay_cycles)
        __nanosleep(64);
    }
    __syncthreads();
  }
}

__device__ void complete_task(Context const &c, Task const &task,
                              bool reserved) {
  // Every writer, including peer writers, participates before the single
  // publication. Consumers acquire this token before accessing the region.
  __threadfence_system();
  __syncthreads();

  if (threadIdx.x == 0) {
    system_atomic(workspace_at<unsigned long long>(c, kTokens) + task.slot)
        .store(c.epoch, cuda::memory_order_release);

    device_atomic(workspace_at<unsigned>(c, kStates) + task.slot)
        .store(3, cuda::memory_order_relaxed);

    if (reserved)
      device_atomic(&header(c)->active)
          .fetch_sub(1, cuda::memory_order_relaxed);
    device_atomic(&header(c)->complete)
        .fetch_add(1, cuda::memory_order_release);
  }
  __syncthreads();
}

// One bounded ready queue for every worker. A try-lock protects only finite,
// non-blocking queue operations; no data/collective wait occurs under the lock.
// State: 0 pending, 1 queued, 2 running, 3 complete. Remote dependency probes
// rotate through pending tasks, amortized in batches across all idle workers.
__device__ int claim(Context const &c, bool &reserved) {
  auto *h = header(c);

  unsigned expected = 0;
  if (!device_atomic(&h->lock).compare_exchange_strong(
          expected, 1,
          cuda::memory_order_acquire,
          cuda::memory_order_relaxed)) {
    return -1;
  }

  auto *states = workspace_at<unsigned>(c, kStates);
  auto *queue = workspace_at<unsigned>(c, kQueue);

  for (unsigned probe = 0; probe < kTasksPerRank && probe < 32; ++probe) {
    unsigned slot = h->scan;
    h->scan = (slot + 1) % (kTasksPerRank == 0 ? 1 : kTasksPerRank);

    if (device_atomic(states + slot).load(cuda::memory_order_relaxed) != 0)
      continue;

    auto const &task = kTasks[c.rank * kTasksPerRank + slot];
    if (!dependencies_ready(c, task.dependencies)) {
      continue;
    }

    device_atomic(states + slot).store(1, cuda::memory_order_relaxed);

    queue[h->tail] = slot;
    h->tail = (h->tail + 1) % (kTasksPerRank == 0 ? 1 : kTasksPerRank);
    ++h->size;
  }

  int result = -1;
  unsigned candidates = h->size;
  while (candidates-- && result < 0) {
    unsigned slot = queue[h->head];

    h->head = (h->head + 1) % (kTasksPerRank == 0 ? 1 : kTasksPerRank);
    --h->size;

    auto const &task = kTasks[c.rank * kTasksPerRank + slot];
    bool may_wait = false;

    for (unsigned stage = 1; stage < task.stages.count; ++stage) {
      may_wait |= !stage_ready(c, task, stage);
    }

    if (may_wait &&
        device_atomic(&h->active).load(cuda::memory_order_relaxed) >=
            c.workers - 1) {
      queue[h->tail] = slot;
      h->tail = (h->tail + 1) % (kTasksPerRank == 0 ? 1 : kTasksPerRank);
      ++h->size;
      continue;
    }

    reserved = may_wait;
    if (reserved) {
      device_atomic(&h->active).fetch_add(1, cuda::memory_order_relaxed);
    }

    device_atomic(states + slot).store(2, cuda::memory_order_relaxed);
    result = static_cast<int>(slot);
  }

  device_atomic(&h->lock).store(0, cuda::memory_order_release);
  return result;
}

__global__ void initialize(Context c) {
  auto *h = header(c);

  if (threadIdx.x == 0) {
    unsigned long long previous = h->epoch;
    h->lock = h->head = h->tail = h->size = h->scan = h->complete = h->active =
        0;
    // Never erase an earlier failure when calls are queued or a Graph replays.
    if (previous == ~0ULL) fail(c, kInvalidEpoch);
    else h->epoch = previous + 1;
  }

  for (unsigned slot = threadIdx.x; slot < kTasksPerRank; slot += blockDim.x) {
    workspace_at<unsigned>(c, kStates)[slot] = 0;
  }

  __threadfence_system();
  __syncthreads();

  if (threadIdx.x == 0)
    system_atomic(workspace_at<unsigned long long>(c, kTokens) + kTasksPerRank)
        .store(h->epoch, cuda::memory_order_release);
}
