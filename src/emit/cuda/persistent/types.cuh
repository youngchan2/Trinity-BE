struct Dependency { unsigned rank, slot; };
struct Range { unsigned begin, count; };
struct Task {
  unsigned operation, slot, x, y, z;
  Range dependencies, stages;
};
enum PersistentError : int {
  kInvalidEpoch = -1, kPeerUnavailable = -2, kCollectiveFailed = -3,
  kNvshmemNotInitialized = -8, kReleaseRequired = -9
};
struct Header {
  unsigned lock, head, tail, size, scan, complete, active;
  int error;
  unsigned long long epoch;
};
static_assert(sizeof(Header) <= kWorkspaceHeaderBytes);
static_assert(kWorkspaceAlignment % alignof(Header) == 0);
static_assert(kTokens % alignof(unsigned long long) == 0);
static_assert(kStates % alignof(unsigned) == 0);
static_assert(kQueue % alignof(unsigned) == 0);
struct Context {
  Bindings bindings;
  unsigned char* workspace;
  unsigned rank, workers;
  unsigned long long epoch;
  int delay_rank, delay_task;
  unsigned long long delay_cycles;
};

// Caller owns allocations. Workspace is zeroed once before its first launch.
// Initialization advances the device epoch on actual execution, including Graph
// replay. Delay fields are verification controls, not Python public arguments.
using LaunchParams = trinity::abi::PersistentLaunch;
