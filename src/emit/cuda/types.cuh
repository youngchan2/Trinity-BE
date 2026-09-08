// Operation operands and coordinates are independent of execution scheduling.
struct Bindings { void* values[kBuffers == 0 ? 1 : kBuffers]; };
struct Tile { unsigned x, y, z; };
enum Error : int { kInvalidLaunch = -4, kUnsupportedDevice = -5 };
