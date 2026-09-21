// Operation operands and coordinates are independent of execution scheduling.
struct Bindings { void* values[kBuffers == 0 ? 1 : kBuffers]; };
enum Error : int {
  kInvalidLaunch = -4, kUnsupportedDevice = -5, kWrongDevice = -6, kNotPrepared = -7
};
