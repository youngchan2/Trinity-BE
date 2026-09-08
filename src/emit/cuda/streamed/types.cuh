// Host binding array is copied by value into each streamed kernel argument.
// The stream orders Actions and owns completion; no control allocation is needed.
struct LaunchParams {
  void* const* bindings;
  std::size_t binding_count;
  void* stream;
};
