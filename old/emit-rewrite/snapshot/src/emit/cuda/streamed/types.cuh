// Host binding array is copied by value into each streamed kernel argument.
// The stream orders body launches and owns completion; no control allocation is needed.
using LaunchParams = trinity::abi::StreamedLaunch;
