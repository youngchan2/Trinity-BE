// Tests native ownership on the available device. This is NOT a Trinity emitted GEMM.
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include "abi.h"
#include "fixture_metadata.h"

using namespace trinity::abi;

static unsigned preparations=0;

static bool fault(char const* mode) {
  auto value=std::getenv("TRINITY_FIXTURE_FAILURE");
  return value && std::strcmp(value,mode)==0;
}

__global__ void copy_kernel(unsigned short const* input,unsigned short* output) {
  // Make pending-GC coverage deterministic without a device-wide synchronization.
  auto start=clock64();
  while(clock64()-start<20000000ULL) __nanosleep(256);
  for(unsigned i=threadIdx.x;i<128*128;i+=blockDim.x) output[i]=input[i];
}

extern "C" Descriptor const* trinity_abi() {
  static auto d=descriptor(1,metadata,sizeof(metadata)-1);return &d;
}

extern "C" int trinity_prepare(unsigned* maximum,ErrorInfo* error) {
  ++preparations;*maximum=1;
  return report(error,driver,fault("prepare")?17:0,prepare);
}

extern "C" int trinity_launch(StreamedLaunch const* p,ErrorInfo* error) {
  if(preparations!=1) return report(error,generated,-77,validate);

  copy_kernel<<<1,128,0,reinterpret_cast<cudaStream_t>(p->stream)>>>(
      static_cast<unsigned short const*>(p->bindings[0]),static_cast<unsigned short*>(p->bindings[1]));
  auto code=cudaGetLastError();
  return report(error,runtime,fault("launch")?719:code,launch,true);
}

extern "C" int trinity_status(void* stream,ErrorInfo* error) {
  return report(error,runtime,cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),completion,true);
}

extern "C" int trinity_release(ErrorInfo* error) {
  return report(error,driver,fault("release")?19:0,release);
}
