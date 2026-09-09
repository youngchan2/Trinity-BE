#include "abi.h"

using namespace trinity::abi;

static int preparations=0, release_failure=0, prepare_failure=0;

extern "C" Descriptor const* trinity_abi() {
  static auto d=descriptor(1,"{}",2);
#ifdef BAD_VERSION
  d.version=999;
#endif
#ifdef BAD_OFFSET
  d.offsets[0]=999;
#endif
  return &d;
}

extern "C" int trinity_prepare(unsigned* maximum,ErrorInfo* error) {
  ++preparations; *maximum=7;
  return report(error,transport,prepare_failure,registration);
}

extern "C" int trinity_launch(StreamedLaunch const*,ErrorInfo* error) {
  return report(error,runtime,719,launch,true);
}

extern "C" int trinity_status(void*,ErrorInfo*) { return 0; }

extern "C" int trinity_release(ErrorInfo* error) {
  return report(error,transport,release_failure,release);
}

extern "C" int fixture_count() { return preparations; }

extern "C" void fixture_fail(int prepare,int release) { prepare_failure=prepare;release_failure=release; }
