#include "core.hpp"
#include <cassert>
#include <iostream>

using namespace trinity::native;

int main(int argc,char** argv) {
  assert(argc==4);
  for(int i=2;i<4;++i) {
    bool rejected=false;
    try { Library bad(argv[i],1); } catch(RuntimeFailure const&) { rejected=true; }
    assert(rejected);
  }

  Library a(argv[1],1), b(argv[1],1);
  assert(a.handle()!=b.handle());
  assert(a.prepare()==7);
  assert(symbol<int(*)()>(a.handle(),"fixture_count")()==1);
  assert(symbol<int(*)()>(b.handle(),"fixture_count")()==0);

  auto fail=symbol<void(*)(int,int)>(a.handle(),"fixture_fail");
  fail(0,73);
  try { a.close(); assert(false); } catch(RuntimeFailure const& error) {
    assert(error.info.code==73 && error.info.domain==trinity::abi::transport);
  }
  assert(symbol<int(*)()>(a.handle(),"fixture_count")()==1); // failed release pins code
  fail(0,0); a.close(); a.close();

  auto fail_b=symbol<void(*)(int,int)>(b.handle(),"fixture_fail");
  fail_b(17,19);
  try { b.prepare();assert(false); } catch(RuntimeFailure const& e) { assert(e.info.code==17); }
  try { b.close();assert(false); } catch(RuntimeFailure const& e) { assert(e.info.code==19); }
  fail_b(0,0);b.close();

  BufferSpec spec{0,128,16,8,8,8,1,false};
  TensorView tensor{1024,128,8,8,8,1,0,true,true,false};
  validate_tensor(spec,tensor,0);

  tensor.address+=2;
  try { validate_tensor(spec,tensor,0);assert(false); } catch(std::invalid_argument const&) {}

  tensor.address=1024;
  auto other=tensor; other.address+=128;
  validate_aliases({spec,spec},{tensor,other});
  other.address-=16;
  try { validate_aliases({spec,spec},{tensor,other});assert(false); } catch(std::invalid_argument const&) {}

  spec.symmetric=true;
  try { validate_tensor(spec,tensor,0);assert(false); } catch(std::invalid_argument const&) {}

  auto& queue=RetireQueue::instance();
  std::atomic<bool> event=false, destroyed=false;
  auto lease=std::shared_ptr<int>(new int(1),[&](int* p){destroyed=true;delete p;});
  std::thread gc([&] { queue.retire([lease,&event]{return event.load();},41); });gc.join();
  lease.reset();queue.drain(0);queue.drain(41);assert(!destroyed);
  event=true;queue.drain(0);assert(!destroyed); // no implicit collective-world cleanup
  queue.drain(41);assert(destroyed);

  bool retried=false;
  queue.retire([&]{ if(!retried){retried=true;throw RuntimeFailure("release failed");}return true;},42);
  queue.drain(42);assert(retried);queue.drain(42);

  // Shutdown must not drain world callbacks, including those that would block forever.
  queue.retire([]{std::abort();return false;},43);
  queue.stop();
  std::cout << "ABI isolation, rollback, Tensor regions, deferred GC and shutdown passed\n";
}
