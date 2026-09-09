#include <torch/extension.h>
#include <pybind11/stl.h>
#include "runtime.hpp"

namespace py=pybind11;
using namespace trinity::native;
using trinity::abi::ErrorInfo;

static PyObject* failure_type;
static PyObject* busy_type;

PYBIND11_MODULE(_native,m) {
  // Python exception objects are used only by the translator under the GIL.
  auto errors=py::module_::import("trinity_lowering.errors");
  failure_type=errors.attr("RuntimeFailure").ptr();busy_type=errors.attr("ResourceBusy").ptr();
  Py_INCREF(failure_type); Py_INCREF(busy_type);
  py::register_exception_translator([](std::exception_ptr p) {
    try { if(p) std::rethrow_exception(p); }
    catch(ResourceBusy const& e) { PyErr_SetString(busy_type,e.what()); }
    catch(RuntimeFailure const& e) {
      auto error=py::reinterpret_borrow<py::object>(failure_type)(e.what(),py::arg("domain")=e.info.domain,py::arg("code")=e.info.code,
          py::arg("stage")=e.info.stage,py::arg("rank")=e.rank,py::arg("submitted")=bool(e.info.submitted),
          py::arg("cleanup_domain")=e.info.cleanup_domain,py::arg("cleanup_code")=e.info.cleanup_code);
      PyErr_SetObject(failure_type,error.ptr());
    }
  });

  py::class_<BufferSpec>(m,"BufferSpec")
    .def(py::init<std::size_t,std::size_t,std::size_t,std::vector<std::int64_t>,std::vector<std::int64_t>,std::string,bool>());

  py::class_<World,std::shared_ptr<World>>(m,"World")
    .def_static("create",&World::create,py::call_guard<py::gil_scoped_release>())
    .def("unique_id",[](World& w){std::string data;{py::gil_scoped_release release;data=w.unique_id();}return py::bytes(data);})
    .def("initialize",[](World& w,py::bytes data){auto bytes=data.cast<std::string>();py::gil_scoped_release release;w.initialize(bytes);})
    .def("allocate",&World::allocate,py::call_guard<py::gil_scoped_release>())
    .def("owns",&World::owns)
    .def("record_stream",&World::record,py::call_guard<py::gil_scoped_release>())
    .def("check_multicast",&World::check_multicast,py::call_guard<py::gil_scoped_release>())
    .def("collectable",&World::collectable,py::call_guard<py::gil_scoped_release>())
    .def("collect",&World::collect,py::call_guard<py::gil_scoped_release>())
    .def("close",&World::close,py::call_guard<py::gil_scoped_release>())
    .def("poison",&World::poison,py::call_guard<py::gil_scoped_release>())
    .def("healthy",&World::healthy)
    .def("check_idle",&World::check_idle,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("poisoned",&World::poisoned)
    .def_property_readonly("module_count",[](World& w){return w.modules.load();});

  py::class_<Module,std::shared_ptr<Module>>(m,"Module")
    .def_static("load",&Module::create,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("requirements_json",[](Module const& m){std::lock_guard<std::recursive_mutex> lock(m.mutex);if(m.closed)throw RuntimeFailure("module closed");return m.library->metadata;})
    .def("prepare",&Module::prepare,py::call_guard<py::gil_scoped_release>())
    .def("close",&Module::close,py::call_guard<py::gil_scoped_release>());

  py::class_<Execution,std::shared_ptr<Execution>>(m,"Execution")
    .def_static("create",[](std::shared_ptr<Module> module,std::vector<BufferSpec> specs,
        std::vector<at::Tensor> tensors,std::optional<at::Tensor> workspace,
        std::size_t output,unsigned workers,std::uint64_t stream) {
      return Execution::create(std::move(module),std::move(specs),std::move(tensors),
          workspace.value_or(at::Tensor()),output,workers,stream);
    },py::call_guard<py::gil_scoped_release>())
    .def("run",&Execution::run,py::call_guard<py::gil_scoped_release>())
    .def("wait",&Execution::wait,py::arg("timeout")=-1,py::call_guard<py::gil_scoped_release>())
    .def("close",&Execution::close,py::arg("wait")=true,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("closed",&Execution::closed)
    .def_property_readonly("output",&Execution::output);

  py::class_<Graph,std::shared_ptr<Graph>>(m,"Graph")
    .def_static("create",&Graph::create,py::call_guard<py::gil_scoped_release>())
    .def("begin",&Graph::begin,py::call_guard<py::gil_scoped_release>())
    .def("end_warmup",&Graph::end_warmup,py::call_guard<py::gil_scoped_release>())
    .def("end_capture",&Graph::end_capture,py::call_guard<py::gil_scoped_release>())
    .def("abort",&Graph::abort,py::call_guard<py::gil_scoped_release>())
    .def("replay",&Graph::replay,py::call_guard<py::gil_scoped_release>())
    .def("wait",&Graph::wait,py::arg("timeout")=-1,py::call_guard<py::gil_scoped_release>())
    .def("close",&Graph::close,py::arg("wait")=true,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("closed",&Graph::closed)
    .def_property_readonly("output",&Graph::result)
    .def_readonly("calls",&Graph::calls);

  m.def("collect_local",[]{RetireQueue::instance().drain(0);},py::call_guard<py::gil_scoped_release>());
  m.def("shutdown",[]{RetireQueue::instance().stop();});

  m.attr("torch_version")=TORCH_VERSION;
  m.attr("cuda_version")=CUDART_VERSION;
#ifdef TRINITY_NVSHMEM
  m.attr("nvshmem_enabled")=true;
#else
  m.attr("nvshmem_enabled")=false;
#endif
}
