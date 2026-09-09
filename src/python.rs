//! Python compiler bindings, compiled as part of the lowering crate.

use crate as tl;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;

fn bad(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn target(name: &str) -> PyResult<tl::TargetCapability> {
    match name {
        "hopper" | "sm_90a" => Ok(tl::TargetCapability::Cuda(tl::CudaTargetCapability::Hopper)),
        _ => Err(bad("only Hopper/sm_90a is supported")),
    }
}

fn dtype(name: &str) -> PyResult<tl::DType> {
    name.parse().map_err(|_| bad("expected bf16 or fp32"))
}

fn storage(name: &str) -> PyResult<tl::Storage> {
    match name {
        "external" => Ok(tl::Storage::External),
        "global" => Ok(tl::Storage::Global),
        "shared" => Ok(tl::Storage::Shared),
        "register" => Ok(tl::Storage::Register),
        _ => Err(bad("unknown storage")),
    }
}

#[pyclass(frozen, module = "trinity_lowering._compiler")]
#[derive(Clone)]
struct Implementation {
    instance: tl::ImplementationInstance,
    shapes: Vec<Vec<usize>>,
    dtypes: Vec<tl::DType>,
    communication: bool,
    world_size: Option<usize>,
}

#[pymethods]
impl Implementation {
    #[getter]
    fn id(&self) -> &str {
        self.instance.id().as_str()
    }

    #[getter]
    fn attributes(&self) -> std::collections::BTreeMap<&str, usize> {
        self.instance.attributes().iter().collect()
    }
}

#[pyclass(frozen, module = "trinity_lowering._compiler")]
struct GemmDefinition {
    index: usize,
    target: tl::TargetCapability,
}

#[pymethods]
impl GemmDefinition {
    #[getter]
    fn id(&self) -> &str {
        tl::gemm_implementations(self.target)[self.index]
            .id()
            .as_str()
    }

    fn enumerate(
        &self,
        dtypes: Vec<String>,
        shapes: Vec<Vec<usize>>,
    ) -> PyResult<Vec<Implementation>> {
        if dtypes.len() != 3
            || shapes.len() != 3
            || shapes.iter().any(|s| s.len() != 2 || s.contains(&0))
        {
            return Err(bad(
                "GEMM needs three dtypes and three positive matrix shapes",
            ));
        }
        if shapes[0][1] != shapes[1][0] || shapes[2] != vec![shapes[0][0], shapes[1][1]] {
            return Err(bad("inconsistent GEMM M/N/K extents"));
        }

        let ds: Vec<_> = dtypes.iter().map(|s| dtype(s)).collect::<PyResult<_>>()?;
        Ok(tl::gemm_implementations(self.target)[self.index]
            .enumerate([ds[0], ds[1], ds[2]], [&shapes[0], &shapes[1], &shapes[2]])
            .into_iter()
            .map(|instance| Implementation {
                instance,
                shapes: shapes.clone(),
                dtypes: ds.clone(),
                communication: false,
                world_size: None,
            })
            .collect())
    }
}

#[pyclass(frozen, module = "trinity_lowering._compiler")]
struct AllGatherDefinition {
    index: usize,
    target: tl::TargetCapability,
}

#[pymethods]
impl AllGatherDefinition {
    #[getter]
    fn id(&self) -> &str {
        tl::all_gather_implementations(self.target)[self.index]
            .id()
            .as_str()
    }

    fn enumerate(
        &self,
        dtype_name: &str,
        shapes: Vec<Vec<usize>>,
        shard_axis: usize,
        world_size: usize,
    ) -> PyResult<Vec<Implementation>> {
        if shapes.len() != 2
            || shapes.iter().any(|s| s.len() != 2 || s.contains(&0))
            || shard_axis > 1
            || world_size < 2
        {
            return Err(bad(
                "AllGather needs two positive matrix shapes, axis 0/1 and world_size > 1",
            ));
        }
        if shapes[0][shard_axis].checked_mul(world_size) != Some(shapes[1][shard_axis])
            || shapes[0][1 - shard_axis] != shapes[1][1 - shard_axis]
        {
            return Err(bad("inconsistent AllGather extents"));
        }

        let d = dtype(dtype_name)?;
        Ok(tl::all_gather_implementations(self.target)[self.index]
            .enumerate(d, [&shapes[0], &shapes[1]], shard_axis, world_size)
            .into_iter()
            .map(|instance| Implementation {
                instance,
                shapes: shapes.clone(),
                dtypes: vec![d; 2],
                communication: true,
                world_size: Some(world_size),
            })
            .collect())
    }
}

#[pyfunction]
#[pyo3(signature=(target_name="hopper"))]
fn gemm_implementations(target_name: &str) -> PyResult<Vec<GemmDefinition>> {
    let t = target(target_name)?;
    Ok((0..tl::gemm_implementations(t).len())
        .map(|index| GemmDefinition { index, target: t })
        .collect())
}

#[pyfunction]
#[pyo3(signature=(target_name="hopper"))]
fn all_gather_implementations(target_name: &str) -> PyResult<Vec<AllGatherDefinition>> {
    let t = target(target_name)?;
    Ok((0..tl::all_gather_implementations(t).len())
        .map(|index| AllGatherDefinition { index, target: t })
        .collect())
}

#[pyclass(module = "trinity_lowering._compiler")]
struct PhysicalPlanBuilder {
    builder: Option<tl::PhysicalPlanBuilder>,
    values: Vec<(tl::ValueInstanceId, tl::DType, Vec<usize>)>,
    operations: Vec<tl::OperationId>,
    world_size: usize,
}

#[pymethods]
impl PhysicalPlanBuilder {
    #[new]
    #[pyo3(signature=(world_size=1,target_name="hopper"))]
    fn new(world_size: usize, target_name: &str) -> PyResult<Self> {
        if world_size == 0 {
            return Err(bad("world_size must be positive"));
        }

        Ok(Self {
            builder: Some(tl::PhysicalPlanBuilder::new(
                target(target_name)?,
                world_size,
            )),
            values: vec![],
            operations: vec![],
            world_size,
        })
    }

    fn add_value(
        &mut self,
        dtype_name: &str,
        shape: Vec<usize>,
        storage_name: &str,
    ) -> PyResult<usize> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(bad("positive shape required"));
        }

        let d = dtype(dtype_name)?;
        let s = storage(storage_name)?;
        let id = self.open()?.add_value(d, shape.clone(), s);
        self.values.push((id, d, shape));

        Ok(self.values.len() - 1)
    }

    fn bind_input(&mut self, name: &str, value: usize) -> PyResult<()> {
        let id = self.value(value)?.0;
        self.open()?.bind_input(name, id);

        Ok(())
    }

    fn add_operation(
        &mut self,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        implementation: PyRef<'_, Implementation>,
    ) -> PyResult<usize> {
        let values = inputs
            .iter()
            .chain(outputs.iter())
            .map(|i| self.value(*i))
            .collect::<PyResult<Vec<_>>>()?;
        if implementation
            .world_size
            .is_some_and(|w| w != self.world_size)
        {
            return Err(bad("implementation world size differs from builder"));
        }
        if outputs.len() != 1
            || inputs.len() != if implementation.communication { 1 } else { 2 }
            || values.iter().map(|v| v.1).collect::<Vec<_>>() != implementation.dtypes
            || values.iter().map(|v| v.2.clone()).collect::<Vec<_>>() != implementation.shapes
        {
            return Err(bad(
                "implementation was enumerated for different operand types/shapes",
            ));
        }

        let ins = inputs.iter().map(|i| self.values[*i].0).collect::<Vec<_>>();
        let outs = outputs
            .iter()
            .map(|i| self.values[*i].0)
            .collect::<Vec<_>>();
        let payload = if implementation.communication {
            tl::OperationPayload::Communication(tl::CommunicationOperation::new(
                tl::CommunicationKind::AllGather,
                implementation.instance.clone(),
            ))
        } else {
            tl::OperationPayload::Compute(tl::ComputeOperation::new(
                implementation.instance.clone(),
            ))
        };

        let id = self.open()?.add_operation(ins, outs, payload);
        self.operations.push(id);

        Ok(self.operations.len() - 1)
    }

    fn add_action(&mut self, operations: Vec<usize>) -> PyResult<()> {
        let ids = operations
            .iter()
            .map(|i| {
                self.operations
                    .get(*i)
                    .copied()
                    .ok_or_else(|| bad("unknown operation"))
            })
            .collect::<PyResult<Vec<_>>>()?;

        self.open()?.add_action(ids);
        Ok(())
    }

    fn finalize(&mut self, output_name: &str, output: usize) -> PyResult<PhysicalPlan> {
        let id = self.value(output)?.0;
        let b = self
            .builder
            .take()
            .ok_or_else(|| bad("builder is finalized"))?;

        Ok(PhysicalPlan(b.finalize(output_name, id).map_err(bad)?))
    }
}

impl PhysicalPlanBuilder {
    fn open(&mut self) -> PyResult<&mut tl::PhysicalPlanBuilder> {
        self.builder
            .as_mut()
            .ok_or_else(|| bad("builder is finalized"))
    }

    fn value(&self, i: usize) -> PyResult<&(tl::ValueInstanceId, tl::DType, Vec<usize>)> {
        self.values.get(i).ok_or_else(|| bad("unknown value"))
    }
}

#[pyclass(frozen, module = "trinity_lowering._compiler")]
struct PhysicalPlan(tl::PhysicalPlan);

#[pymethods]
impl PhysicalPlan {
    #[getter]
    fn world_size(&self) -> usize {
        self.0.world_size()
    }

    fn metadata_json(&self) -> String {
        serde_json::json!({
            "world_size":self.0.world_size(),
            "inputs":self.0.inputs().iter().map(|b|serde_json::json!({"name":b.tensor(),"value":b.value().index()})).collect::<Vec<_>>(),
            "output":{"name":self.0.output().tensor(),"value":self.0.output().value().index()},
            "values":self.0.value_instances().map(|(id,v)|serde_json::json!({"value":id.index(),"dtype":v.dtype(),"shape":v.shape(),"storage":format!("{:?}",v.storage()).to_lowercase()})).collect::<Vec<_>>(),
            "operations":self.0.operations().map(|(id,o)|serde_json::json!({"id":id.index(),"inputs":o.inputs().iter().map(|v|v.index()).collect::<Vec<_>>(),"outputs":o.outputs().iter().map(|v|v.index()).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "actions":self.0.actions().map(|(id,a)|serde_json::json!({"id":id.index(),"operations":a.operations().iter().map(|o|o.index()).collect::<Vec<_>>()})).collect::<Vec<_>>()
        }).to_string()
    }
}

#[pyclass(frozen, module = "trinity_lowering._compiler")]
struct CudaSource(tl::CudaSource);

#[pymethods]
impl CudaSource {
    #[getter]
    fn code(&self) -> &str {
        self.0.code()
    }

    fn requirements_json(&self) -> String {
        serde_json::to_string(self.0.requirements()).unwrap()
    }
}

#[pyfunction]
fn emit(py: Python<'_>, plan: PyRef<'_, PhysicalPlan>) -> PyResult<CudaSource> {
    let p = plan.0.clone();
    py.allow_threads(move || tl::emit(&p).map(CudaSource).map_err(bad))
}

#[pyclass(frozen, module = "trinity_lowering._compiler")]
#[derive(Clone)]
struct CompileConfig(tl::CompileConfig);

#[pymethods]
impl CompileConfig {
    #[new]
    #[pyo3(signature=(*,cuda_root=None,cutlass_root=None,nvshmem_root=None,host_compiler=None))]
    fn new(
        cuda_root: Option<PathBuf>,
        cutlass_root: Option<PathBuf>,
        nvshmem_root: Option<PathBuf>,
        host_compiler: Option<PathBuf>,
    ) -> Self {
        Self(tl::CompileConfig {
            cuda_root,
            cutlass_root,
            nvshmem_root,
            host_compiler,
        })
    }
}

#[pyclass(module = "trinity_lowering._compiler")]
struct CudaArtifact(Option<tl::CudaArtifact>);

impl CudaArtifact {
    fn open(&self) -> PyResult<&tl::CudaArtifact> {
        self.0
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("artifact is closed or persisted"))
    }
}

#[pymethods]
impl CudaArtifact {
    #[getter]
    fn directory(&self) -> PyResult<PathBuf> {
        Ok(self.open()?.directory().into())
    }

    #[getter]
    fn artifact_path(&self) -> PyResult<PathBuf> {
        Ok(self.open()?.artifact_path().into())
    }

    fn manifest_json(&self) -> PyResult<String> {
        Ok(self.open()?.manifest().into())
    }

    #[getter]
    fn code(&self) -> PyResult<String> {
        Ok(self.open()?.source().code().into())
    }

    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics(py, self.open()?.diagnostics())
    }

    fn close(&mut self) {
        self.0.take();
    }

    fn persist(&mut self) -> PyResult<PathBuf> {
        Ok(self
            .0
            .take()
            .ok_or_else(|| bad("artifact is closed"))?
            .persist())
    }
}

fn diagnostics<'py>(py: Python<'py>, d: &tl::CompileDiagnostics) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("compiler", d.compiler.to_string_lossy())?;
    out.set_item(
        "arguments",
        d.arguments
            .iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
    )?;
    out.set_item("source", &d.generated_source)?;
    out.set_item("exit_code", d.status.and_then(|s| s.code()))?;
    out.set_item("stdout", PyBytes::new(py, &d.stdout))?;
    out.set_item("stderr", PyBytes::new(py, &d.stderr))?;

    Ok(out)
}

pyo3::create_exception!(_compiler, CompileError, PyRuntimeError);

#[pyfunction]
#[pyo3(signature=(source,config=None))]
fn compile(
    py: Python<'_>,
    source: PyRef<'_, CudaSource>,
    config: Option<PyRef<'_, CompileConfig>>,
) -> PyResult<CudaArtifact> {
    let s = source.0.clone();
    let c = config.map(|c| c.0.clone()).unwrap_or_default();

    match py.allow_threads(move || tl::compile_with_config(s, &c)) {
        Ok(a) => Ok(CudaArtifact(Some(a))),
        Err(e) => {
            let err = CompileError::new_err(e.to_string());
            if let Some(d) = e.diagnostics() {
                err.value(py).setattr("diagnostics", diagnostics(py, d)?)?;
            }
            Err(err)
        }
    }
}

#[pymodule]
fn _compiler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PhysicalPlanBuilder>()?;
    m.add_class::<PhysicalPlan>()?;
    m.add_class::<Implementation>()?;
    m.add_class::<GemmDefinition>()?;
    m.add_class::<AllGatherDefinition>()?;
    m.add_class::<CudaSource>()?;
    m.add_class::<CudaArtifact>()?;
    m.add_class::<CompileConfig>()?;
    m.add("CompileError", m.py().get_type::<CompileError>())?;

    m.add_function(wrap_pyfunction!(gemm_implementations, m)?)?;
    m.add_function(wrap_pyfunction!(all_gather_implementations, m)?)?;
    m.add_function(wrap_pyfunction!(emit, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;

    Ok(())
}
