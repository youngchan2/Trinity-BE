//! Python compiler bindings, compiled as part of the lowering crate.

use crate as tl;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;

fn bad(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn target(name: &str) -> PyResult<tl::TargetCapability> {
    match name {
        "hopper" | "sm_90a" => Ok(tl::TargetCapability::Cuda(tl::CudaTargetCapability::Hopper)),
        "sm89" | "sm_89" => Ok(tl::TargetCapability::Cuda(tl::CudaTargetCapability::Sm89)),
        "sm120" | "sm_120" => Ok(tl::TargetCapability::Cuda(tl::CudaTargetCapability::Sm120)),
        _ => Err(bad(
            "supported targets: hopper/sm_90a, sm89/sm_89, sm120/sm_120",
        )),
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
            .map(|instance| Implementation { instance })
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
            .map(|instance| Implementation { instance })
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

#[derive(Clone, Copy)]
enum TensorFamily {
    Pointwise,
    ReduceSum,
    Broadcast,
}

#[pyclass(frozen, module = "trinity_lowering._compiler")]
struct TensorDefinition {
    index: usize,
    target: tl::TargetCapability,
    family: TensorFamily,
}

#[pymethods]
impl TensorDefinition {
    #[getter]
    fn id(&self) -> &str {
        match self.family {
            TensorFamily::Pointwise => tl::pointwise_implementations(self.target)[self.index]
                .id()
                .as_str(),
            TensorFamily::ReduceSum => tl::reduce_sum_implementations(self.target)[self.index]
                .id()
                .as_str(),
            TensorFamily::Broadcast => tl::broadcast_implementations(self.target)[self.index]
                .id()
                .as_str(),
        }
    }

    #[pyo3(signature=(dtypes, shapes, *, scalar=None, axis=None))]
    fn enumerate(
        &self,
        dtypes: Vec<String>,
        shapes: Vec<Vec<usize>>,
        scalar: Option<f32>,
        axis: Option<usize>,
    ) -> PyResult<Vec<Implementation>> {
        let input_count = match self.family {
            TensorFamily::Pointwise => {
                tl::pointwise_implementations(self.target)[self.index].input_count()
            }
            _ => 1,
        };
        if dtypes.len() != input_count + 1
            || shapes.len() != dtypes.len()
            || shapes
                .iter()
                .any(|s| !matches!(s.len(), 1 | 2) || s.contains(&0))
        {
            return Err(bad(
                "expected input/output dtypes and positive vector/matrix shapes",
            ));
        }
        let ds: Vec<_> = dtypes.iter().map(|d| dtype(d)).collect::<PyResult<_>>()?;
        let ss: Vec<_> = shapes.iter().map(Vec::as_slice).collect();
        let instances = match self.family {
            TensorFamily::Pointwise => {
                let definition = tl::pointwise_implementations(self.target)[self.index];
                if axis.is_some()
                    || scalar.is_some() != definition.requires_scalar()
                    || shapes.iter().any(|s| *s != shapes[0])
                {
                    return Err(bad(
                        "pointwise shapes must match; only scalar_div requires scalar, and axis is not accepted",
                    ));
                }
                definition.enumerate(&ds, &ss, scalar)
            }
            TensorFamily::ReduceSum => {
                let axis = axis.unwrap_or(1);
                if scalar.is_some()
                    || shapes[0].len() != 2
                    || axis >= 2
                    || shapes[1] != vec![shapes[0][1 - axis]]
                {
                    return Err(bad(
                        "reduction needs a matrix, a valid axis and its reduced vector shape",
                    ));
                }
                tl::reduce_sum_implementations(self.target)[self.index].enumerate(
                    [ds[0], ds[1]],
                    [ss[0], ss[1]],
                    axis,
                )
            }
            TensorFamily::Broadcast => {
                let axis = axis.unwrap_or(1);
                if scalar.is_some()
                    || shapes[1].len() != 2
                    || axis >= 2
                    || shapes[0] != vec![shapes[1][1 - axis]]
                    || ds[0] != ds[1]
                {
                    return Err(bad(
                        "broadcast needs a vector, matching matrix shape, valid axis and matching dtypes",
                    ));
                }
                tl::broadcast_implementations(self.target)[self.index].enumerate(
                    ds[0],
                    [ss[0], ss[1]],
                    axis,
                )
            }
        };
        Ok(instances
            .into_iter()
            .map(|instance| Implementation { instance })
            .collect())
    }
}

fn tensor_definitions(target_name: &str, family: TensorFamily) -> PyResult<Vec<TensorDefinition>> {
    let target = target(target_name)?;
    let count = match family {
        TensorFamily::Pointwise => tl::pointwise_implementations(target).len(),
        TensorFamily::ReduceSum => tl::reduce_sum_implementations(target).len(),
        TensorFamily::Broadcast => tl::broadcast_implementations(target).len(),
    };
    Ok((0..count)
        .map(|index| TensorDefinition {
            index,
            target,
            family,
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature=(target_name="hopper"))]
fn pointwise_implementations(target_name: &str) -> PyResult<Vec<TensorDefinition>> {
    tensor_definitions(target_name, TensorFamily::Pointwise)
}

#[pyfunction]
#[pyo3(signature=(target_name="hopper"))]
fn reduce_sum_implementations(target_name: &str) -> PyResult<Vec<TensorDefinition>> {
    tensor_definitions(target_name, TensorFamily::ReduceSum)
}

#[pyfunction]
#[pyo3(signature=(target_name="hopper"))]
fn broadcast_implementations(target_name: &str) -> PyResult<Vec<TensorDefinition>> {
    tensor_definitions(target_name, TensorFamily::Broadcast)
}

fn python_index(value: &Bound<'_, PyAny>) -> PyResult<tl::IndexExpr> {
    if let Ok(n) = value.extract::<i64>() {
        return Ok(tl::IndexExpr::Constant(n));
    }
    let text = value.extract::<String>()?;
    tl::plan::parse_index(&text).map_err(bad)
}

#[pyclass(module = "trinity_lowering._compiler")]
struct PhysicalPlanBuilder {
    builder: Option<tl::PhysicalPlanBuilder>,
    values: Vec<tl::ValueInstanceId>,
    statements: Vec<tl::Statement>,
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
            statements: vec![],
        })
    }

    #[pyo3(signature=(dtype_name, shape, storage_name, *, name=None))]
    fn add_value(
        &mut self,
        dtype_name: &str,
        shape: Vec<usize>,
        storage_name: &str,
        name: Option<String>,
    ) -> PyResult<usize> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(bad("positive shape required"));
        }

        let d = dtype(dtype_name)?;
        let s = storage(storage_name)?;
        let name = name.unwrap_or_else(|| format!("v{}", self.values.len()));
        let id = self.open()?.add_named_value(name, d, shape.clone(), s);
        self.values.push(id);

        Ok(self.values.len() - 1)
    }

    fn bind_input(&mut self, name: &str, value: usize) -> PyResult<()> {
        let id = *self.value(value)?;
        self.open()?.bind_input(name, id);

        Ok(())
    }

    /// Registers an explicit operation; implementation selection belongs to Emit.
    #[pyo3(signature=(inflows, outflows, *, expression))]
    fn add_operation(
        &mut self,
        inflows: Vec<usize>,
        outflows: Vec<usize>,
        expression: &str,
    ) -> PyResult<usize> {
        let inflows = inflows
            .iter()
            .map(|&i| self.value(i).copied())
            .collect::<PyResult<Vec<_>>>()?;
        let outflows = outflows
            .iter()
            .map(|&i| self.value(i).copied())
            .collect::<PyResult<Vec<_>>>()?;
        let expression = self.open()?.parse_expression(expression).map_err(bad)?;
        let id = self.open()?.add_operation(inflows, outflows, expression);
        self.statements.push(tl::Statement::Operation(id));
        Ok(self.statements.len() - 1)
    }

    /// Creates a loop node from explicitly supplied bounds and child statement IDs.
    /// Pass root node IDs to build in execution order.
    fn add_loop(
        &mut self,
        kind: &str,
        variable: &str,
        start: &Bound<'_, PyAny>,
        stop: &Bound<'_, PyAny>,
        step: &Bound<'_, PyAny>,
        body: Vec<usize>,
    ) -> PyResult<usize> {
        self.open()?;
        let kind = match kind {
            "parallel" | "ploop" => tl::LoopKind::Parallel,
            "sequential" | "sloop" => tl::LoopKind::Sequential,
            _ => return Err(bad("expected parallel or sequential loop")),
        };
        let domain = tl::LoopDomain {
            variable: variable.into(),
            start: python_index(start)?,
            stop: python_index(stop)?,
            step: python_index(step)?,
        };
        let body = self.statement_nodes(&body)?;
        self.statements
            .push(tl::Statement::Loop(tl::Loop { kind, domain, body }));
        Ok(self.statements.len() - 1)
    }

    /// Builds a plan from root node IDs in execution order.
    fn build(
        &mut self,
        statements: Vec<usize>,
        output_name: &str,
        output: usize,
    ) -> PyResult<PhysicalPlan> {
        let statements = self.statement_nodes(&statements)?;
        let id = *self.value(output)?;
        let b = self
            .builder
            .take()
            .ok_or_else(|| bad("builder is finalized"))?;

        Ok(PhysicalPlan(
            b.build(statements, output_name, id).map_err(bad)?,
        ))
    }
}

impl PhysicalPlanBuilder {
    fn statement_nodes(&self, ids: &[usize]) -> PyResult<Vec<tl::Statement>> {
        ids.iter()
            .map(|&id| {
                self.statements
                    .get(id)
                    .cloned()
                    .ok_or_else(|| bad("unknown statement"))
            })
            .collect()
    }
    fn open(&mut self) -> PyResult<&mut tl::PhysicalPlanBuilder> {
        self.builder
            .as_mut()
            .ok_or_else(|| bad("builder is finalized"))
    }

    fn value(&self, i: usize) -> PyResult<&tl::ValueInstanceId> {
        self.values.get(i).ok_or_else(|| bad("unknown value"))
    }
}

#[pyclass(frozen, module = "trinity_lowering._compiler")]
struct PhysicalPlan(tl::PhysicalPlan);

#[pyfunction]
#[pyo3(signature=(text, symbols, dtypes, target_name="hopper", world_size=1))]
fn lower_ir(
    text: &str,
    symbols: std::collections::BTreeMap<String, i64>,
    dtypes: std::collections::BTreeMap<String, String>,
    target_name: &str,
    world_size: usize,
) -> PyResult<Vec<PhysicalPlan>> {
    let config = tl::IrConfig {
        target: target(target_name)?,
        world_size,
        symbols,
        dtypes: dtypes
            .into_iter()
            .map(|(n, d)| Ok((n, dtype(&d)?)))
            .collect::<PyResult<_>>()?,
    };
    tl::lower_ir(text, &config)
        .map(|plans| plans.into_iter().map(PhysicalPlan).collect())
        .map_err(bad)
}

fn statement_metadata(s: &tl::Statement) -> serde_json::Value {
    let mut data = serde_json::json!({"operations": s.operations().iter().map(|id| id.index()).collect::<Vec<_>>()});
    if let tl::Statement::Region(body) = s {
        data["region"] = serde_json::json!(body.iter().map(statement_metadata).collect::<Vec<_>>());
    }
    if let tl::Statement::Loop(l) = s {
        data["loop"] = serde_json::json!({"kind": match l.kind { tl::LoopKind::Parallel => "parallel", tl::LoopKind::Sequential => "sequential", tl::LoopKind::Split=>"split" },
            "domain": l.domain, "body": l.body.iter().map(statement_metadata).collect::<Vec<_>>()});
    }
    data
}

#[pymethods]
impl PhysicalPlan {
    fn bind_symbols(&self, bindings: std::collections::BTreeMap<String, i64>) -> PyResult<Self> {
        self.0.bind_symbols(&bindings).map(Self).map_err(bad)
    }

    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.0.symbols().into_iter().collect()
    }

    #[getter]
    fn world_size(&self) -> usize {
        self.0.world_size()
    }

    fn metadata_json(&self) -> String {
        serde_json::json!({
            "world_size":self.0.world_size(),
            "inputs":self.0.inputs().iter().map(|b|serde_json::json!({"name":b.tensor(),"value":b.value().index()})).collect::<Vec<_>>(),
            "output":self.0.outputs().first().map(|b|serde_json::json!({"name":b.tensor(),"value":b.value().index()})),
            "outputs":self.0.outputs().iter().map(|b|serde_json::json!({"name":b.tensor(),"value":b.value().index()})).collect::<Vec<_>>(),
            "mutable_inputs":self.0.mutable_inputs().iter().map(|v|v.index()).collect::<Vec<_>>(),
            "values":self.0.value_instances().map(|(id,v)|serde_json::json!({"value":id.index(),"dtype":v.dtype(),"shape":v.shape(),"storage":format!("{:?}",v.storage()).to_lowercase()})).collect::<Vec<_>>(),
            "operations":self.0.operations().map(|(id,o)|serde_json::json!({"id":id.index(),"inflows":o.inflows().iter().map(|v|v.index()).collect::<Vec<_>>(),"outflows":o.outflows().iter().map(|v|v.index()).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "statements":self.0.statements().iter().map(statement_metadata).collect::<Vec<_>>()
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
    py.allow_threads(move || {
        tl::emit(&p).map(CudaSource).map_err(|e| match e {
            tl::EmitError::Unavailable
            | tl::EmitError::UnsupportedExecution { .. }
            | tl::EmitError::NoProvider { .. } => PyNotImplementedError::new_err(e.to_string()),
            tl::EmitError::InvalidExecution { .. } | tl::EmitError::Combination { .. } => {
                PyValueError::new_err(e.to_string())
            }
            tl::EmitError::Provider { .. } | tl::EmitError::Render { .. } => {
                PyRuntimeError::new_err(e.to_string())
            }
        })
    })
}

#[pyfunction]
fn emit_python(py: Python<'_>, plan: PyRef<'_, PhysicalPlan>) -> PyResult<String> {
    let p = plan.0.clone();
    py.allow_threads(move || tl::emit::emit_python(&p).map(|p| p.emit()).map_err(bad))
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
    m.add_class::<TensorDefinition>()?;
    m.add_class::<CudaSource>()?;
    m.add_class::<CudaArtifact>()?;
    m.add_class::<CompileConfig>()?;
    m.add("CompileError", m.py().get_type::<CompileError>())?;

    m.add_function(wrap_pyfunction!(gemm_implementations, m)?)?;
    m.add_function(wrap_pyfunction!(all_gather_implementations, m)?)?;
    m.add_function(wrap_pyfunction!(pointwise_implementations, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_sum_implementations, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_implementations, m)?)?;
    m.add_function(wrap_pyfunction!(emit, m)?)?;
    m.add_function(wrap_pyfunction!(emit_python, m)?)?;
    m.add_function(wrap_pyfunction!(lower_ir, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;

    Ok(())
}
