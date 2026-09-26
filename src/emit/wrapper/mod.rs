//! Compose operation/whole-region candidates or preserve a Triton-only program.
use super::{EmitError, kernel_candidates, request::KernelRequest};
use crate::analysis::regions::OperationScope;
use crate::emit::provider::quack::recognition::{
    GemmInitialization, QuackPatternAnalysis, RegionAnalysis,
};
use crate::{
    Constant, CudaTargetCapability, Expression as E, PhysicalPlan, Statement, TargetCapability,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

mod finalized;
pub use finalized::{PythonExecutable, PythonProvider, PythonSelection, emit_python_executable};

#[derive(Debug, Clone)]
pub struct PythonProgram {
    manifest: Value,
    sources: BTreeMap<String, String>,
    fallback: Option<String>,
}
impl PythonProgram {
    pub(super) fn candidates(manifest: Value, sources: BTreeMap<String, String>) -> Self {
        Self {
            manifest,
            sources,
            fallback: None,
        }
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn sources(&self) -> &BTreeMap<String, String> {
        &self.sources
    }
    /// Self-contained Python module. Eligible operation/region candidates use
    /// correctness and timing selection; other programs use direct fallback.
    pub fn emit(&self) -> String {
        if let Some(source) = &self.fallback {
            return format!(
                "{source}\nimport json\n_MANIFEST = json.loads({})\n{}",
                serde_json::to_string(&self.manifest.to_string()).unwrap(),
                include_str!("fallback.py")
            );
        }
        format!(
            "{}\n{}\n{}\n_MANIFEST = json.loads({})\n_SOURCES = json.loads({})\n\ndef prepare(inputs, *, providers=None, rtol=1e-2, atol=1e-2, repeats=10):\n    return _prepare(_MANIFEST, _SOURCES, inputs, providers=providers, rtol=rtol, atol=atol, repeats=repeats)\n",
            include_str!("reference.py"),
            include_str!("selection.py"),
            include_str!("runtime.py"),
            serde_json::to_string(&self.manifest.to_string()).unwrap(),
            serde_json::to_string(&serde_json::to_string(&self.sources).unwrap()).unwrap()
        )
    }
}

/// Compare independent operations, or preserve a scheduled program through the
/// Triton provider when its loops/regions cannot be split into host-call choices.
pub fn emit_python(plan: &PhysicalPlan) -> Result<PythonProgram, EmitError> {
    if plan.world_size() != 1 {
        return Err(EmitError::UnsupportedExecution {
            reason: "Python kernel execution requires a single GPU".into(),
        });
    }
    if plan.outputs().len() != 1
        || !plan.mutable_inputs().is_empty()
        || plan
            .statements()
            .iter()
            .any(|s| !matches!(s, Statement::Operation(_)))
        || plan
            .operations()
            .any(|(_, op)| !supports_reference(op.expression()))
    {
        let patterns = QuackPatternAnalysis::analyze(plan);
        let classifications: Vec<_> = patterns.regions().map(region_json).collect();
        let regions = super::region_candidates(plan)?;
        let availability: Vec<_> = regions.iter().map(|r| json!({
            "scope":r.scope.statement_path,"pattern":r.pattern,
            "providers":r.candidates.iter().map(|c|c.provider()).collect::<Vec<_>>(),
            "rejections":r.rejections.iter().map(|r|json!({"provider":r.provider,"reason":r.reason})).collect::<Vec<_>>()
        })).collect();
        if let Some(selected) = super::region::program(plan, regions, json!(classifications)) {
            return Ok(selected);
        }
        let program = super::TritonKernelProvider
            .lower_program(plan, Default::default())
            .map_err(|e| EmitError::Combination {
                reason: e.to_string(),
            })?;
        let names = &program.plan().metadata().tensor_names;
        let manifest = json!({"version":1,"mode":"triton_program", "inputs":plan.inputs().iter().map(|b|json!({"name":b.tensor(),"value":b.value().index(),"argument":names[b.value().index()]})).collect::<Vec<_>>(), "outputs":plan.outputs().iter().map(|b|json!({"name":b.tensor(),"value":b.value().index()})).collect::<Vec<_>>(), "kernels":program.plan().kernels().len(), "region_patterns": classifications,"region_candidates":availability});
        let source = program.emit();
        return Ok(PythonProgram {
            manifest,
            sources: [("triton_program".into(), source.clone())].into(),
            fallback: Some(source),
        });
    }
    let all = kernel_candidates(plan)?;
    let mut sources = BTreeMap::new();
    let mut operations = Vec::new();
    for statement in plan.statements() {
        let Statement::Operation(id) = statement else {
            unreachable!()
        };
        let request = KernelRequest::new(plan, *id).map_err(|e| EmitError::Combination {
            reason: e.to_string(),
        })?;
        let mut candidates = Vec::new();
        let mut rejected: Vec<_> = all[id]
            .rejections
            .iter()
            .map(|r| json!({"provider":r.provider,"status":"unsupported","reason":r.reason}))
            .collect();
        for (i, candidate) in all[id].candidates.iter().enumerate() {
            if let Some(source) = candidate.emit_python() {
                let key = format!("op{}_{}_{}", id.index(), candidate.provider(), i);
                sources.insert(key.clone(), source);
                candidates.push(json!({"key":key,"provider":candidate.provider()}));
            } else {
                rejected.push(json!({"provider":candidate.provider(),"status":"unsupported","reason":"native CUDA fragments require the CUDA composition/execution path"}));
            }
        }
        if candidates.is_empty() {
            return Err(EmitError::NoProvider {
                operation: id.index(),
                reasons: rejected.iter().map(Value::to_string).collect(),
            });
        }
        let E::Store { destination, value } = &request.expression else {
            unreachable!()
        };
        operations.push(json!({"id":id.index(),"region_pattern":all[id].region_pattern,"scope":all[id].scope.as_ref().map(scope_json),"inputs":request.inputs.iter().map(|i|i.index()).collect::<Vec<_>>(),"output":request.output.index(),"output_view_shape":destination.shape(&request.tensors[&destination.value].shape),"expression":reference(value),"candidates":candidates,"rejections":rejected}));
    }
    let TargetCapability::Cuda(target) = plan.target();
    let capability = match target {
        CudaTargetCapability::Hopper => [9, 0],
        CudaTargetCapability::Sm89 => [8, 9],
        CudaTargetCapability::Sm120 => [12, 0],
    };
    let manifest = json!({
        "version":1, "capability":capability,
        "inputs":plan.inputs().iter().map(|b|json!({"name":b.tensor(),"value":b.value().index()})).collect::<Vec<_>>(),
        "output":plan.output().value().index(),
        "values":plan.value_instances().map(|(id,v)|json!({"id":id.index(),"dtype":v.dtype(),"shape":v.shape()})).collect::<Vec<_>>(),
        "operations":operations,
    });
    Ok(PythonProgram {
        manifest,
        sources,
        fallback: None,
    })
}

fn scope_json(scope: &OperationScope) -> Value {
    json!({"operation":scope.operation.index(), "statement_path":scope.statement_path, "region_path":scope.region_path})
}

fn region_json(region: &RegionAnalysis<'_>) -> Value {
    json!({
        "statement_path": region.scope.statement_path,
        "operations": region.scope.operations.iter().map(|id| id.index()).collect::<Vec<_>>(),
        "kind": region.kind,
        "reason": region.reason,
        "computation_reason": region.computation_reason,
        "gemm": region.gemm.as_ref().map(|g| json!({
            "matmul_operation": g.matmul_operation.index(),
            "accumulation_operation": g.accumulation_operation.map(|id| id.index()),
            "reduction_loop": g.reduction_loop.map(|l| &l.domain),
            "accumulator": g.accumulator.value.index(),
            "initialization": g.initialization.map(|init| match init {
                GemmInitialization::ZeroRecurrence => json!({"kind":"zero_recurrence"}),
                GemmInitialization::ExplicitZero(id) => json!({"kind":"explicit_zero", "operation":id.index()}),
            }),
            "epilogue_operations": g.epilogue_operations.iter().map(|id| id.index()).collect::<Vec<_>>(),
            "output": g.output().value.index(),
        })),
    })
}

// Extended expressions can be emitted by Triton, but must not enter comparison
// until the independent PyTorch reference also implements their semantics.
fn supports_reference(e: &E) -> bool {
    !matches!(e, E::Apply { .. } | E::Index(_)) && e.children().iter().all(supports_reference)
}

fn reference(e: &E) -> Value {
    match e {
        E::Load(a) => json!({"op":"load","value":a.value.index(),"view_shape":a.view_shape}),
        E::Constant(c) => {
            json!({"op":"constant","value":match c { Constant::Integer(i) => json!(i), Constant::Float32(bits) => json!(f32::from_bits(*bits)), Constant::Float64(bits) => json!(f64::from_bits(*bits)) }})
        }
        E::Add(a) | E::Sub(a) | E::Mul(a) | E::Div(a) | E::Matmul(a) => {
            json!({"op":match e { E::Add(_)=>"add",E::Sub(_)=>"sub",E::Mul(_)=>"mul",E::Div(_)=>"div",_=>"matmul"}, "args":[reference(&a[0]),reference(&a[1])]})
        }
        E::Sqr(v) | E::Sqrt(v) | E::Sigmoid(v) | E::Relu(v) => {
            json!({"op":match e {E::Sqr(_)=>"sqr",E::Sqrt(_)=>"sqrt",E::Sigmoid(_)=>"sigmoid",_=>"relu"},"args":[reference(v)]})
        }
        E::ReduceSum { value, axis }
        | E::Broadcast { value, axis }
        | E::Unsqueeze { value, axis } => {
            json!({"op":if matches!(e,E::ReduceSum{..}) {"sum"} else {"unsqueeze"},"axis":axis,"args":[reference(value)]})
        }
        _ => unreachable!("candidate adapters reject effects in computation expressions"),
    }
}
