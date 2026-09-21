//! Compose independent candidate calls in original PhysicalPlan statement order.
use super::{EmitError, kernel_candidates, request::KernelRequest};
use crate::{
    Constant, CudaTargetCapability, Expression as E, PhysicalPlan, Statement, TargetCapability,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct PythonProgram {
    manifest: Value,
    sources: BTreeMap<String, String>,
}
impl PythonProgram {
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn sources(&self) -> &BTreeMap<String, String> {
        &self.sources
    }
    /// Self-contained Python module: prepare(inputs) validates and selects;
    /// the returned executable runs only the selected implementations.
    pub fn emit(&self) -> String {
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

/// Generate a streamed Python executable from loop-free full-tensor operations.
/// Looped scheduled IR continues to use triton::compile; no loops are dropped.
pub fn emit_python(plan: &PhysicalPlan) -> Result<PythonProgram, EmitError> {
    if plan.world_size() != 1
        || plan
            .statements()
            .iter()
            .any(|s| matches!(s, Statement::Loop(_)))
    {
        return Err(EmitError::Combination {
            reason:
                "Python candidate execution currently requires a single-GPU, loop-free PhysicalPlan"
                    .into(),
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
        let E::Store { value, .. } = &request.expression else {
            unreachable!()
        };
        operations.push(json!({"id":id.index(),"inputs":request.inputs.iter().map(|i|i.index()).collect::<Vec<_>>(),"output":request.output.index(),"expression":reference(value),"candidates":candidates,"rejections":rejected}));
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
    Ok(PythonProgram { manifest, sources })
}

fn reference(e: &E) -> Value {
    match e {
        E::Load(a) => json!({"op":"load","value":a.value.index()}),
        E::Constant(c) => {
            json!({"op":"constant","value":match c { Constant::Integer(i) => json!(i), Constant::Float32(bits) => json!(f32::from_bits(*bits)) }})
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
