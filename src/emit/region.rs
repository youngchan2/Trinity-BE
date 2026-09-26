//! Whole-region candidates coexist with the native operation/phase interface.
//! A region's coverage is never inferred from membership of one operation.
use super::{CandidateRejection, EmitError, QuackRegionSpecification, TritonKernelProvider};
use crate::analysis::regions::{RegionFacts, RegionScope};
use crate::analysis::views::tensor_view;
use crate::emit::provider::quack::recognition::{QuackPatternAnalysis, QuackPatternKind};
use crate::{
    Constant, CudaTargetCapability, Expression as E, PhysicalPlan, Storage, TargetCapability,
    ValueOp,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct RegionKernelCandidate {
    provider: String,
    source: String,
    quack: Option<QuackRegionSpecification>,
}
impl RegionKernelCandidate {
    pub fn provider(&self) -> &str {
        &self.provider
    }
    pub fn emit_python(&self) -> &str {
        &self.source
    }
    pub fn quack(&self) -> Option<&QuackRegionSpecification> {
        self.quack.as_ref()
    }
}
#[derive(Debug, Clone)]
pub struct RegionCandidates {
    pub scope: RegionScope,
    /// Quack-owned diagnostic only; never gates another provider's candidates.
    pub pattern: QuackPatternKind,
    pub candidates: Vec<RegionKernelCandidate>,
    pub rejections: Vec<CandidateRejection>,
    pub(crate) globals: Vec<usize>,
    pub(crate) reference: Option<Value>,
    pub(crate) output: Option<usize>,
    pub(crate) output_shape: Option<Vec<usize>>,
}
/// Enumerate fixed-shape Triton and Quack implementations for complete regions.
/// This generates source but does not import Quack, compile GPU code or benchmark.
pub fn region_candidates(plan: &PhysicalPlan) -> Result<Vec<RegionCandidates>, EmitError> {
    Ok(discover_regions(plan)?.0)
}

pub(super) fn discover_regions(
    plan: &PhysicalPlan,
) -> Result<
    (
        Vec<RegionCandidates>,
        Result<super::TritonProgram, super::provider::triton::Error>,
    ),
    EmitError,
> {
    let patterns = QuackPatternAnalysis::analyze(plan);
    let triton = TritonKernelProvider.lower_program(plan, Default::default());
    let mut result = Vec::new();
    for (index, region) in RegionFacts::collect(plan).iter().enumerate() {
        let diagnostic = patterns.region(&region.scope.statement_path);
        let mut entry = RegionCandidates {
            scope: region.scope.clone(),
            pattern: diagnostic
                .map(|r| r.kind)
                .unwrap_or(QuackPatternKind::Other),
            candidates: vec![],
            rejections: vec![],
            globals: region.global_values(),
            reference: None,
            output: None,
            output_shape: None,
        };
        if let Some(c) = diagnostic.and_then(|r| r.computation.as_ref()) {
            entry.reference = reference(plan, &c.expression);
            if let Some(v) = tensor_view(plan, &E::Load(c.output.clone())) {
                entry.output = Some(v.value);
                entry.output_shape = Some(v.shape);
            }
        }
        // Reference the original normalized stores, not the provider's API
        // description. Register intermediates remain FP32 in this oracle.
        if let Ok(r) = crate::emit::provider::quack::recognition::rope::recognize(region) {
            entry.reference = store_reference(plan, &r.stores, &r.output.shape);
            entry.output = Some(r.output.value);
            entry.output_shape = Some(r.output.shape);
        }
        match &triton {
            Ok(p) => match p.plan().emit_region(index) {
                Ok(source) => entry.candidates.push(RegionKernelCandidate {
                    provider: "triton".into(),
                    source,
                    quack: None,
                }),
                Err(reason) => entry.rejections.push(CandidateRejection {
                    provider: "triton".into(),
                    reason,
                }),
            },
            Err(e) => entry.rejections.push(CandidateRejection {
                provider: "triton".into(),
                reason: e.to_string(),
            }),
        }
        match super::provider::quack::pattern::match_region(plan, region) {
            Ok(q) => entry.candidates.push(RegionKernelCandidate {
                provider: "quack".into(),
                source: q.emit_python(),
                quack: Some(q),
            }),
            Err(e) => entry.rejections.push(CandidateRejection {
                provider: "quack".into(),
                reason: e.to_string(),
            }),
        }
        result.push(entry);
    }
    Ok((result, triton))
}

pub(super) fn program(
    plan: &PhysicalPlan,
    regions: Vec<RegionCandidates>,
    patterns: Value,
) -> Option<super::wrapper::PythonProgram> {
    if plan.outputs().len() != 1
        || !plan.mutable_inputs().is_empty()
        || !regions
            .iter()
            .any(|r| r.candidates.iter().any(|c| c.provider == "quack"))
        || regions.iter().any(|r| {
            r.candidates.is_empty()
                || (r.reference.is_none() && !r.candidates.iter().any(|c| c.provider == "triton"))
        })
    {
        return None;
    }
    let mut sources = BTreeMap::new();
    let mut operations = Vec::new();
    for (id, r) in regions.into_iter().enumerate() {
        let mut candidates = Vec::new();
        for c in r.candidates {
            if c.provider == "quack" && r.reference.is_none() {
                continue;
            }
            let key = format!("region_{id}_{}", c.provider);
            candidates.push(json!({"key":key,"provider":c.provider,"implementation":c.quack.as_ref().map(|q|q.manifest())}));
            sources.insert(key, c.source);
        }
        let rejected = r
            .rejections
            .iter()
            .map(|r| json!({"provider":r.provider,"status":"unsupported","reason":r.reason}))
            .collect::<Vec<_>>();
        operations.push(json!({"id":id,"scope":r.scope.statement_path,"pattern":r.pattern,"inputs":r.globals.iter().filter(|v|Some(**v)!=r.output).collect::<Vec<_>>(),"boundary":r.globals,"output":r.output,"output_view_shape":r.output_shape,"expression":r.reference,"candidates":candidates,"rejections":rejected}));
    }
    let TargetCapability::Cuda(target) = plan.target();
    let capability = match target {
        CudaTargetCapability::Hopper => [9, 0],
        CudaTargetCapability::Sm120 => [12, 0],
        CudaTargetCapability::Sm89 => [8, 9],
    };
    let manifest = json!({"version":2,"mode":"region_candidates","capability":capability,"kernels":operations.len(),"region_patterns":patterns,
        "inputs":plan.inputs().iter().map(|b|json!({"name":b.tensor(),"value":b.value().index()})).collect::<Vec<_>>(),
        "output":plan.output().value().index(),
        "values":plan.value_instances().filter(|(_,v)|matches!(v.storage(),Storage::Global|Storage::External)).map(|(id,v)|json!({"id":id.index(),"dtype":v.dtype(),"shape":v.shape()})).collect::<Vec<_>>(),"operations":operations});
    Some(super::wrapper::PythonProgram::candidates(manifest, sources))
}

/// Serialize the proven whole-domain equation, with its strided access mapping.
/// No candidate's output is used as the correctness oracle for another candidate.
fn reference(plan: &PhysicalPlan, e: &E) -> Option<Value> {
    if let Some(v) = tensor_view(plan, e) {
        return Some(json!({"op":"view","view":v}));
    }
    let result = match e {
        E::Constant(c) => {
            json!({"op":"constant","value":match c{Constant::Integer(n)=>*n as f64,Constant::Float32(n)=>f32::from_bits(*n) as f64,Constant::Float64(n)=>f64::from_bits(*n)}})
        }
        E::Add(a) | E::Sub(a) | E::Mul(a) | E::Div(a) | E::Matmul(a) => {
            json!({"op":match e{E::Add(_)=>"add",E::Sub(_)=>"sub",E::Mul(_)=>"mul",E::Div(_)=>"div",_=>"matmul"},"args":[reference(plan,&a[0])?,reference(plan,&a[1])?]})
        }
        E::Sqr(x) | E::Sqrt(x) | E::Sigmoid(x) | E::Relu(x) => {
            json!({"op":match e{E::Sqr(_)=>"sqr",E::Sqrt(_)=>"sqrt",E::Sigmoid(_)=>"sigmoid",_=>"relu"},"args":[reference(plan,x)?]})
        }
        E::ReduceSum { value, axis }
        | E::Broadcast { value, axis }
        | E::Unsqueeze { value, axis } => {
            json!({"op":if matches!(e,E::ReduceSum{..}){"sum"}else{"unsqueeze"},"axis":axis,"args":[reference(plan,value)?]})
        }
        E::Apply { op, args } => {
            if *op == ValueOp::Concat && args.len() == 3 {
                let E::Constant(Constant::Integer(axis)) = args[2] else {
                    return None;
                };
                return Some(
                    json!({"op":"concat","axis":axis,"args":[reference(plan,&args[0])?,reference(plan,&args[1])?]}),
                );
            }
            let name = match op {
                ValueOp::ReduceSum => "sum",
                ValueOp::ReduceMax => "max_reduce",
                ValueOp::ReduceMin => "min_reduce",
                ValueOp::Broadcast | ValueOp::Unsqueeze => "unsqueeze",
                ValueOp::Squeeze => "squeeze",
                ValueOp::Exp => "exp",
                ValueOp::Erf => "erf",
                ValueOp::Abs => "abs",
                ValueOp::Cast(_) => "cast",
                ValueOp::Maximum => "maximum",
                ValueOp::Minimum => "minimum",
                _ => return None,
            };
            let axis = match args.get(1) {
                Some(E::Constant(Constant::Integer(n))) => Some(*n),
                _ => None,
            };
            let unary = matches!(
                op,
                ValueOp::Exp | ValueOp::Erf | ValueOp::Abs | ValueOp::Cast(_)
            );
            let binary = matches!(op, ValueOp::Maximum | ValueOp::Minimum);
            if !unary && !binary && axis.is_none() {
                return None;
            }
            let children = if binary {
                args.iter()
                    .map(|e| reference(plan, e))
                    .collect::<Option<Vec<_>>>()?
            } else {
                vec![reference(plan, args.first()?)?]
            };
            json!({"op":name,"axis":axis,"dtype":match op{ValueOp::Cast(s)=>Some(s),_=>None},"args":children})
        }
        _ => return None,
    };
    Some(result)
}

fn store_reference(
    plan: &PhysicalPlan,
    stores: &[crate::emit::provider::quack::recognition::DomainStore],
    shape: &[usize],
) -> Option<Value> {
    let steps = stores
        .iter()
        .map(|s| {
            let v = plan.value_instance(s.destination.value)?;
            Some(
                json!({"output":tensor_view(plan,&E::Load(s.destination.clone()))?,
            "shape":v.shape(),"dtype":v.dtype(),"register":v.storage()==Storage::Register,
            "expression":reference(plan,&s.expression)?}),
            )
        })
        .collect::<Option<Vec<_>>>()?;
    Some(json!({"op":"stores","steps":steps,"shape":shape}))
}
