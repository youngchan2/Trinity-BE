//! Quack-specific whole-region support matching. Computation recognizers live
//! beside this matcher; identity, storage and tensor views come from analysis.
use super::super::ProviderError;
use crate::analysis::regions::RegionFacts;
use crate::analysis::regions::RegionScope;
use crate::analysis::views::{TensorView, tensor_view};
use crate::emit::provider::quack::recognition::{QuackPatternKind, RegionOperation, summarize};
use crate::{
    Constant, CudaTargetCapability, DType, Expression as E, PhysicalPlan, Storage,
    TargetCapability, ValueOp,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;

type Result<T> = std::result::Result<T, ProviderError>;
fn reject(s: &str) -> ProviderError {
    ProviderError::Unsupported(s.into())
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QuackRegionOperation {
    Gemm {
        lhs: TensorView,
        rhs: TensorView,
        residual: Option<TensorView>,
        bias: Option<TensorView>,
        activation: Option<String>,
    },
    GatedGemm {
        lhs: TensorView,
        gate: TensorView,
        up: TensorView,
        packed: Option<TensorView>,
        activation: String,
    },
    RmsNorm {
        input: TensorView,
        weight: Option<TensorView>,
        bias: Option<TensorView>,
        epsilon: f64,
    },
    LayerNorm {
        input: TensorView,
        weight: Option<TensorView>,
        bias: Option<TensorView>,
        epsilon: f64,
    },
    Softmax {
        input: TensorView,
    },
    Rotary {
        input: TensorView,
        cos: TensorView,
        sin: TensorView,
        /// Permute the logical view into Quack's B,T,H,D convention.
        order: Vec<usize>,
        interleaved: bool,
        conjugate: bool,
    },
    GemmRotary {
        lhs: TensorView,
        rhs: TensorView,
        cos: TensorView,
        sin: TensorView,
        pair_shape: Vec<usize>,
    },
}
#[derive(Debug, Clone)]
pub struct QuackRegionSpecification {
    pub scope: RegionScope,
    pub pattern: QuackPatternKind,
    pub operation: QuackRegionOperation,
    pub output: TensorView,
    pub(crate) arguments: Value,
    pub(crate) capability: [u32; 2],
    pub preparation: Vec<String>,
    pub conditions: Vec<String>,
    pub semantic: Option<crate::emit::provider::quack::recognition::rope::RopePattern>,
}
impl QuackRegionSpecification {
    pub fn emit_python(&self) -> String {
        super::render_region::render(self)
    }
    pub fn manifest(&self) -> Value {
        json!({"provider":"quack","region":self.scope.statement_path,"operations":self.scope.operations.iter().map(|i|i.index()).collect::<Vec<_>>(),"pattern":self.pattern,"operation":self.operation,"output":self.output,"arguments":self.arguments,"preparation":self.preparation,"conditions":self.conditions,"semantic":self.semantic,"capability":self.capability})
    }
}
fn view(plan: &PhysicalPlan, e: &E) -> Result<TensorView> {
    tensor_view(plan,e).ok_or_else(||reject("Quack requires a proven affine tensor view, not computed input/implicit dtype conversion"))
}
fn matrix(plan: &PhysicalPlan, e: &E) -> Result<TensorView> {
    let v = view(plan, e)?;
    if v.shape.len() != 2 {
        return Err(reject("Quack GEMM adapter requires matrix operands"));
    }
    Ok(v)
}
fn vector(plan: &PhysicalPlan, e: &E, n: usize) -> Result<TensorView> {
    let mut v = view(plan, e)?;
    while v.shape.len() > 1 && v.shape[0] == 1 {
        v.shape.remove(0);
        v.strides.remove(0);
    }
    if v.shape != [n] {
        return Err(reject(
            "Quack affine weight/bias must be a broadcast last-axis vector",
        ));
    }
    Ok(v)
}
fn dtype(plan: &PhysicalPlan, v: &TensorView) -> DType {
    plan.value_instances().nth(v.value).unwrap().1.dtype()
}
fn gemm_types(plan: &PhysicalPlan, a: &TensorView, b: &TensorView) -> Result<()> {
    if !matches!(dtype(plan, a), DType::Bf16 | DType::Fp16) || dtype(plan, a) != dtype(plan, b) {
        return Err(reject(
            "Quack GEMM requires matching FP16/BF16 operands in this adapter",
        ));
    }
    if a.shape[1] != b.shape[0] {
        return Err(reject("Quack GEMM contraction extents differ"));
    }
    if !a.shape[1].is_multiple_of(8) || !b.shape[1].is_multiple_of(8) {
        return Err(reject("Quack GEMM requires K/N multiples of 8"));
    }
    Ok(())
}
fn activation(e: &E) -> (&E, Option<String>) {
    match e {
        E::Relu(x) => (x, Some("relu".into())),
        E::Apply {
            op: ValueOp::Maximum,
            args,
        } if args.len() == 2 => {
            if matches!(args[1], E::Constant(Constant::Integer(0))) {
                (&args[0], Some("relu".into()))
            } else {
                (e, None)
            }
        }
        E::Mul(a) => {
            for (x, s) in [(&a[0], &a[1]), (&a[1], &a[0])] {
                if matches!(s,E::Sigmoid(g) if g.as_ref()==x) {
                    return (x, Some("silu".into()));
                }
            }
            (e, None)
        }
        _ => (e, None),
    }
}
/// `packed` is a view proof, not a promise that unrelated allocations are
/// contiguous. Otherwise render an explicit, per-call interleave preparation.
fn packed(g: &TensorView, u: &TensorView) -> Option<TensorView> {
    if g.value == u.value
        && g.shape == u.shape
        && g.strides == u.strides
        && g.strides[1] == 2
        && u.offset == g.offset + 1
        && g.offset == 0
        && g.strides[0] == 2 * g.shape[1]
    {
        Some(TensorView {
            value: g.value,
            shape: vec![g.shape[0], 2 * g.shape[1]],
            strides: vec![2 * g.shape[1], 1],
            offset: 0,
        })
    } else {
        None
    }
}
pub fn match_region(
    plan: &PhysicalPlan,
    region: &RegionFacts<'_>,
) -> Result<QuackRegionSpecification> {
    if plan.world_size() != 1 {
        return Err(reject("Quack requires a single GPU"));
    }
    let TargetCapability::Cuda(target) = plan.target();
    let capability = match target {
        CudaTargetCapability::Hopper => [9, 0],
        CudaTargetCapability::Sm120 => [12, 0],
        _ => return Err(reject("Quack target is unsupported by this adapter")),
    };
    // Provider-owned dispatch: optional semantic helpers, never a QuackPatternKind
    // gate. A new provider can use this same raw region/facts without enums.
    let rotary = crate::emit::provider::quack::recognition::rope::recognize(region);
    if let Ok(r) = rotary {
        return super::rope::specify(plan, region, r, capability);
    }
    let rope_reason = rotary.unwrap_err();
    let c = summarize(plan, region.statement, &region.scope)
        .map_err(|e| reject(&format!("{e}; RoPE: {rope_reason}")))?;
    region
        .require_single_output(c.output.value)
        .map_err(|e| reject(&e))?;
    let output = view(plan, &E::Load(c.output.clone()))?;
    if output.offset != 0
        || output.shape.iter().product::<usize>()
            != plan
                .value_instance(c.output.value)
                .unwrap()
                .shape()
                .iter()
                .product::<usize>()
    {
        return Err(reject("Quack region must write the complete output buffer"));
    }
    if !matches!(
        plan.value_instance(c.output.value).unwrap().storage(),
        Storage::Global | Storage::External
    ) {
        return Err(reject("Quack region output must have global storage"));
    }
    let inputs = c.inputs();
    for &id in &inputs {
        if id == c.output.value
            || plan.mutable_inputs().contains(&id)
            || !matches!(
                plan.value_instance(id).unwrap().storage(),
                Storage::Global | Storage::External
            )
        {
            return Err(reject(
                "Quack region has local/mutated/aliased boundary input",
            ));
        }
    }
    let operation = match &c.operation {
        RegionOperation::Gemm => {
            let (e, activation) = activation(&c.expression);
            let mut addend = None;
            let mm = match e {
                E::Add(a) => {
                    if matches!(a[0], E::Matmul(_)) {
                        addend = Some(&a[1]);
                        &a[0]
                    } else {
                        addend = Some(&a[0]);
                        &a[1]
                    }
                }
                _ => e,
            };
            let E::Matmul(a) = mm else {
                return Err(reject(
                    "Quack GEMM epilogue is not identity, add, ReLU or SiLU",
                ));
            };
            let lhs = matrix(plan, &a[0])?;
            let rhs = matrix(plan, &a[1])?;
            gemm_types(plan, &lhs, &rhs)?;
            if output.shape != [lhs.shape[0], rhs.shape[1]] {
                return Err(reject("Quack GEMM output shape mismatch"));
            }
            let (mut residual, mut bias) = (None, None);
            if let Some(e) = addend {
                let a = view(plan, e)?;
                if a.shape == output.shape {
                    residual = Some(a)
                } else {
                    bias = Some(vector(plan, e, output.shape[1])?)
                }
            }
            QuackRegionOperation::Gemm {
                lhs,
                rhs,
                residual,
                bias,
                activation,
            }
        }
        RegionOperation::GatedGemm { gate, up } => {
            let (E::Matmul(g), E::Matmul(u)) = (gate, up) else {
                unreachable!()
            };
            let lhs = matrix(plan, &g[0])?;
            if lhs != matrix(plan, &u[0])? {
                return Err(reject("gated GEMM inputs do not have the same view"));
            }
            let gate = matrix(plan, &g[1])?;
            let up = matrix(plan, &u[1])?;
            gemm_types(plan, &lhs, &gate)?;
            gemm_types(plan, &lhs, &up)?;
            if gate.shape != up.shape || output.shape != [lhs.shape[0], gate.shape[1]] {
                return Err(reject("gated GEMM shapes differ"));
            }
            let packed = packed(&gate, &up);
            QuackRegionOperation::GatedGemm {
                lhs,
                gate,
                up,
                packed,
                activation: "swiglu".into(),
            }
        }
        RegionOperation::Normalization(n) => {
            let input = matrix(plan, &n.input)?;
            if n.axis != 1 || input.shape != output.shape {
                return Err(reject(
                    "Quack norm adapter requires complete last-axis matrix normalization",
                ));
            }
            if dtype(plan, &input) != dtype(plan, &output) {
                return Err(reject("Quack norm input/output dtype must agree"));
            }
            let weight = n
                .weight
                .as_ref()
                .map(|w| vector(plan, w, input.shape[1]))
                .transpose()?;
            let bias = n
                .bias
                .as_ref()
                .map(|w| vector(plan, w, input.shape[1]))
                .transpose()?;
            if n.centered {
                QuackRegionOperation::LayerNorm {
                    input,
                    weight,
                    bias,
                    epsilon: n.epsilon,
                }
            } else {
                QuackRegionOperation::RmsNorm {
                    input,
                    weight,
                    bias,
                    epsilon: n.epsilon,
                }
            }
        }
        RegionOperation::Softmax { input, axis } => {
            let input = matrix(plan, input)?;
            if *axis != 1
                || input.shape != output.shape
                || dtype(plan, &input) != dtype(plan, &output)
            {
                return Err(reject(
                    "Quack softmax requires equal input/output matrix dtype and last-axis reduction",
                ));
            }
            QuackRegionOperation::Softmax { input }
        }
        RegionOperation::Other => {
            return Err(reject(&format!(
                "no Quack API matches the complete region; RoPE: {rope_reason}"
            )));
        }
    };
    let ids: BTreeSet<_> = inputs.into_iter().chain([c.output.value]).collect();
    let arguments = json!(
        ids.into_iter()
            .map(|id| {
                let v = plan.value_instance(id).unwrap();
                json!({"id":id.index(),"shape":v.shape(),"dtype":v.dtype()})
            })
            .collect::<Vec<_>>()
    );
    Ok(QuackRegionSpecification {
        scope: region.scope.clone(),
        pattern: c.kind(),
        operation,
        output,
        arguments,
        capability,
        preparation: vec!["per-call alignment/layout copies, weight packing and output copy if required; included in run()".into()],
        conditions: vec!["whole-region coverage; complete single output; no input mutation".into()],
        semantic: None,
    })
}
