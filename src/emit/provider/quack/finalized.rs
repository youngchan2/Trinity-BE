//! Specialize a proven Quack call at emission time. No runtime pattern switch,
//! JSON specification, or unused epilogue branches are shipped in the program.
use super::pattern::{QuackRegionOperation as Op, QuackRegionSpecification};
use crate::analysis::views::TensorView;
use crate::emit::provider::python::{PythonKernel, tuple};
use crate::{PhysicalPlan, ValueInstanceId};
use std::collections::BTreeMap;

const MATRIX: &str = "def _quack_matrix(value):\n    aligned = value.data_ptr() % 16 == 0\n    if aligned and ((value.stride(-1) == 1 and value.stride(-2) % 8 == 0)\n                    or (value.stride(-2) == 1 and value.stride(-1) % 8 == 0)):\n        return value\n    result = value.contiguous()\n    return result if result.data_ptr() % 16 == 0 else result.clone()\n";

fn view(plan: &PhysicalPlan, v: &TensorView) -> String {
    let name = format!("v{}", v.value);
    let base = plan
        .value_instance(ValueInstanceId::from_index(v.value))
        .unwrap();
    let mut stride = 1;
    let contiguous = v
        .shape
        .iter()
        .rev()
        .zip(v.strides.iter().rev())
        .all(|(d, s)| {
            let same = *s == stride;
            stride *= d;
            same
        });
    if v.offset == 0
        && contiguous
        && v.shape.iter().product::<usize>() == base.shape().iter().product::<usize>()
    {
        if v.shape == base.shape() {
            name
        } else {
            format!("{name}.view{}", tuple(&v.shape))
        }
    } else {
        let offset = if v.offset == 0 {
            String::new()
        } else {
            format!(", {name}.storage_offset() + {}", v.offset)
        };
        format!(
            "{name}.as_strided({}, {}{offset})",
            tuple(&v.shape),
            tuple(&v.strides)
        )
    }
}

impl QuackRegionSpecification {
    pub(crate) fn python_kernel(
        &self,
        plan: &PhysicalPlan,
        region: usize,
        arguments: Vec<ValueInstanceId>,
    ) -> PythonKernel {
        let entrypoint = format!("_run_quack_{region}");
        let v = |a: &TensorView| view(plan, a);
        let optional = |a: &Option<TensorView>, suffix: &str| {
            a.as_ref()
                .map(|a| format!("{}{suffix}", v(a)))
                .unwrap_or("None".into())
        };
        let mut body = vec![
            format!("out = {}", v(&self.output)),
            "if not out.is_contiguous() or out.data_ptr() % 16:".into(),
            "    raise ValueError('Quack requires an aligned contiguous output')".into(),
        ];
        let mut helpers = BTreeMap::new();
        if matches!(
            self.operation,
            Op::Gemm { .. } | Op::GatedGemm { .. } | Op::GemmRotary { .. }
        ) {
            helpers.insert("_quack_matrix".into(), MATRIX.into());
        }
        match &self.operation {
            Op::Gemm {
                lhs,
                rhs,
                residual,
                bias,
                activation,
            } => {
                body.extend([
                    format!("a = _quack_matrix({})", v(lhs)),
                    format!("b = _quack_matrix({})", v(rhs)),
                ]);
                let c = residual
                    .as_ref()
                    .map(|r| format!("_quack_matrix({})", v(r)))
                    .unwrap_or("None".into());
                let bias = optional(bias, ".contiguous()");
                if let Some(activation) = activation {
                    body.push("from quack.gemm_interface import gemm_act".into());
                    body.push(format!("gemm_act(a, b, C={c}, bias={bias}, activation={activation:?}, postact_out=out, out_dtype=out.dtype, postact_dtype=out.dtype, store_preact=False, tuned=True)"));
                } else if residual.is_some() {
                    body.push("from quack.gemm_interface import gemm_add".into());
                    body.push(format!(
                        "gemm_add(a, b, {c}, out=out, bias={bias}, tuned=True, split_k=1)"
                    ));
                } else {
                    body.push("from quack.gemm_interface import gemm".into());
                    body.push(format!(
                        "gemm(a, b, out=out, bias={bias}, tuned=True, split_k=1)"
                    ));
                }
            }
            Op::GatedGemm {
                lhs,
                gate,
                up,
                packed,
                activation,
            } => {
                body.push("from quack.gemm_interface import gemm_act".into());
                body.push(format!("a = _quack_matrix({})", v(lhs)));
                body.push(format!(
                    "weight = {}",
                    packed.as_ref().map(&v).unwrap_or_else(|| format!(
                        "torch.stack(({}, {}), dim=-1).flatten(-2)",
                        v(gate),
                        v(up)
                    ))
                ));
                body.push(format!("gemm_act(a, _quack_matrix(weight), activation={activation:?}, postact_out=out, out_dtype=out.dtype, postact_dtype=out.dtype, store_preact=False, tuned=True)"));
            }
            Op::RmsNorm {
                input,
                weight,
                bias,
                epsilon,
            }
            | Op::LayerNorm {
                input,
                weight,
                bias,
                epsilon,
            } => {
                body.push(format!("x = {}.contiguous()", v(input)));
                body.push(format!("w = {}", optional(weight, ".contiguous()")));
                body.push(format!("b = {}", optional(bias, ".contiguous()")));
                if matches!(self.operation, Op::LayerNorm { .. }) {
                    body.push("from quack.rmsnorm import layernorm_fwd".into());
                    body.push("w = w.float() if w is not None else torch.ones(x.shape[-1], device=x.device, dtype=torch.float32)".into());
                    body.push("b = b.float() if b is not None else None".into());
                    body.push(format!(
                        "out.copy_(layernorm_fwd(x, w, b, eps={epsilon:?}))"
                    ));
                } else {
                    body.push("from quack.rmsnorm import rmsnorm_fwd".into());
                    body.push(format!("result, _, _ = rmsnorm_fwd(x, w, b, out_dtype=out.dtype, eps={epsilon:?}, store_rstd=False)"));
                    body.push("out.copy_(result)".into());
                }
            }
            Op::Softmax { input } => {
                body.push("from quack.softmax import softmax_fwd".into());
                body.push(format!("out.copy_(softmax_fwd({}.contiguous()))", v(input)));
            }
            Op::Rotary {
                input,
                cos,
                sin,
                order,
                interleaved,
                conjugate,
            } => {
                body.push("from quack.rotary import apply_rotary".into());
                body.push(format!(
                    "x = {}.permute({}).contiguous()",
                    v(input),
                    tuple(order)
                ));
                for (name, table) in [("c", cos), ("s", sin)] {
                    body.push(format!(
                        "{name} = {}.permute({})[0, :, 0, :].expand(x.shape[1], -1).contiguous()",
                        v(table),
                        tuple(order)
                    ));
                }
                body.push(format!(
                    "result = apply_rotary(x, c, s, interleaved={}, conjugate={}, inplace=False)",
                    if *interleaved { "True" } else { "False" },
                    if *conjugate { "True" } else { "False" }
                ));
                let inverse: Vec<_> = (0..order.len())
                    .map(|a| order.iter().position(|b| *b == a).unwrap())
                    .collect();
                body.push(format!("out.copy_(result.permute({}))", tuple(inverse)));
            }
            Op::GemmRotary {
                lhs,
                rhs,
                cos,
                sin,
                pair_shape,
            } => {
                body.push("from quack.epilogue.library import rope_epi".into());
                body.push(format!("a = _quack_matrix({})", v(lhs)));
                body.push(format!("b = _quack_matrix({})", v(rhs)));
                body.push(format!(
                    "c = {}.float().expand{}",
                    v(cos),
                    tuple(pair_shape)
                ));
                body.push(format!(
                    "s = {}.float().expand{}",
                    v(sin),
                    tuple(pair_shape)
                ));
                body.push("table = torch.stack((c, s), dim=-1).flatten(-2).contiguous()".into());
                body.push("rope_epi(a, b, out={'D': out}, table=table, tuned=True)".into());
            }
        }
        let params = arguments
            .iter()
            .map(|v| format!("v{}", v.index()))
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "def {entrypoint}({params}):\n{}\n",
            body.iter()
                .map(|l| format!("    {l}\n"))
                .collect::<String>()
        );
        PythonKernel {
            entrypoint,
            arguments,
            source,
            imports: Default::default(),
            helpers,
        }
    }
}
