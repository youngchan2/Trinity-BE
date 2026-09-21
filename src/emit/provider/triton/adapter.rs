//! Convert a resolved PhysicalPlan operation into existing source analysis.
//! IDs are explicitly mapped; no loop, operation or store boundary is invented.
use crate::analysis::{IrNode as N, analyze};
use crate::emit::request::KernelRequest;
use crate::triton::{Error, Options, TritonPlan, invalid, lower as lower_triton};
use crate::{Constant, DType, Expression as E, TensorAccess};

pub(super) fn lower(request: &KernelRequest) -> Result<TritonPlan, Error> {
    let mut options = Options::default();
    for (&id, t) in &request.tensors {
        let name = format!("v{}", id.index());
        options.shapes.insert(name.clone(), t.shape.clone());
        options.dtypes.insert(name, t.dtype.into());
    }
    let E::Store { destination, value } = &request.expression else {
        return Err(invalid("expected a store"));
    };
    let (value, shape, _) = expression(value, request)?;
    let output_shape = &request.tensors[&destination.value].shape;
    if broadcast(&shape, output_shape)? != *output_shape {
        return Err(invalid("result shape differs from output"));
    }
    let root = N::call(
        "store",
        [
            view(destination, request),
            value,
            N::call("keyed_index", []),
        ],
    );
    lower_triton(analyze(root)?, options)
}

fn view(access: &TensorAccess, r: &KernelRequest) -> N {
    let t = &r.tensors[&access.value];
    let kind = if access.value == r.output {
        "output"
    } else {
        "input"
    };
    N::call(
        "view",
        [
            N::call(kind, [N::atom(format!("v{}", access.value.index()))]),
            N::call(
                "layout",
                t.shape.iter().enumerate().map(|(i, size)| {
                    N::call(
                        "axis",
                        [N::atom(format!("d{i}")), N::atom(size.to_string())],
                    )
                }),
            ),
        ],
    )
}

// Shape checks use logical dimensions, before power-of-two Triton padding.
fn expression(e: &E, r: &KernelRequest) -> Result<(N, Vec<usize>, DType), Error> {
    Ok(match e {
        E::Load(a) => (
            N::call("load", [view(a, r), N::call("keyed_index", [])]),
            r.tensors[&a.value].shape.clone(),
            r.tensors[&a.value].dtype,
        ),
        E::Constant(c) => {
            let value = match c {
                Constant::Integer(i) => i.to_string(),
                Constant::Float32(bits) => {
                    let n = f32::from_bits(*bits);
                    if !n.is_finite() {
                        return Err(invalid("nonfinite constant"));
                    }
                    format!("{n:?}")
                }
            };
            (N::atom(value), vec![], DType::Fp32)
        }
        E::Add(args) | E::Sub(args) | E::Mul(args) | E::Div(args) | E::Matmul(args) => {
            let (mut a, ashape, adtype) = expression(&args[0], r)?;
            let (mut b, bshape, bdtype) = expression(&args[1], r)?;
            let op = match e {
                E::Add(_) => "+",
                E::Sub(_) => "-",
                E::Mul(_) => "*",
                E::Div(_) => "/",
                _ => "@",
            };
            let shape = if op == "@" {
                if ashape.len() != 2
                    || bshape.len() != 2
                    || ashape[1] != bshape[0]
                    || adtype != bdtype
                {
                    return Err(invalid(
                        "typed matmul requires compatible rank-2 operands with equal dtypes",
                    ));
                }
                let dtype = match adtype {
                    DType::Bf16 => "bf16",
                    DType::Fp32 => "fp32",
                };
                a = N::call("cast", [N::atom(dtype), a]);
                b = N::call("cast", [N::atom(dtype), b]);
                vec![ashape[0], bshape[1]]
            } else {
                broadcast(&ashape, &bshape)?
            };
            (N::call(op, [a, b]), shape, DType::Fp32)
        }
        E::Sqr(value) | E::Sqrt(value) | E::Sigmoid(value) | E::Relu(value) => {
            let (value, shape, _) = expression(value, r)?;
            let node = match e {
                E::Relu(_) => N::call("max", [value, N::atom("0")]),
                _ => N::call(
                    match e {
                        E::Sqr(_) => "sqr",
                        E::Sqrt(_) => "sqrt",
                        _ => "sigmoid",
                    },
                    [value],
                ),
            };
            (node, shape, DType::Fp32)
        }
        E::ReduceSum { value, axis }
        | E::Broadcast { value, axis }
        | E::Unsqueeze { value, axis } => {
            let (value, mut shape, dtype) = expression(value, r)?;
            let reduction = matches!(e, E::ReduceSum { .. });
            if (reduction && *axis >= shape.len()) || (!reduction && *axis > shape.len()) {
                return Err(invalid("axis out of range"));
            }
            let op = if reduction {
                shape.remove(*axis);
                "rsum"
            } else {
                shape.insert(*axis, 1);
                "unsqueeze"
            };
            (
                N::call(op, [value, N::atom(axis.to_string())]),
                shape,
                if reduction { DType::Fp32 } else { dtype },
            )
        }
        _ => {
            return Err(invalid(
                "unsupported computation in independent Triton candidate",
            ));
        }
    })
}

fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, Error> {
    let rank = a.len().max(b.len());
    (0..rank)
        .map(|i| {
            let x = a.get(i.wrapping_sub(rank - a.len())).copied().unwrap_or(1);
            let y = b.get(i.wrapping_sub(rank - b.len())).copied().unwrap_or(1);
            if x == y || x == 1 || y == 1 {
                Ok(x.max(y))
            } else {
                Err(invalid("incompatible logical broadcast shapes"))
            }
        })
        .collect()
}
