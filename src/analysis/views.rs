//! Resolve existing tensor accesses to shape/stride/offset facts without selecting a provider.
use crate::{AccessIndex, Constant, Expression as E, PhysicalPlan, ValueOp};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorView {
    pub value: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

/// Resolve view/axis-only expressions to a bounded strided view of existing
/// contiguous storage. No tensor copies, dtype changes or library rules here.
pub fn tensor_view(plan: &PhysicalPlan, e: &E) -> Option<TensorView> {
    match e {
        E::Load(a) => {
            let shape = a.shape(plan.value_instance(a.value)?.shape());
            let mut strides = vec![1; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            let mut v = TensorView {
                value: a.value.index(),
                shape: shape.to_vec(),
                strides,
                offset: 0,
            };
            for (i, index) in a.indices.iter().enumerate() {
                let (start, width) = match index {
                    AccessIndex::FullTile => (0, shape[i]),
                    AccessIndex::Element(x) => {
                        (usize::try_from(x.evaluate(plan.bindings()).ok()?).ok()?, 1)
                    }
                    AccessIndex::Slice { start, width } => (
                        usize::try_from(start.evaluate(plan.bindings()).ok()?).ok()?,
                        width.resolve(plan.bindings()).ok()?,
                    ),
                    _ => return None,
                };
                if start.checked_add(width)? > shape[i] {
                    return None;
                }
                v.offset += start * v.strides[i];
                v.shape[i] = width;
            }
            Some(v)
        }
        E::Apply { op, args } => {
            let mut v = tensor_view(plan, args.first()?)?;
            match op {
                ValueOp::Squeeze => {
                    let a = integer(args.get(1)?)?;
                    if *v.shape.get(a)? != 1 {
                        return None;
                    }
                    v.shape.remove(a);
                    v.strides.remove(a);
                }
                ValueOp::Transpose => {
                    let n = v.shape.len();
                    if n < 2 {
                        return None;
                    }
                    v.shape.swap(n - 2, n - 1);
                    v.strides.swap(n - 2, n - 1);
                }
                ValueOp::Permute => {
                    let order = args[1..].iter().map(integer).collect::<Option<Vec<_>>>()?;
                    if order.iter().copied().collect::<BTreeSet<_>>()
                        != (0..v.shape.len()).collect()
                    {
                        return None;
                    }
                    v.shape = order.iter().map(|&i| v.shape[i]).collect();
                    v.strides = order.iter().map(|&i| v.strides[i]).collect();
                }
                ValueOp::Broadcast | ValueOp::Unsqueeze => {
                    let a = integer(args.get(1)?)?;
                    if a > v.shape.len() {
                        return None;
                    }
                    v.shape.insert(a, 1);
                    v.strides.insert(a, 0);
                }
                _ => return None,
            }
            Some(v)
        }
        E::Broadcast { value, axis } | E::Unsqueeze { value, axis } => {
            let mut v = tensor_view(plan, value)?;
            if *axis > v.shape.len() {
                return None;
            }
            v.shape.insert(*axis, 1);
            v.strides.insert(*axis, 0);
            Some(v)
        }
        _ => None,
    }
}
pub(crate) fn integer(e: &E) -> Option<usize> {
    match e {
        E::Constant(Constant::Integer(n)) => usize::try_from(*n).ok(),
        _ => None,
    }
}
