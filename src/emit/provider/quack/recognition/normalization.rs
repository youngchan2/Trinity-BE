//! Recognize normalization equations, including epsilon placement and the
//! population-mean divisor. This does not select an implementation or alter IR.
use super::region::{axis_op, number, tensor_view, unary};
use super::*;

#[derive(Debug, Clone)]
pub struct Normalization {
    pub input: E,
    pub weight: Option<E>,
    pub bias: Option<E>,
    pub axis: usize,
    pub epsilon: f64,
    pub centered: bool,
}
fn square(e: &E) -> Option<&E> {
    match e {
        E::Sqr(x) => Some(x),
        E::Mul(a) if a[0] == a[1] => Some(&a[0]),
        _ => None,
    }
}
fn mean(e: &E, axis: usize, n: usize) -> Option<&E> {
    let sum = match e {
        E::Div(a) if number(&a[1]) == Some(n as f64) => &a[0],
        E::Mul(a) => {
            let reciprocal = |x: &E| {
                number(x).is_some_and(|v| v.is_finite() && (v * n as f64 - 1.0).abs() <= 1e-7)
            };
            if reciprocal(&a[0]) {
                &a[1]
            } else if reciprocal(&a[1]) {
                &a[0]
            } else {
                return None;
            }
        }
        _ => return None,
    };
    let (x, a) = axis_op(sum, ValueOp::ReduceSum)?;
    (a == axis).then_some(x)
}
fn inverse_sqrt(e: &E) -> Option<&E> {
    let E::Div(a) = e else {
        return None;
    };
    if number(&a[0]) != Some(1.0) {
        return None;
    }
    let E::Sqrt(x) = &a[1] else {
        return None;
    };
    Some(x)
}
fn core(plan: &PhysicalPlan, e: &E) -> Option<Normalization> {
    let (center, variance, axis) = match e {
        E::Div(a) => {
            let (d, axis) = axis_op(&a[1], ValueOp::Broadcast)?;
            let E::Sqrt(v) = d else {
                return None;
            };
            (&a[0], v.as_ref(), axis)
        }
        E::Mul(a) => [(&a[0], &a[1]), (&a[1], &a[0])]
            .into_iter()
            .find_map(|(x, s)| {
                let (s, axis) = axis_op(s, ValueOp::Broadcast)?;
                Some((x, inverse_sqrt(s)?, axis))
            })?,
        _ => return None,
    };
    let (input, centered) = if let E::Sub(a) = center {
        (&a[0], true)
    } else {
        (center, false)
    };
    let view = tensor_view(plan, input)?;
    let n = *view.shape.get(axis)?;
    let E::Add(a) = variance else {
        return None;
    };
    let (variance, epsilon) = if let Some(eps) = number(&a[1]) {
        (&a[0], eps)
    } else {
        (&a[1], number(&a[0])?)
    };
    if !epsilon.is_finite() || epsilon < 0.0 {
        return None;
    }
    if square(mean(variance, axis, n)?)? != center {
        return None;
    }
    if centered {
        let E::Sub(a) = center else { unreachable!() };
        let (mu, mu_axis) = axis_op(&a[1], ValueOp::Broadcast)?;
        if mu_axis != axis || mean(mu, axis, n)? != input {
            return None;
        }
    }
    Some(Normalization {
        input: input.clone(),
        weight: None,
        bias: None,
        axis,
        epsilon,
        centered,
    })
}
fn affine(plan: &PhysicalPlan, e: &E) -> Option<Normalization> {
    if let Some(n) = core(plan, e) {
        return Some(n);
    }
    if let E::Mul(a) = e {
        for (x, w) in [(&a[0], &a[1]), (&a[1], &a[0])] {
            if let Some(mut n) = core(plan, x) {
                n.weight = Some(w.clone());
                return Some(n);
            }
        }
    }
    None
}
pub(super) fn recognize(plan: &PhysicalPlan, e: &E) -> Option<Normalization> {
    if let Some(n) = affine(plan, e) {
        return Some(n);
    }
    if let E::Add(a) = e {
        for (x, b) in [(&a[0], &a[1]), (&a[1], &a[0])] {
            if let Some(mut n) = affine(plan, x) {
                n.bias = Some(b.clone());
                return Some(n);
            }
        }
    }
    None
}
/// Only the stable equation with an explicit max correction is mapped. An
/// uncorrected exp/sum is not silently assigned a different numerical policy.
pub(super) fn softmax(e: &E) -> Option<(E, usize)> {
    let E::Div(a) = e else {
        return None;
    };
    let exponent = unary(&a[0], ValueOp::Exp)?;
    let (sum, axis) = axis_op(&a[1], ValueOp::Broadcast)?;
    let (summed, sum_axis) = axis_op(sum, ValueOp::ReduceSum)?;
    if sum_axis != axis || summed != &a[0] {
        return None;
    }
    let E::Sub(s) = exponent else {
        return None;
    };
    let (max, max_axis) = axis_op(&s[1], ValueOp::Broadcast)?;
    let (input, reduce_axis) = axis_op(max, ValueOp::ReduceMax)?;
    if max_axis != axis || reduce_axis != axis || input != &s[0] {
        return None;
    }
    Some((input.clone(), axis))
}
