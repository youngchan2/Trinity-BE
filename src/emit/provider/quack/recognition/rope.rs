//! Quack's equation/access-based rotary recognition. GPU/API restrictions are
//! checked separately. Recognition preserves original stores and coverage.
use super::{DomainStore, TensorView, domain_stores, region::integer, tensor_view};
use crate::analysis::regions::RegionFacts;
use crate::{Expression as E, OperationId, PhysicalPlan, ValueOp};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pairing {
    Interleaved,
    SplitHalf,
}

#[derive(Debug, Clone, Serialize)]
pub struct Projection {
    #[serde(serialize_with = "operation_id")]
    pub operation: OperationId,
    pub lhs: TensorView,
    pub rhs: TensorView,
    /// The exact producer store, including its original view, is in `stores`.
    pub result: TensorView,
}

#[derive(Debug, Clone, Serialize)]
pub struct RopePattern {
    pub input: TensorView,
    pub output: TensorView,
    pub first: TensorView,
    pub second: TensorView,
    pub cos: TensorView,
    pub sin: TensorView,
    pub pairing: Pairing,
    /// Axis after collapsing an explicitly represented pair dimension.
    pub rotation_axis: usize,
    pub rotary_dimension: usize,
    pub conjugate: bool,
    /// Logical output axes on which the tables vary. Other prefix axes are
    /// broadcast axes; a provider may map those to batch/head dimensions.
    pub table_axes: Vec<usize>,
    pub passthrough: Option<TensorView>,
    pub projection: Option<Projection>,
    #[serde(serialize_with = "operation_ids")]
    pub operations: Vec<OperationId>,
    #[serde(skip)]
    pub stores: Vec<DomainStore>,
}

fn operation_id<S: serde::Serializer>(id: &OperationId, s: S) -> Result<S::Ok, S::Error> {
    id.index().serialize(s)
}
fn operation_ids<S: serde::Serializer>(ids: &[OperationId], s: S) -> Result<S::Ok, S::Error> {
    ids.iter()
        .map(|i| i.index())
        .collect::<Vec<_>>()
        .serialize(s)
}

fn concat(e: &E) -> Option<(&E, &E, usize)> {
    let E::Apply {
        op: ValueOp::Concat,
        args,
    } = e
    else {
        return None;
    };
    (args.len() == 3).then_some(())?;
    Some((&args[0], &args[1], integer(&args[2])?))
}
fn mul(e: &E) -> Option<[&E; 2]> {
    let E::Mul(a) = e else {
        return None;
    };
    Some([&a[0], &a[1]])
}

/// x0*c - x1*s, x0*s + x1*c (or the conjugate sign arrangement).
/// Multiplication/addition may commute; subtraction and pairing may not.
fn equation<'a>(plan: &PhysicalPlan, a: &'a E, b: &'a E) -> Option<([&'a E; 4], bool)> {
    let (a, b, conjugate) = match (a, b) {
        (E::Sub(a), E::Add(b)) => (a, b, false),
        (E::Add(a), E::Sub(b)) => (a, b, true),
        _ => return None,
    };
    let [p, q] = [mul(&a[0])?, mul(&a[1])?];
    let [r, s] = [mul(&b[0])?, mul(&b[1])?];
    let equal = |a: &E, b: &E| {
        tensor_view(plan, a)
            .zip(tensor_view(plan, b))
            .is_some_and(|(a, b)| a == b)
    };
    for i in 0..2 {
        for j in 0..2 {
            let (x0, c, x1, sin) = (p[i], p[1 - i], q[j], q[1 - j]);
            if !tensor_view(plan, x0)
                .zip(tensor_view(plan, x1))
                .is_some_and(|(a, b)| a.value == b.value && a.offset != b.offset)
            {
                continue;
            }
            let pair = |v: [&E; 2], x: &E, y: &E| {
                (equal(v[0], x) && equal(v[1], y)) || (equal(v[1], x) && equal(v[0], y))
            };
            let valid = if conjugate {
                pair(r, x1, c) && pair(s, x0, sin)
            } else {
                (pair(r, x0, sin) && pair(s, x1, c)) || (pair(s, x0, sin) && pair(r, x1, c))
            };
            if valid {
                return Some(([x0, x1, c, sin], conjugate));
            }
        }
    }
    None
}

fn contiguous(v: &TensorView) -> bool {
    let mut stride = 1;
    for (&size, &s) in v.shape.iter().zip(&v.strides).rev() {
        if size != 1 && s != stride {
            return false;
        }
        stride *= size;
    }
    true
}

pub fn recognize(facts: &RegionFacts<'_>) -> Result<RopePattern, String> {
    let plan = facts.plan;
    let stores = domain_stores(facts)?;
    let last = stores.last().ok_or("empty region")?;
    let (mut a, mut b, mut axis) = concat(&last.expression)
        .ok_or("RoPE requires an explicit concatenation of the two rotation components")?;
    let mut tail = None;
    if equation(plan, a, b).is_none()
        && let Some((left, right, inner_axis)) = concat(a)
    {
        if inner_axis != axis {
            return Err("partial rotation concat axes differ".into());
        }
        tail = Some(tensor_view(plan, b).ok_or("unrotated tail is not a direct input view")?);
        (a, b, axis) = (left, right, inner_axis);
    }
    let ([x0, x1, c, s], conjugate) = equation(plan, a, b).ok_or(
        "not the paired multiply/subtract/add RoPE equation (casts/computed tables are not folded)",
    )?;
    let get = |e: &E| {
        tensor_view(plan, e).ok_or_else(|| "RoPE operand is not a proven tensor view".to_string())
    };
    let first = get(x0)?;
    let second = get(x1)?;
    let mut cos = get(c)?;
    let mut sin = get(s)?;
    let mut output = get(&E::Load(last.destination.clone()))?;
    let rank = first.shape.len();
    if output.shape.len() != rank {
        return Err("RoPE output rank differs from its component concatenation".into());
    }
    if first.value != second.value
        || first.shape != second.shape
        || first.strides != second.strides
        || first.offset != 0
        || axis >= rank
        || cos.value == first.value
        || sin.value == first.value
    {
        return Err("RoPE components must be disjoint paired views of the same input; tables must be separate values".into());
    }
    let mut input = first.clone();
    let (pairing, rotation_axis, rotary_dimension) = if axis + 1 == rank
        && first.shape[axis] == 1
        && rank >= 2
        && first.strides[axis - 1] == 2 * first.strides[axis]
        && second.offset == first.strides[axis]
    {
        if tail.is_some() {
            return Err("partial interleaved pair-axis view is not yet recognized".into());
        }
        let p = first.shape[axis - 1];
        if output.shape[..axis] != first.shape[..axis] || output.shape[axis] != 2 {
            return Err("interleaved output does not reconstruct the input pairs".into());
        }
        input.shape[axis - 1] = 2 * p;
        input.strides[axis - 1] = input.strides[axis];
        input.shape.pop();
        input.strides.pop();
        output.shape[axis - 1] *= 2;
        output.strides[axis - 1] = output.strides[axis];
        output.shape.pop();
        output.strides.pop();
        // Tables must broadcast over the singleton component axis.
        if cos.shape.len() != rank
            || sin.shape.len() != rank
            || cos.shape[axis] != 1
            || sin.shape[axis] != 1
        {
            return Err("interleaved tables must have a singleton pair axis".into());
        }
        cos.shape.pop();
        cos.strides.pop();
        sin.shape.pop();
        sin.strides.pop();
        (Pairing::Interleaved, axis - 1, 2 * p)
    } else if axis + 1 == rank && second.offset == first.shape[axis] * first.strides[axis] {
        let r = 2 * first.shape[axis];
        input.shape[axis] = *output.shape.get(axis).ok_or("output rank mismatch")?;
        (Pairing::SplitHalf, axis, r)
    } else {
        return Err("unsupported RoPE pairing/view: expected contiguous halves or an adjacent final pair axis".into());
    };
    if input.shape != output.shape
        || input.strides[rotation_axis] != 1
        || !contiguous(&input)
        || !contiguous(&output)
        || output.offset != 0
    {
        return Err(
            "RoPE input/output must describe matching complete contiguous logical views".into(),
        );
    }
    let d = input.shape[rotation_axis];
    if d < rotary_dimension {
        return Err("rotary dimension exceeds input axis".into());
    }
    if let Some(t) = &tail {
        let mut expected = input.clone();
        expected.shape[rotation_axis] = d - rotary_dimension;
        expected.offset = rotary_dimension;
        if expected != *t || d == rotary_dimension {
            return Err("partial RoPE tail must copy exactly the untouched input interval".into());
        }
    } else if d != rotary_dimension {
        return Err("partial RoPE has no explicit pass-through tail".into());
    }
    if cos.shape != sin.shape
        || cos.shape.len() != input.shape.len()
        || cos.shape[rotation_axis] != rotary_dimension / 2
        || cos
            .shape
            .iter()
            .enumerate()
            .any(|(i, n)| i != rotation_axis && *n != 1 && *n != input.shape[i])
    {
        return Err("cos/sin table access does not broadcast to the same rotation pairs".into());
    }
    let table_axes = cos
        .shape
        .iter()
        .enumerate()
        .filter_map(|(i, n)| (i != rotation_axis && *n > 1).then_some(i))
        .collect();
    let input_id = plan.value_instances().nth(input.value).unwrap().0;
    let projection = if let Some(producers) = facts
        .producers
        .get(&input_id)
        .filter(|ids| ids.iter().any(|id| id.index() < last.operation.index()))
    {
        if producers.len() != 1 || stores.len() != 2 || producers[0] != stores[0].operation {
            return Err("RoPE projection needs one preceding full-contraction GEMM store; extra/serial computations are not covered".into());
        }
        let p = &stores[0];
        let E::Matmul(mm) = &p.expression else {
            return Err("rotation producer is not a direct GEMM (intermediate casts/epilogues are not absorbed)".into());
        };
        let result = get(&E::Load(p.destination.clone()))?;
        let lhs = get(&mm[0])?;
        let rhs = get(&mm[1])?;
        if result.offset != 0
            || !contiguous(&result)
            || result.shape.len() != 2
            || lhs.shape.len() != 2
            || rhs.shape.len() != 2
            || result.shape != [lhs.shape[0], rhs.shape[1]]
            || lhs.shape[1] != rhs.shape[0]
            || result.shape.iter().product::<usize>() != input.shape.iter().product::<usize>()
        {
            return Err("projection and paired views do not cover the same GEMM result".into());
        }
        Some(Projection {
            operation: p.operation,
            lhs,
            rhs,
            result,
        })
    } else {
        if stores.len() != 1 {
            return Err("region contains additional computations outside RoPE".into());
        }
        None
    };
    if input.shape.iter().product::<usize>()
        != plan
            .value_instance(input_id)
            .unwrap()
            .shape()
            .iter()
            .product::<usize>()
    {
        return Err("RoPE input is not a complete logical tensor view".into());
    }
    Ok(RopePattern {
        input,
        output,
        first,
        second,
        cos,
        sin,
        pairing,
        rotation_axis,
        rotary_dimension,
        conjugate,
        table_axes,
        passthrough: tail,
        projection,
        operations: facts.scope.operations.clone(),
        stores,
    })
}
