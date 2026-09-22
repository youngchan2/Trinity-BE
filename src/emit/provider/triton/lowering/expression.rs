//! Validate expression shapes and lower operators without arithmetic rewrites.
use super::super::plan::{Expr, ExprKind};
use super::super::shape::{constant, product};
use super::super::{Error, Options, TileAccess, invalid};
use crate::analysis::*;
use std::collections::BTreeSet;

pub(crate) fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, Error> {
    let rank = a.len().max(b.len());
    let mut result = vec![1; rank];
    for (i, d) in result.iter_mut().enumerate() {
        let x = a.get(i.wrapping_sub(rank - a.len())).copied().unwrap_or(1);
        let y = b.get(i.wrapping_sub(rank - b.len())).copied().unwrap_or(1);
        if x != y && x != 1 && y != 1 {
            return Err(invalid(format!("incompatible broadcast {a:?} and {b:?}")));
        }
        *d = x.max(y);
    }
    Ok(result)
}

fn axis(expr: &ValueExpr, rank: usize) -> Result<usize, Error> {
    let ValueExpr::Literal(n) = expr else {
        return Err(invalid("axis must be an integer literal"));
    };
    let mut n = n
        .parse::<i64>()
        .map_err(|_| invalid("axis must be an integer literal"))?;
    if n < 0 {
        n += rank as i64;
    }
    if n < 0 || n as usize >= rank {
        return Err(invalid("axis out of range"));
    }
    Ok(n as usize)
}

pub(super) fn expression(
    expr: &ValueExpr,
    accesses: &[TileAccess],
    options: &Options,
) -> Result<Expr, Error> {
    let ValueExpr::Apply(op, args) = expr else {
        return Ok(match expr {
            ValueExpr::Literal(n) => {
                if !n.parse::<f64>().is_ok_and(f64::is_finite) {
                    return Err(invalid("nonfinite scalar literal"));
                }
                Expr {
                    kind: ExprKind::Scalar(format!("{:?}", n.parse::<f64>().unwrap())),
                    shape: vec![],
                }
            }
            ValueExpr::Index(i) => {
                if i.loop_dependencies().is_empty() {
                    constant(i, options)?;
                }
                Expr {
                    kind: ExprKind::Index(i.clone()),
                    shape: vec![],
                }
            }
            ValueExpr::Load(id) => Expr {
                kind: ExprKind::Load(*id),
                shape: accesses[id.index()].shape.clone(),
            },
            _ => unreachable!(),
        });
    };
    if op == "cast" {
        if args.len() != 2 {
            return Err(invalid("cast expects dtype and value"));
        }
        let ValueExpr::Index(IndexExpr::Symbol(dtype)) = &args[0] else {
            return Err(invalid("cast dtype must be a symbol"));
        };
        let dtype = match dtype.as_str() {
            "fp16" | "float16" | "half" => "float16",
            "fp32" | "float32" | "float" => "float32",
            "bf16" | "bfloat16" => "bfloat16",
            "int32" | "int64" | "int8" | "uint8" | "bool" => dtype.as_str(),
            _ => return Err(invalid(format!("unsupported cast dtype {dtype}"))),
        };
        let value = expression(&args[1], accesses, options)?;
        return Ok(Expr {
            shape: value.shape.clone(),
            kind: ExprKind::Cast(dtype.into(), Box::new(value)),
        });
    }
    if op == "const" {
        if args.len() != 1 {
            return Err(invalid("const expects one value"));
        }
        return expression(&args[0], accesses, options);
    }
    let arity = match op.as_str() {
        "sqr" | "exp" | "sqrt" | "sigmoid" | "erf" | "abs" | "transpose" => 1,
        "+" | "-" | "*" | "/" | "<=" | "max" | "min" | "@" | "rsum" | "rmax" | "rmin" | "bcast"
        | "unsqueeze" | "squeeze" => 2,
        "permute" | "permute3" | "permute4" => args.len(),
        "concat" => 3,
        _ => return Err(invalid(format!("unsupported value operator {op}"))),
    };
    if args.len() != arity || args.is_empty() {
        return Err(invalid(format!("{op}: wrong argument count")));
    }
    let left = expression(&args[0], accesses, options)?;
    let mut shape = left.shape.clone();
    let kind = match op.as_str() {
        "sqr" | "exp" | "sqrt" | "sigmoid" | "erf" | "abs" => {
            ExprKind::Unary(op.clone(), Box::new(left))
        }
        "transpose" | "permute" | "permute3" | "permute4" => {
            let order = if op == "transpose" {
                if shape.len() < 2 {
                    return Err(invalid("transpose requires rank >= 2"));
                }
                let mut order: Vec<_> = (0..shape.len()).collect();
                let n = order.len();
                order.swap(n - 1, n - 2);
                order
            } else {
                args[1..]
                    .iter()
                    .map(|a| axis(a, shape.len()))
                    .collect::<Result<Vec<_>, _>>()?
            };
            if order.len() != shape.len()
                || order.iter().copied().collect::<BTreeSet<_>>().len() != shape.len()
            {
                return Err(invalid("invalid permutation"));
            }
            shape = order.iter().map(|i| shape[*i]).collect();
            ExprKind::Permute(order, Box::new(left))
        }
        "bcast" | "unsqueeze" => {
            let ax = axis(&args[1], shape.len() + 1)?;
            shape.insert(ax, 1);
            ExprKind::Transform(op.clone(), ax, Box::new(left))
        }
        "squeeze" => {
            let ax = axis(&args[1], shape.len())?;
            if shape[ax] != 1 {
                return Err(invalid("squeeze can only remove a singleton tile axis"));
            }
            shape.remove(ax);
            ExprKind::Transform(op.clone(), ax, Box::new(left))
        }
        "rsum" | "rmax" | "rmin" => {
            let ax = axis(&args[1], shape.len())?;
            shape.remove(ax);
            ExprKind::Reduce(op.clone(), ax, Box::new(left))
        }
        "concat" => {
            let right = expression(&args[1], accesses, options)?;
            let ax = axis(&args[2], shape.len())?;
            if shape.len() != right.shape.len()
                || shape
                    .iter()
                    .zip(&right.shape)
                    .enumerate()
                    .any(|(i, (a, b))| i != ax && a != b)
            {
                return Err(invalid("concat requires matching non-concatenated axes"));
            }
            shape[ax] = logical_extent(&left, ax, accesses)
                .checked_add(logical_extent(&right, ax, accesses))
                .and_then(usize::checked_next_power_of_two)
                .ok_or_else(|| invalid("concat extent overflow"))?;
            if product(&shape)? > 1_048_576 {
                return Err(invalid("concat exceeds Triton's element limit"));
            }
            ExprKind::Concat(ax, Box::new(left), Box::new(right))
        }
        _ => {
            let right = expression(&args[1], accesses, options)?;
            if op == "@" {
                let n = left.shape.len();
                if n < 2 || right.shape.len() != n || left.shape[n - 1] != right.shape[n - 2] {
                    return Err(invalid(format!(
                        "unsupported dot shapes {:?}, {:?}",
                        left.shape, right.shape
                    )));
                }
                shape = broadcast(&left.shape[..n - 2], &right.shape[..n - 2])?;
                shape.extend([left.shape[n - 2], right.shape[n - 1]]);
                if left.shape[n - 2] < 16 || left.shape[n - 1] < 16 || right.shape[n - 1] < 16 {
                    let mut expanded = shape.clone();
                    expanded.push(left.shape[n - 1]);
                    if product(&expanded)? > 1_048_576 {
                        return Err(invalid("small-dot fallback exceeds Triton's element limit"));
                    }
                }
                ExprKind::Dot(Box::new(left), Box::new(right))
            } else {
                shape = broadcast(&left.shape, &right.shape)?;
                ExprKind::Binary(op.clone(), Box::new(left), Box::new(right))
            }
        }
    };
    Ok(Expr { kind, shape })
}

fn logical_extent(expr: &Expr, axis: usize, accesses: &[TileAccess]) -> usize {
    match &expr.kind {
        ExprKind::Load(id) => accesses[id.index()].axes[axis].width,
        ExprKind::Unary(_, x) | ExprKind::Cast(_, x) => logical_extent(x, axis, accesses),
        ExprKind::Permute(order, x) => logical_extent(x, order[axis], accesses),
        ExprKind::Transform(op, ax, x) if op == "squeeze" => {
            logical_extent(x, axis + usize::from(axis >= *ax), accesses)
        }
        ExprKind::Transform(_, ax, x) => {
            if axis == *ax {
                1
            } else {
                logical_extent(x, axis - usize::from(axis > *ax), accesses)
            }
        }
        ExprKind::Reduce(_, ax, x) => logical_extent(x, axis + usize::from(axis >= *ax), accesses),
        ExprKind::Concat(ax, a, b) if axis == *ax => {
            logical_extent(a, axis, accesses) + logical_extent(b, axis, accesses)
        }
        ExprKind::Concat(_, a, _) => logical_extent(a, axis, accesses),
        ExprKind::Binary(_, a, b) => {
            let width = |x: &Expr| {
                if axis + x.shape.len() < expr.shape.len() {
                    1
                } else {
                    logical_extent(x, axis + x.shape.len() - expr.shape.len(), accesses)
                }
            };
            width(a).max(width(b))
        }
        ExprKind::Dot(a, b) => {
            let n = expr.shape.len();
            if axis == n - 1 {
                logical_extent(b, axis, accesses)
            } else if axis == n - 2 {
                logical_extent(a, axis, accesses)
            } else {
                logical_extent(a, axis, accesses).max(logical_extent(b, axis, accesses))
            }
        }
        _ => expr.shape[axis],
    }
}
