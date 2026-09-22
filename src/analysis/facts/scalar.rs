//! Scalar expressions and logical bounds shared by implementation providers.
use crate::analysis::{IndexExpr, ResolveError, ScheduledIr, ScopeId, invalid};
use std::collections::{BTreeMap, BTreeSet};

pub fn symbols(expr: &IndexExpr) -> BTreeSet<String> {
    match expr {
        IndexExpr::Symbol(s) => BTreeSet::from([s.clone()]),
        IndexExpr::Apply(_, args) => args.iter().flat_map(symbols).collect(),
        _ => BTreeSet::new(),
    }
}

pub fn constant(expr: &IndexExpr, bindings: &BTreeMap<String, i64>) -> Result<i64, ResolveError> {
    match expr {
        IndexExpr::Integer(v) => Ok(*v),
        IndexExpr::Symbol(name) => bindings
            .get(name)
            .copied()
            .ok_or_else(|| invalid(format!("missing symbol {name}"))),
        IndexExpr::Apply(op, args) if args.len() == 2 => {
            let a = constant(&args[0], bindings)?;
            let b = constant(&args[1], bindings)?;
            let value = match op.as_str() {
                "+" => a.checked_add(b),
                "-" => a.checked_sub(b),
                "*" => a.checked_mul(b),
                "/" | "//" if b > 0 => Some(a.div_euclid(b)),
                _ => None,
            };
            value.ok_or_else(|| invalid(format!("invalid constant expression {expr:?}")))
        }
        _ => Err(invalid(format!(
            "expected compile-time integer, got {expr:?}"
        ))),
    }
}

pub fn positive(expr: &IndexExpr, bindings: &BTreeMap<String, i64>) -> Result<usize, ResolveError> {
    let v = constant(expr, bindings)?;
    if v <= 0 {
        return Err(invalid(format!("expected positive extent, got {v}")));
    }
    usize::try_from(v).map_err(|_| invalid("extent overflow"))
}

pub fn product(shape: &[usize]) -> Result<usize, ResolveError> {
    shape.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b)
            .filter(|n| *n <= i64::MAX as usize)
            .ok_or_else(|| invalid("shape product overflow"))
    })
}

pub fn validate_index(
    expr: &IndexExpr,
    bindings: &BTreeMap<String, i64>,
) -> Result<(), ResolveError> {
    match expr {
        IndexExpr::LoopVar(_) => Ok(()),
        IndexExpr::Apply(op, args)
            if args.len() == 2 && ["+", "-", "*", "/", "//"].contains(&op.as_str()) =>
        {
            for arg in args {
                validate_index(arg, bindings)?;
            }
            if ["/", "//"].contains(&op.as_str()) {
                positive(&args[1], bindings)?;
            }
            Ok(())
        }
        _ => constant(expr, bindings).map(|_| ()),
    }
}

pub fn loop_range(
    ir: &ScheduledIr,
    id: ScopeId,
    bindings: &BTreeMap<String, i64>,
) -> Result<(i64, i64, usize), ResolveError> {
    let info = ir.scope(id).loop_info.as_ref().unwrap();
    let start = constant(&info.start, bindings)?;
    let end = constant(&info.end, bindings)?;
    let step = positive(&info.step, bindings)?;
    if start < 0 || end <= start {
        return Err(invalid(format!(
            "{}: require nonempty nonnegative static loop bounds",
            info.variable
        )));
    }
    Ok((start, end, step))
}
