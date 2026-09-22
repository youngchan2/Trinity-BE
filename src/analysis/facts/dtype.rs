//! Logical tensor/storage dtypes, independent of provider register arithmetic.
//! The IR has no tensor dtype annotations: ABI defaults and caller overrides
//! anchor propagation. Half-typed exp/reduction/GEMM do not widen storage.
use crate::DType;
use crate::analysis::{AccessId, IndexExpr, ResolveError, ScheduledIr, TensorKind, ValueExpr};
use std::collections::BTreeMap;

/// Join logical operand types. Untyped scalars do not promote tensor operands.
pub fn merge(a: Option<DType>, b: Option<DType>) -> Option<DType> {
    match (a, b) {
        (Some(a), Some(b)) if a != b => Some(DType::Fp32),
        (a, b) => a.or(b),
    }
}

/// Floating storage contract after an explicit cast. Integer/bool expression
/// casts remain actual casts, but storage currently supports only floating types.
pub fn cast_storage_dtype(name: &str) -> Result<DType, ResolveError> {
    match name {
        "fp16" | "float16" | "half" => Ok(DType::Fp16),
        "bf16" | "bfloat16" => Ok(DType::Bf16),
        "fp32" | "float32" | "float" | "int32" | "int64" | "int8" | "uint8" | "bool" => {
            Ok(DType::Fp32)
        }
        _ => Err(ResolveError(format!("unsupported cast dtype {name}"))),
    }
}

/// Interpret only the logical data operands, excluding shape/axis expressions.
/// Arity and shape validation remain the IR reader/plan's responsibility.
pub fn expression(
    expr: &ValueExpr,
    load: &impl Fn(AccessId) -> Option<DType>,
) -> Result<Option<DType>, ResolveError> {
    let ValueExpr::Apply(op, args) = expr else {
        return Ok(match expr {
            ValueExpr::Load(id) => load(*id),
            _ => None,
        });
    };
    let invalid = || ResolveError(format!("invalid logical dtype expression {op}"));
    let operand = |i| expression(args.get(i).ok_or_else(invalid)?, load);
    match op.as_str() {
        "cast" => {
            let Some(ValueExpr::Index(IndexExpr::Symbol(dtype))) = args.first() else {
                return Err(invalid());
            };
            if args.len() != 2 {
                return Err(invalid());
            }
            cast_storage_dtype(dtype).map(Some)
        }
        "const" | "sqr" | "exp" | "sqrt" | "sigmoid" | "erf" | "abs" | "relu" | "transpose"
        | "permute" | "permute3" | "permute4" | "rsum" | "rmax" | "rmin" | "bcast"
        | "unsqueeze" | "squeeze" => operand(0),
        "+" | "-" | "*" | "/" | "<=" | "max" | "min" | "@" | "concat" => {
            Ok(merge(operand(0)?, operand(1)?))
        }
        _ => Err(ResolveError(format!(
            "unsupported logical dtype operator {op}"
        ))),
    }
}

/// Resolve one dtype per TensorId before constructing the common PhysicalPlan.
/// Recurrences reach a fixed point from typed operands; scalar-only producers
/// receive the default after anchored recurrences have converged.
pub fn resolve(
    ir: &ScheduledIr,
    default: DType,
    overrides: &BTreeMap<String, DType>,
) -> Result<Vec<DType>, ResolveError> {
    let inferred: Vec<_> = ir
        .tensors()
        .iter()
        .map(|t| {
            t.declarations.contains(&TensorKind::Intermediate)
                && !t.declarations.contains(&TensorKind::Input)
                && !t.declarations.contains(&TensorKind::Output)
                && !overrides.contains_key(&t.name)
        })
        .collect();
    let mut dtypes: Vec<_> = ir
        .tensors()
        .iter()
        .enumerate()
        .map(|(i, t)| (!inferred[i]).then(|| overrides.get(&t.name).copied().unwrap_or(default)))
        .collect();
    loop {
        let mut changed = false;
        for statement in ir.statements() {
            let id = ir
                .access(
                    *statement
                        .accesses
                        .last()
                        .ok_or_else(|| ResolveError("statement has no write".into()))?,
                )
                .tensor
                .index();
            if inferred[id] {
                let value = expression(&statement.expression, &|a| {
                    dtypes[ir.access(a).tensor.index()]
                })?;
                let dtype = merge(dtypes[id], value);
                changed |= dtype != dtypes[id];
                dtypes[id] = dtype;
            }
        }
        if !changed {
            if dtypes.iter().any(Option::is_none) {
                for dtype in &mut dtypes {
                    dtype.get_or_insert(default);
                }
                continue;
            }
            return Ok(dtypes
                .into_iter()
                .map(|d| d.expect("resolved logical dtype"))
                .collect());
        }
    }
}
