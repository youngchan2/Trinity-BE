//! Logical/storage types are independent of FP32 register computation.
//! exp, reduction and accumulation do not widen a half-typed value's contract.
use super::super::plan::{Expr, ExprKind};
use super::super::{Options, TensorDType};
use crate::analysis::{ScheduledIr, TensorKind};

pub(super) fn resolve(
    ir: &ScheduledIr,
    expressions: &[Expr],
    options: &Options,
) -> Vec<TensorDType> {
    let inferred: Vec<_> = ir
        .tensors()
        .iter()
        .map(|t| {
            t.declarations.contains(&TensorKind::Intermediate)
                && !t.declarations.contains(&TensorKind::Input)
                && !t.declarations.contains(&TensorKind::Output)
                && !options.dtypes.contains_key(&t.name)
        })
        .collect();
    let mut dtypes: Vec<_> = ir
        .tensors()
        .iter()
        .enumerate()
        .map(|(i, t)| {
            (!inferred[i]).then(|| {
                options
                    .dtypes
                    .get(&t.name)
                    .copied()
                    .unwrap_or(options.default_dtype)
            })
        })
        .collect();

    // Propagate only logical operand types, forward through definitions. Unknown
    // recurrence values must not seed a spurious FP16/BF16 -> FP32 promotion.
    loop {
        let mut changed = false;
        for (statement, expr) in ir.statements().iter().zip(expressions) {
            let tid = ir
                .access(*statement.accesses.last().unwrap())
                .tensor
                .index();
            if inferred[tid] {
                let value = logical_dtype(expr, &|id| dtypes[ir.access(id).tensor.index()]);
                let dtype = merge_dtype(dtypes[tid], value);
                changed |= dtype != dtypes[tid];
                dtypes[tid] = dtype;
            }
        }
        if !changed {
            // Scalar-only producers have no operand type. Once anchored
            // recurrences have converged, assign the default and propagate it
            // through consumers too, so storage and GEMM typing stay consistent.
            if dtypes.iter().any(Option::is_none) {
                for dtype in &mut dtypes {
                    dtype.get_or_insert(options.default_dtype);
                }
                continue;
            }
            return dtypes
                .into_iter()
                .map(|d| d.unwrap_or(options.default_dtype))
                .collect();
        }
    }
}

fn logical_dtype(
    expr: &Expr,
    load: &impl Fn(crate::analysis::AccessId) -> Option<TensorDType>,
) -> Option<TensorDType> {
    match &expr.kind {
        ExprKind::Scalar(_) | ExprKind::Index(_) => None,
        ExprKind::Load(id) => load(*id),
        ExprKind::Cast(dtype, _) => Some(match dtype.as_str() {
            "float16" => TensorDType::Fp16,
            "bfloat16" => TensorDType::Bf16,
            _ => TensorDType::Fp32,
        }),
        ExprKind::Unary(_, x)
        | ExprKind::Reduce(_, _, x)
        | ExprKind::Permute(_, x)
        | ExprKind::Transform(_, _, x) => logical_dtype(x, load),
        ExprKind::Binary(_, a, b) | ExprKind::Dot(a, b) | ExprKind::Concat(_, a, b) => {
            merge_dtype(logical_dtype(a, load), logical_dtype(b, load))
        }
    }
}

impl super::super::TritonPlan {
    /// A half-typed exp/reduction/dot result remains a half GEMM operand even
    /// though its register computation and accumulator are FP32. Explicit FP32
    /// inputs/casts/storage retain FP32. Numerical stabilization belongs in IR.
    pub(crate) fn value_dtype(&self, expr: &Expr) -> Option<TensorDType> {
        logical_dtype(expr, &|id| {
            Some(self.tensor_dtype(self.analysis.access(id).tensor))
        })
    }
}

pub(crate) fn merge_dtype(a: Option<TensorDType>, b: Option<TensorDType>) -> Option<TensorDType> {
    match (a, b) {
        (Some(a), Some(b)) if a != b => Some(TensorDType::Fp32),
        (a, b) => a.or(b),
    }
}
