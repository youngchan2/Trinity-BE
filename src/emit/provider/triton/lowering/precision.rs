//! Type queries for lowered Triton expressions, using finalized common value dtypes.
//! Tensor dtype inference is owned by analysis::dtype; no storage decisions here.
use super::super::TensorDType;
use super::super::plan::{Expr, ExprKind};
fn logical_dtype(
    expr: &Expr,
    load: &impl Fn(crate::analysis::AccessId) -> Option<TensorDType>,
) -> Option<TensorDType> {
    match &expr.kind {
        ExprKind::Scalar(_) | ExprKind::Index(_) => None,
        ExprKind::Load(id) => load(*id),
        ExprKind::Cast(dtype, _) => Some(
            crate::analysis::dtype::cast_storage_dtype(dtype)
                .expect("validated cast")
                .into(),
        ),
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
    crate::analysis::dtype::merge(a.map(Into::into), b.map(Into::into)).map(Into::into)
}
