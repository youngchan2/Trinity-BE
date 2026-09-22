//! CuTe operator implementations and their specifications.

use crate::emit::native::access;
mod gemm;
mod pointwise;
mod reduce_sum;
mod render;
#[cfg(test)]
mod tests;

pub(super) use gemm::hopper::HopperGemmKernel;
pub(in crate::emit) use gemm::hopper::HopperGemmSpecification;
pub(super) use gemm::sm80::Sm80GemmKernel;
pub(in crate::emit) use gemm::sm80::Sm80GemmSpecification;
pub(super) use pointwise::PointwiseKernel;
pub(in crate::emit) use pointwise::PointwiseSpecification;
pub(super) use reduce_sum::ReduceSumKernel;
pub(in crate::emit) use reduce_sum::ReduceSumSpecification;

use crate::emit::provider::{KernelContext, ProviderError};
use crate::{Expression, Loop, LoopKind};

fn unsupported(message: impl Into<String>) -> ProviderError {
    ProviderError::Unsupported(message.into())
}

/// The innermost single-operation serial loop carrying implicit-zero accumulation.
fn accumulation<'plan>(
    context: &KernelContext<'_, 'plan>,
) -> Result<(&'plan Loop, &'plan Expression), ProviderError> {
    let plan = context.prepared.plan;
    let expression = plan
        .operation(context.operation)
        .ok_or_else(|| ProviderError::Failed("missing operation".into()))?
        .expression();
    let loop_ = context
        .loops
        .last()
        .copied()
        .filter(|l| l.kind == LoopKind::Sequential && l.body.len() == 1)
        .ok_or_else(|| unsupported("expected a single-operation sequential accumulation loop"))?;
    let rhs = crate::plan::accumulation_rhs(expression, &loop_.domain.variable)
        .ok_or_else(|| unsupported("expected zero-initialized accumulation"))?;
    Ok((loop_, rhs))
}
