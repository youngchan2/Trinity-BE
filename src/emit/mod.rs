//! Provider selection, native composition, and CUDA emission.

use crate::{CudaSource, PhysicalPlan};
use thiserror::Error;

mod collect;
mod combine;
mod cuda;
mod execution;
mod prepare;
#[expect(dead_code)]
mod provider;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EmitError {
    #[error("the requested CUDA emission pipeline is unavailable")]
    Unavailable,
    #[error("unsupported CUDA execution: {reason}")]
    UnsupportedExecution { reason: String },
    #[error("invalid CUDA execution: {reason}")]
    InvalidExecution { reason: String },
    #[error("CUDA rendering failed: {message}")]
    Render { message: String },
    #[error("provider {provider} failed for operation {operation}: {message}")]
    Provider {
        operation: usize,
        provider: String,
        message: String,
    },
    #[error("no provider supports operation {operation}: {reasons:?}")]
    NoProvider {
        operation: usize,
        reasons: Vec<String>,
    },
    #[error("kernel combination failed: {reason}")]
    Combination { reason: String },
}

/// Selects kernels per operation and combines compatible kernels from the same provider.
/// Streamed execution uses one CTA per parallel coordinate. Persistent emission
/// remains unavailable while its task and scheduler planning is rebuilt.
pub fn emit(plan: &PhysicalPlan) -> Result<CudaSource, EmitError> {
    if plan.world_size() != 1 {
        return Err(EmitError::UnsupportedExecution {
            reason: "Persistent emission is not implemented".into(),
        });
    }
    let prepared = prepare::prepare(plan)?;
    let execution = execution::plan_execution(prepared.plan);
    let provider = provider::CuTeKernelProvider;
    let selected = collect::collect(&prepared, execution, &[&provider])?;
    let combined = combine::combine(&prepared, execution, selected)?;
    cuda::emit(prepared, &combined)
}
