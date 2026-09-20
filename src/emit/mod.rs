//! Entry point for the kernel-provider emitter under construction.
//! The previous CUDA pipeline is reference-only under old/emit-rewrite.

use crate::{CudaSource, PhysicalPlan};
use thiserror::Error;

mod collect;
#[expect(dead_code)]
mod combine;
mod execution;
mod prepare;
#[expect(dead_code)]
mod provider;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EmitError {
    #[error(
        "CUDA emission pipeline is incomplete; execution planning and CUDA rendering are not connected"
    )]
    Unavailable,
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
/// The remaining emission pipeline is not yet available.
pub fn emit(plan: &PhysicalPlan) -> Result<CudaSource, EmitError> {
    let prepared = prepare::prepare(plan);
    let execution = execution::plan_execution(prepared.plan);
    let provider = provider::CuTeKernelProvider;
    let selected = collect::collect(&prepared, execution, &[&provider])?;
    let _combined = combine::combine(&prepared, execution, selected)?;

    Err(EmitError::Unavailable)
}
