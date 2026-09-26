//! Provider selection, native composition, and CUDA emission.

use crate::{CudaSource, PhysicalPlan};
use thiserror::Error;

mod candidate;
mod execution;
pub(crate) mod fusion;
pub(crate) mod implementation;
mod native;
mod prepare;
pub(crate) mod provider;
mod region;
mod request;
mod wrapper;
pub use region::{RegionCandidates, RegionKernelCandidate, region_candidates};

pub use candidate::{
    CandidateKind, CandidateRejection, KernelCandidate, OperationCandidates, kernel_candidates,
};
pub use provider::quack::{
    QuackApi, QuackPatternAnalysis, QuackPatternKind, QuackRegionOperation,
    QuackRegionSpecification, QuackSpecification,
};
pub use provider::triton::TritonSpecification;
pub use provider::triton::{TritonKernelProvider, TritonProgram};
pub use wrapper::{PythonProgram, emit_python};

/// Emit an ordered single-GPU program through the Triton fallback provider.
pub fn emit_triton(
    plan: &PhysicalPlan,
    options: crate::emit::provider::triton::Options,
) -> Result<String, crate::emit::provider::triton::Error> {
    Ok(TritonKernelProvider.lower_program(plan, options)?.emit())
}
pub use request::{KernelRequest, TensorArgument};

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
    native::emit(plan)
}
