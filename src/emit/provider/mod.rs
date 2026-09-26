//! Backend-neutral candidate discovery. Native composition contracts live in `native`.

use super::execution::ExecutionModel;
use super::prepare::PreparedPlan;
use crate::{Loop, OperationId};
use thiserror::Error;

pub(super) mod cute;
pub(crate) mod python;
pub(super) mod quack;
pub mod triton;
pub(super) use cute::CuTeKernelProvider;

/// Read-only program and loop context for kernel specification.
pub(super) struct KernelContext<'a, 'plan> {
    pub prepared: &'a PreparedPlan<'plan>,
    pub execution: ExecutionModel,
    pub operation: OperationId,
    /// Enclosing loops, outermost first, without evaluating their bounds.
    pub loops: &'a [&'plan Loop],
}

/// Enumerate operation implementations without requiring native phases or ABI bindings.
pub(super) trait KernelProvider {
    fn name(&self) -> &str;
    fn candidates(
        &self,
        context: &KernelContext<'_, '_>,
    ) -> Result<Vec<super::candidate::CandidateSpecification>, ProviderError>;
}

#[derive(Debug, Error)]
pub(super) enum ProviderError {
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("provider error: {0}")]
    Failed(String),
}
