//! CUDA execution placement and final source assembly, independent of providers.
mod execution;
mod render;
mod validation;

use super::{EmitError, combine::CombinedPlan, prepare::PreparedPlan};
use crate::CudaSource;

fn invalid(reason: impl Into<String>) -> EmitError {
    EmitError::InvalidExecution {
        reason: reason.into(),
    }
}

fn unsupported(reason: impl Into<String>) -> EmitError {
    EmitError::UnsupportedExecution {
        reason: reason.into(),
    }
}

pub(super) fn emit(
    prepared: PreparedPlan<'_>,
    combined: &CombinedPlan<'_>,
) -> Result<CudaSource, EmitError> {
    let bindings = super::native::bindings::build(prepared.plan)?;
    let execution = execution::build(&prepared, &bindings, combined)?;
    validation::validate(prepared.plan, combined, &execution)?;
    render::render(&prepared, &bindings, combined, &execution)
}

#[cfg(test)]
mod tests;
