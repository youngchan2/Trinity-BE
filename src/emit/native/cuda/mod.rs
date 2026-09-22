//! CUDA execution placement and final source assembly, independent of providers.
mod execution;
mod render;
mod validation;

use crate::CudaSource;
use crate::emit::{EmitError, native::combine::CombinedPlan, prepare::PreparedPlan};

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

pub(in crate::emit) fn emit(
    prepared: PreparedPlan<'_>,
    combined: &CombinedPlan<'_>,
) -> Result<CudaSource, EmitError> {
    let bindings = crate::emit::native::bindings::build(prepared.plan)?;
    let execution = execution::build(&prepared, &bindings, combined)?;
    validation::validate(prepared.plan, combined, &execution)?;
    render::render(&prepared, &bindings, combined, &execution)
}

#[cfg(test)]
mod tests;
