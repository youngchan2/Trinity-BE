//! Common source emission entry points and platform dispatch.

mod bindings;
pub mod cuda;
mod error;
mod prepare;
mod source;

use crate::PhysicalPlan;

pub use bindings::BufferBindingRequirement;
pub(crate) use bindings::BufferBindings;
pub use error::EmitError;
pub(crate) use prepare::prepare;
pub(crate) use source::EmittedSource;

pub use cuda::{
    Accesses, Binding, Body, Code, CudaImplementation, CudaRequirements, CudaSource, Dependency,
    Execution, Grid, OperationSchedule, PersistentExecution, Phase, Region, Resources, Stage,
    StageAccessPattern, StreamedExecution, StreamedLaunch, Symbol, SymbolId, Task, TaskScheduling,
    Work, render_template,
};

/// Generate source and launch requirements from a physical plan.
///
/// # Errors
///
/// Returns an error for unsupported plans, invalid backend contracts, or
/// failures while preparing or rendering the program.
pub fn emit(plan: &PhysicalPlan) -> Result<CudaSource, EmitError> {
    match prepare(plan)?.render()? {
        EmittedSource::Cuda(source) => Ok(source),
    }
}

/// Validate fusion candidates without rendering source.
pub(crate) fn validate(plan: &PhysicalPlan) -> Result<(), EmitError> {
    prepare(plan).map(|_| ())
}
