//! CUDA source generation.
//!
//! Statement boundaries, coordinate-taking bodies and logical validation are
//! shared. Streamed maps domains to grids; Persistent alone expands tile tasks,
//! physical stage accesses and scheduler metadata. Native launch ABIs are stable.

mod access;
pub(crate) mod address;
mod backend;
mod body;
mod domain;
mod execution;
mod indexing;
mod persistent;
mod prepare;
mod regions;
mod render;
mod requirements;
mod source;
mod streamed;
mod validation;

#[cfg(test)]
mod tests;

pub(super) use prepare::prepare;

pub use super::{BufferBindingRequirement, EmitError, emit};
pub use backend::{
    Accesses, AccumulationBody, AccumulationScope, AccumulationTile, CudaImplementation,
    CudaPhaseTemplate, OperationSchedule, Region, StageAccessPattern, Work,
};
pub use body::{Binding, Body, Code, Phase, Resources, Symbol, SymbolId};

pub use domain::Task;
pub use execution::Execution;
pub use persistent::{Dependency, PersistentExecution, Stage, TaskScheduling};
pub use render::render_template;
pub use requirements::CudaRequirements;
pub use source::CudaSource;
pub use streamed::{Grid, StreamedExecution, StreamedLaunch};
