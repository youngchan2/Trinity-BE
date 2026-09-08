//! Source emission entry points. Platform-specific code belongs to its backend
//! module; the existing CUDA API remains available through these re-exports.

pub mod cuda;

pub use cuda::{
    BufferBindingRequirement, CudaImplementation, CudaRequirements, CudaSource, Dependency,
    EmitError, Execution, OperationEmission, Region, Stage, Task, Work, emit, render_template,
};
