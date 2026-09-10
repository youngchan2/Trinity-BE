//! Source emission entry points. Platform-specific code belongs to its backend
//! module; the existing CUDA API remains available through these re-exports.

pub mod cuda;

pub use cuda::{
    Binding, Body, BufferBindingRequirement, Code, CudaImplementation, CudaRequirements,
    CudaSource, Dependency, EmitError, Execution, OperationSchedule, Phase, Region, Resources,
    Stage, Symbol, SymbolId, Task, Work, emit, render_template,
};
