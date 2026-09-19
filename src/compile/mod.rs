//! Compilation into CUDA shared-library artifacts.

pub mod cuda;
mod source;

pub use source::{BufferBindingRequirement, CudaRequirements, CudaSource};

pub use cuda::{
    CompileConfig, CompileDiagnostics, CompileError, CudaArtifact, compile, compile_with_config,
};
