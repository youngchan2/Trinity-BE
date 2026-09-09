//! Compilation into CUDA shared-library artifacts.

pub mod cuda;

pub use cuda::{
    CompileConfig, CompileDiagnostics, CompileError, CudaArtifact, compile, compile_with_config,
};
