//! Concrete CUDA implementation candidates and physical tensor program plans.
//!
//! Callers resolve logical semantics and tensor presentations, enumerate applicable
//! implementations, and assemble a graph with [`PhysicalPlanBuilder`]. Finalized
//! plans own their physical data and do not retain a compiler's source graph.

pub mod compile;
mod config;
mod dtype;
pub mod emit;
mod fusion;
mod implementation;
mod physical;
pub mod platform;
mod python;

pub use compile::{
    CompileConfig, CompileDiagnostics, CompileError, CudaArtifact, compile, compile_with_config,
};
pub use config::LoweringConfig;
pub use dtype::DType;
pub use emit::{CudaSource, EmitError, emit};
pub use fusion::{FusionError, FusionRewrite, FusionRule, fuse};
pub use implementation::{
    AllGatherImplementation, AttributeSet, GemmImplementation, ImplementationDefinition,
    ImplementationId, ImplementationInstance, all_gather_implementations, fusion_rules,
    gemm_implementations,
};
pub use physical::{
    Action, ActionId, CommunicationKind, CommunicationOperation, ComputeOperation, Operation,
    OperationId, OperationPayload, PhysicalInvariantError, PhysicalPlan, PhysicalPlanBuilder,
    Storage, TensorBinding, ValueInstance, ValueInstanceId,
};
pub use platform::{CudaTargetCapability, TargetCapability};

#[cfg(test)]
mod tests;
