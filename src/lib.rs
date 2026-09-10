//! Scheduled Trinity IR analysis, Triton source generation, and CUDA physical plans.
//!
//! [`triton::compile`] emits Python kernels from Trinity IR. [`lower_loop_ir`] and
//! [`PhysicalPlanBuilder`] produce CUDA plans for [`emit`] and [`compile`]. These
//! entry points retain their own analysis and lowering contracts.

pub mod analysis;
pub mod compile;
mod config;
mod dtype;
pub mod emit;
mod fusion;
mod implementation;
mod loop_ir;
mod physical;
pub mod platform;
mod python;
pub mod triton;

pub use compile::{
    CompileConfig, CompileDiagnostics, CompileError, CudaArtifact, compile, compile_with_config,
};
pub use config::LoweringConfig;
pub use dtype::DType;
pub use emit::{CudaSource, EmitError, emit};
pub use fusion::{FusionError, FusionRewrite, FusionRule, fuse};
pub use implementation::{
    AllGatherImplementation, AttributeSet, BroadcastImplementation, GemmImplementation,
    ImplementationDefinition, ImplementationId, ImplementationInstance, PointwiseImplementation,
    ReduceSumImplementation, all_gather_implementations, broadcast_implementations, fusion_rules,
    gemm_implementations, pointwise_implementations, reduce_sum_implementations,
};
pub use loop_ir::{LoopIrConfig, LoopIrError, lower_loop_ir};
pub use physical::{
    CommunicationKind, CommunicationOperation, ComputeOperation, Expression, IndexExpr, Loop,
    LoopDomain, LoopKind, Operation, OperationId, OperationPayload, PhysicalInvariantError,
    PhysicalPlan, PhysicalPlanBuilder, Statement, Storage, TensorBinding, ValueInstance,
    ValueInstanceId,
};
pub use platform::{CudaTargetCapability, TargetCapability};

#[cfg(test)]
mod tests;
