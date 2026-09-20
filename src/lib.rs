//! Triton fallback source generation and explicit CUDA tensor program plans.
//!
//! Callers resolve logical semantics and tensor presentations, enumerate applicable
//! implementations, and assemble a graph with [`PhysicalPlanBuilder`]. Finalized
//! plans own their physical data and do not retain a compiler's source graph.

pub mod analysis;
pub mod compile;
mod config;
mod dtype;
pub mod emit;
mod fusion;
mod implementation;
mod plan;
pub mod platform;
mod python;
pub mod triton;

pub use compile::{
    CompileConfig, CompileDiagnostics, CompileError, CudaArtifact, CudaSource, compile,
    compile_with_config,
};
pub use config::LoweringConfig;
pub use dtype::DType;
pub use emit::{EmitError, emit};
pub use fusion::{FusionError, FusionRewrite, FusionRule, fuse};
pub use implementation::{
    AllGatherImplementation, AttributeSet, BroadcastImplementation, GemmImplementation,
    ImplementationDefinition, ImplementationId, ImplementationInstance, PointwiseImplementation,
    ReduceSumImplementation, all_gather_implementations, broadcast_implementations, fusion_rules,
    gemm_implementations, pointwise_implementations, reduce_sum_implementations,
};
pub use plan::{
    AccessIndex, Constant, Expression, IndexExpr, IrConfig, IrError, Loop, LoopDomain, LoopKind,
    Operation, OperationId, PhysicalInvariantError, PhysicalPlan, PhysicalPlanBuilder, Statement,
    Storage, TensorAccess, TensorBinding, ValueInstance, ValueInstanceId, lower_ir,
};
pub use platform::{AsStr, CudaTargetCapability, TargetCapability};

#[cfg(test)]
mod tests;
