//! Concrete CUDA implementation candidates and physical tensor program plans.
//!
//! Callers resolve logical semantics and tensor presentations, enumerate applicable
//! implementations, and assemble a graph with [`PhysicalPlanBuilder`]. Finalized
//! plans own their physical data and do not retain a compiler's source graph.

mod config;
mod dtype;
mod implementation;
mod physical;

pub mod analyzer;
pub mod triton;

pub use config::{LoweringConfig, TargetCapability};
pub use dtype::DType;
pub use implementation::{
    AllGatherImplementation, AttributeSet, GemmImplementation, ImplementationDefinition,
    ImplementationId, ImplementationInstance, all_gather_implementations, gemm_implementations,
};
pub use physical::{
    Action, ActionId, CommunicationKind, CommunicationOperation, ComputeOperation, Operation,
    OperationId, OperationPayload, PhysicalInvariantError, PhysicalPlan, PhysicalPlanBuilder,
    Storage, TensorBinding, ValueInstance, ValueInstanceId,
};

#[cfg(test)]
mod tests;
