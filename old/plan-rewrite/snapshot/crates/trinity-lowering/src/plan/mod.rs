pub(crate) mod address;
mod builder;
mod canonical;
mod error;
mod finalize;
mod loops;
pub(crate) mod normalize;
mod types;
pub(crate) use loops::accumulation_rhs;
pub use loops::{Expression, IndexExpr, Loop, LoopDomain, LoopKind};

pub use builder::PhysicalPlanBuilder;
pub use error::PhysicalInvariantError;
pub use types::{
    CommunicationKind, CommunicationOperation, ComputeOperation, Operation, OperationId,
    OperationPayload, PhysicalPlan, Statement, Storage, TensorBinding, ValueInstance,
    ValueInstanceId,
};

#[cfg(test)]
pub(crate) use canonical::hash_plan;
