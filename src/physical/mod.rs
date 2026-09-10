mod builder;
mod canonical;
mod error;
mod finalize;
mod loops;
pub(crate) mod normalize;
mod plan;
mod rewrite;
pub(crate) use loops::accumulation_rhs;
pub use loops::{Expression, IndexExpr, Loop, LoopDomain, LoopKind};
pub(crate) use rewrite::rewrite_statements;

pub use builder::PhysicalPlanBuilder;
pub use error::PhysicalInvariantError;
pub use plan::{
    CommunicationKind, CommunicationOperation, ComputeOperation, Operation, OperationId,
    OperationPayload, PhysicalPlan, Statement, Storage, TensorBinding, ValueInstance,
    ValueInstanceId,
};

#[cfg(test)]
pub(crate) use canonical::hash_plan;
