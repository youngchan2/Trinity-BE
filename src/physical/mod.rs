mod builder;
mod canonical;
mod error;
mod finalize;
mod plan;

pub use builder::PhysicalPlanBuilder;
pub use error::PhysicalInvariantError;
pub use plan::{
    Action, ActionId, CommunicationKind, CommunicationOperation, ComputeOperation, Operation,
    OperationId, OperationPayload, PhysicalPlan, Storage, TensorBinding, ValueInstance,
    ValueInstanceId,
};

#[cfg(test)]
pub(crate) use canonical::hash_plan;
