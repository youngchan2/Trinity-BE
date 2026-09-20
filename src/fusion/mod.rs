//! Ordered task-set fusion and backend binding/storage proposals.

use thiserror::Error;

use crate::{
    OperationId, PhysicalInvariantError, PhysicalPlan, Statement, Storage, ValueInstanceId,
};

/// Proposes operation and storage rewrites for producer and consumer statements.
/// Proposals preserve the original operation order and sequential scopes.
pub trait FusionRule: Sync {
    /// Returns separate rewrites for storage alternatives (e.g. Shared/Register).
    /// An empty `Ok` means unsupported; `Err` aborts [`fuse`].
    ///
    /// Check backend compatibility of all merged operations and their storage,
    /// including earlier fusions. Rewrite IDs refer to `plan` and may change
    /// when the resulting plan is finalized.
    fn apply(
        &self,
        plan: &PhysicalPlan,
        producer: &Statement,
        consumer: &Statement,
    ) -> Result<Vec<FusionRewrite>, FusionError>;
}

/// A fusion proposal listing merged operations and storage promotions.
///
/// Storage updates must be unique Global-to-Shared/Register promotions. ABI
/// values and values used outside the merged operations cannot be promoted.
/// The common rewriter constructs the replacement Statement tree and CUDA body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionRewrite {
    operations: Vec<OperationId>,
    storage_updates: Vec<(ValueInstanceId, Storage)>,
}

impl FusionRewrite {
    pub fn new(
        operations: impl IntoIterator<Item = OperationId>,
        storage_updates: impl IntoIterator<Item = (ValueInstanceId, Storage)>,
    ) -> Self {
        Self {
            operations: operations.into_iter().collect(),
            storage_updates: storage_updates.into_iter().collect(),
        }
    }

    pub fn operations(&self) -> &[OperationId] {
        &self.operations
    }

    pub fn storage_updates(&self) -> &[(ValueInstanceId, Storage)] {
        &self.storage_updates
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FusionError {
    #[error("fusion validation is being rebuilt with the kernel provider pipeline")]
    Unavailable,
    #[error("downstream fusion is not connected to the structured Statement model")]
    UnsupportedStatementModel,
    #[error("invalid fusion rewrite: {reason}")]
    InvalidRewrite { reason: &'static str },

    #[error(transparent)]
    InvalidPlan(#[from] PhysicalInvariantError),
}

/// Empty rules preserve the original plan. Executable fusion validation is
/// unavailable until the new provider and memory analysis contracts are ready.
pub fn fuse(
    plan: &PhysicalPlan,
    rules: &[&dyn FusionRule],
) -> Result<Vec<PhysicalPlan>, FusionError> {
    if rules.is_empty() {
        Ok(vec![plan.clone()])
    } else {
        Err(FusionError::Unavailable)
    }
}
