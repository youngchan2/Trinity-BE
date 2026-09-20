//! Execution model for kernel collection and combination.

use crate::{PhysicalPlan, TargetCapability};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExecutionModel {
    CudaStreamed,
    CudaPersistent,
}

/// Sets the execution model.
pub(super) fn plan_execution(plan: &PhysicalPlan) -> ExecutionModel {
    match (plan.target(), plan.world_size()) {
        (TargetCapability::Cuda(_), 1) => ExecutionModel::CudaStreamed,
        (TargetCapability::Cuda(_), _) => ExecutionModel::CudaPersistent,
    }
}
