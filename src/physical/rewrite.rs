use super::plan::IdVec;
use super::{
    ActionId, OperationId, PhysicalInvariantError, PhysicalPlan, PhysicalPlanBuilder, Storage,
    ValueInstanceId,
};

/// The fusion caller validates the proposal before this transactional rebuild.
pub(crate) fn rewrite_actions(
    plan: &PhysicalPlan,
    producer: ActionId,
    consumer: ActionId,
    operations: &[OperationId],
    storage_updates: &[(ValueInstanceId, Storage)],
) -> Result<PhysicalPlan, PhysicalInvariantError> {
    let mut builder = PhysicalPlanBuilder {
        target: plan.target,
        world_size: plan.world_size,
        inputs: plan.inputs.to_vec(),
        values: plan.value_instances.clone(),
        operations: plan.operations.clone(),
        action_operations: IdVec::new(),
    };
    for (id, action) in plan.actions() {
        if id == producer {
            builder.add_action(operations.iter().copied());
        } else if id != consumer {
            builder.add_action(action.operations().iter().copied());
        }
    }
    for (id, storage) in storage_updates {
        builder.values.values[id.index()].storage = *storage;
    }
    builder.finalize(plan.output.tensor.clone(), plan.output.value)
}
