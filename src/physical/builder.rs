use super::finalize::finalize;
use super::plan::IdVec;
use super::{
    ActionId, Operation, OperationId, OperationPayload, PhysicalInvariantError, PhysicalPlan,
    Storage, TensorBinding, ValueInstance, ValueInstanceId,
};
use crate::{DType, TargetCapability};

/// Builds one physical implementation branch before validation and canonicalization.
///
/// Callers supply concrete tensor presentations and applicable implementation
/// instances. Finalization checks physical graph invariants and derives Action
/// boundaries; it does not repeat logical or implementation applicability checks.
/// IDs belong to this builder (or its clones) and may be remapped by finalization.
#[derive(Clone)]
pub struct PhysicalPlanBuilder {
    pub(super) target: TargetCapability,
    pub(super) world_size: usize,
    pub(super) inputs: Vec<TensorBinding>,
    pub(super) values: IdVec<ValueInstanceId, ValueInstance>,
    pub(super) operations: IdVec<OperationId, Operation>,
    pub(super) action_operations: IdVec<ActionId, Vec<OperationId>>,
}

impl PhysicalPlanBuilder {
    pub fn new(target: TargetCapability, world_size: usize) -> Self {
        Self {
            target,
            world_size,
            inputs: Vec::new(),
            values: IdVec::new(),
            operations: IdVec::new(),
            action_operations: IdVec::new(),
        }
    }

    pub fn add_value(
        &mut self,
        dtype: DType,
        shape: impl IntoIterator<Item = usize>,
        storage: Storage,
    ) -> ValueInstanceId {
        self.values.push(ValueInstance::new(dtype, shape, storage))
    }

    pub fn bind_input(&mut self, tensor: impl Into<String>, value: ValueInstanceId) {
        self.inputs.push(TensorBinding::new(tensor, value));
    }

    pub fn add_operation(
        &mut self,
        inputs: impl IntoIterator<Item = ValueInstanceId>,
        outputs: impl IntoIterator<Item = ValueInstanceId>,
        payload: OperationPayload,
    ) -> OperationId {
        self.operations
            .push(Operation::new(inputs, outputs, payload))
    }

    pub fn add_action(&mut self, operations: impl IntoIterator<Item = OperationId>) -> ActionId {
        self.action_operations
            .push(operations.into_iter().collect())
    }

    /// Validates and canonicalizes the graph, consuming all construction state.
    ///
    /// Query the returned plan's bindings and graph for canonical IDs rather than
    /// reusing IDs obtained before finalization.
    pub fn finalize(
        self,
        output_tensor: impl Into<String>,
        output_value: ValueInstanceId,
    ) -> Result<PhysicalPlan, PhysicalInvariantError> {
        finalize(self, TensorBinding::new(output_tensor, output_value))
    }
}
