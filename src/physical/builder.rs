use super::finalize::finalize;
use super::plan::IdVec;
use super::{
    Operation, OperationId, OperationPayload, PhysicalInvariantError, PhysicalPlan, Storage,
    TensorBinding, ValueInstance, ValueInstanceId,
};
use crate::{DType, TargetCapability};

/// Builds one physical implementation branch before validation and canonicalization.
///
/// Callers supply concrete tensor presentations and applicable implementation
/// instances and an ordered Statement program. Finalization first normalizes selected implementations into explicit loops and
/// tile expressions, then checks common physical invariants; it does not
/// repeat upstream scheduling/fusion legality analysis.
/// IDs belong to this builder (or its clones) and may be remapped by finalization.
#[derive(Clone)]
pub struct PhysicalPlanBuilder {
    pub(super) target: TargetCapability,
    pub(super) world_size: usize,
    pub(super) inputs: Vec<TensorBinding>,
    pub(super) values: IdVec<ValueInstanceId, ValueInstance>,
    pub(super) operations: IdVec<OperationId, Operation>,
    pub(super) statements: Vec<super::Statement>,
}

impl PhysicalPlanBuilder {
    pub fn new(target: TargetCapability, world_size: usize) -> Self {
        Self {
            target,
            world_size,
            inputs: Vec::new(),
            values: IdVec::new(),
            operations: IdVec::new(),
            statements: Vec::new(),
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

    /// Appends a top-level statement in program order.
    pub fn add_statement(&mut self, statement: super::Statement) {
        self.statements.push(statement);
    }

    pub fn add_named_value(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        shape: impl IntoIterator<Item = usize>,
        storage: Storage,
    ) -> ValueInstanceId {
        let id = self.add_value(dtype, shape, storage);
        self.values.values[id.index()].name = Some(name.into());

        id
    }

    pub fn add_expression(
        &mut self,
        inputs: impl IntoIterator<Item = ValueInstanceId>,
        outputs: impl IntoIterator<Item = ValueInstanceId>,
        expression: super::Expression,
        implementation: crate::ImplementationInstance,
    ) -> OperationId {
        let id = self.add_operation(
            inputs,
            outputs,
            OperationPayload::Compute(super::ComputeOperation::new(implementation)),
        );
        self.operations.values[id.index()].expression = Some(expression);

        id
    }

    /// Normalizes the scheduled program, validates it, and canonicalizes IDs.
    /// Unsupported builder geometry may fail here before CUDA emission.
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
