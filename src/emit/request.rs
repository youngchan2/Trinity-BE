//! Shared operation boundary for independently launched kernels.
use super::provider::{KernelContext, ProviderError};
use crate::{
    AccessIndex, DType, Expression, OperationId, PhysicalPlan, Storage, TargetCapability,
    ValueInstanceId,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorArgument {
    pub(crate) value: ValueInstanceId,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
}
impl TensorArgument {
    pub fn value(&self) -> ValueInstanceId {
        self.value
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Exact operation and memory boundary shared by Triton and external candidates.
/// This initial adapter represents full-tensor operations outside enclosing loops.
#[derive(Debug, Clone)]
pub struct KernelRequest {
    pub(crate) operation: OperationId,
    pub(crate) target: TargetCapability,
    pub(crate) expression: Expression,
    pub(crate) tensors: BTreeMap<ValueInstanceId, TensorArgument>,
    pub(crate) inputs: Vec<ValueInstanceId>,
    pub(crate) output: ValueInstanceId,
}
impl KernelRequest {
    pub fn target(&self) -> TargetCapability {
        self.target
    }
    pub fn operation(&self) -> OperationId {
        self.operation
    }
    pub fn expression(&self) -> &Expression {
        &self.expression
    }
    pub fn inputs(&self) -> &[ValueInstanceId] {
        &self.inputs
    }
    pub fn output(&self) -> ValueInstanceId {
        self.output
    }
    pub fn tensors(&self) -> &BTreeMap<ValueInstanceId, TensorArgument> {
        &self.tensors
    }

    pub(super) fn from_context(context: &KernelContext<'_, '_>) -> Result<Self, ProviderError> {
        if !context.loops.is_empty() {
            return Err(ProviderError::Unsupported("independent host-call adapter requires operations outside loops; loop/view coverage is not implemented".into()));
        }
        Self::new(context.prepared.plan, context.operation)
    }

    pub(super) fn new(plan: &PhysicalPlan, operation: OperationId) -> Result<Self, ProviderError> {
        let unsupported = |s: &str| ProviderError::Unsupported(s.into());
        if plan.world_size() != 1 {
            return Err(unsupported(
                "independent host calls require single-GPU execution",
            ));
        }
        let op = plan
            .operation(operation)
            .ok_or_else(|| ProviderError::Failed("missing operation".into()))?;
        let Expression::Store { destination, .. } = op.expression() else {
            return Err(unsupported(
                "independent computation candidates require a store expression",
            ));
        };
        if op.inflows().contains(&destination.value) {
            return Err(unsupported(
                "independent candidate adapter does not yet support an in-place output",
            ));
        }
        let mut tensors = BTreeMap::new();
        for access in op.expression().accesses() {
            let v = plan
                .value_instance(access.value)
                .ok_or_else(|| ProviderError::Failed("missing value".into()))?;
            if !matches!(v.storage(), Storage::External | Storage::Global) {
                return Err(unsupported(
                    "independent candidates require external/global memory, not CTA-local storage",
                ));
            }
            if v.shape().is_empty()
                || v.shape().contains(&0)
                || access
                    .indices
                    .iter()
                    .any(|i| !matches!(i, AccessIndex::FullTile))
            {
                return Err(unsupported(
                    "independent candidate adapter requires positive full-tensor accesses",
                ));
            }
            tensors.insert(
                access.value,
                TensorArgument {
                    value: access.value,
                    dtype: v.dtype(),
                    shape: v.shape().to_vec(),
                },
            );
        }
        Ok(Self {
            operation,
            target: plan.target(),
            expression: op.expression().clone(),
            tensors,
            inputs: op.inflows().to_vec(),
            output: destination.value,
        })
    }
}
