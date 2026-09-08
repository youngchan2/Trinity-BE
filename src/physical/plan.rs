use std::marker::PhantomData;

use crate::DType;
use crate::{ImplementationInstance, TargetCapability};

macro_rules! plan_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(usize);

        impl $name {
            pub fn index(self) -> usize {
                self.0
            }

            pub(crate) fn from_index(index: usize) -> Self {
                Self(index)
            }
        }

        impl<T> IdVec<$name, T> {
            pub(crate) fn push(&mut self, value: T) -> $name {
                let id = $name::from_index(self.values.len());
                self.values.push(value);
                id
            }

            pub(super) fn get(&self, id: $name) -> Option<&T> {
                self.values.get(id.index())
            }

            pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = ($name, &T)> {
                self.values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| ($name::from_index(index), value))
            }
        }
    };
}

plan_id!(ValueInstanceId);
plan_id!(OperationId);
plan_id!(ActionId);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct IdVec<I, T> {
    pub(super) values: Vec<T>,
    marker: PhantomData<fn(I) -> I>,
}

impl<I, T> IdVec<I, T> {
    pub(crate) fn new() -> Self {
        Self {
            values: Vec::new(),
            marker: PhantomData,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.values.len()
    }

    pub(super) fn from_values(values: Vec<T>) -> Self {
        Self {
            values,
            marker: PhantomData,
        }
    }
}

impl<I, T> Default for IdVec<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The physical storage class of a concrete tensor value.
///
/// ABI boundary values use [`Storage::External`]. Values passed between distinct
/// actions must reside in [`Storage::External`] or [`Storage::Global`], while
/// [`Storage::Shared`] and [`Storage::Register`] are local to one action.
/// A value retains its rank-local tensor shape after promotion; execution
/// concretization allocates storage for its live tile/panel, not that entire
/// shape, and preserves the value's dtype at each operation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Storage {
    /// Storage supplied by the caller through an input or output ABI binding.
    External,

    /// CUDA global memory used to materialize a non-boundary value.
    Global,

    /// CUDA thread-block shared memory local to an action.
    Shared,

    /// CUDA register storage local to an action.
    Register,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueInstance {
    pub(super) dtype: DType,
    pub(super) shape: Box<[usize]>,
    pub(super) storage: Storage,
}

impl ValueInstance {
    pub(crate) fn new(
        dtype: DType,
        shape: impl IntoIterator<Item = usize>,
        storage: Storage,
    ) -> Self {
        Self {
            dtype,
            shape: shape.into_iter().collect(),
            storage,
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn storage(&self) -> Storage {
        self.storage
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorBinding {
    pub(super) tensor: String,
    pub(super) value: ValueInstanceId,
}

impl TensorBinding {
    pub(crate) fn new(tensor: impl Into<String>, value: ValueInstanceId) -> Self {
        Self {
            tensor: tensor.into(),
            value,
        }
    }

    pub fn tensor(&self) -> &str {
        &self.tensor
    }

    pub fn value(&self) -> ValueInstanceId {
        self.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeOperation {
    pub(super) implementation: ImplementationInstance,
}

impl ComputeOperation {
    pub fn implementation(&self) -> &ImplementationInstance {
        &self.implementation
    }

    pub fn new(implementation: ImplementationInstance) -> Self {
        Self { implementation }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CommunicationKind {
    AllGather,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommunicationOperation {
    pub(super) kind: CommunicationKind,
    pub(super) implementation: ImplementationInstance,
}

impl CommunicationOperation {
    pub fn kind(&self) -> CommunicationKind {
        self.kind
    }

    pub fn implementation(&self) -> &ImplementationInstance {
        &self.implementation
    }

    pub fn new(kind: CommunicationKind, implementation: ImplementationInstance) -> Self {
        Self {
            kind,
            implementation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperationPayload {
    Compute(ComputeOperation),
    Communication(CommunicationOperation),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Operation {
    pub(super) inputs: Vec<ValueInstanceId>,
    pub(super) outputs: Vec<ValueInstanceId>,
    pub(super) payload: OperationPayload,
}

impl Operation {
    pub(crate) fn new(
        inputs: impl IntoIterator<Item = ValueInstanceId>,
        outputs: impl IntoIterator<Item = ValueInstanceId>,
        payload: OperationPayload,
    ) -> Self {
        Self {
            inputs: inputs.into_iter().collect(),
            outputs: outputs.into_iter().collect(),
            payload,
        }
    }

    pub fn inputs(&self) -> &[ValueInstanceId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[ValueInstanceId] {
        &self.outputs
    }

    pub fn payload(&self) -> &OperationPayload {
        &self.payload
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Action {
    pub(super) operations: Vec<OperationId>,
    pub(super) inputs: Vec<ValueInstanceId>,
    pub(super) outputs: Vec<ValueInstanceId>,
}

impl Action {
    pub fn operations(&self) -> &[OperationId] {
        &self.operations
    }

    pub fn inputs(&self) -> &[ValueInstanceId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[ValueInstanceId] {
        &self.outputs
    }
}

/// A validated, canonical physical graph that owns its tensor and ABI metadata.
///
/// Plans do not retain a logical source graph or depend on a compiler's lifetime.
#[derive(Clone)]
pub struct PhysicalPlan {
    pub(super) target: TargetCapability,
    pub(super) world_size: usize,
    pub(super) inputs: Box<[TensorBinding]>,
    pub(super) value_instances: IdVec<ValueInstanceId, ValueInstance>,
    pub(super) operations: IdVec<OperationId, Operation>,
    pub(super) actions: IdVec<ActionId, Action>,
    pub(super) output: TensorBinding,
    pub(super) hash: u64,
}

impl PhysicalPlan {
    pub fn target(&self) -> TargetCapability {
        self.target
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn inputs(&self) -> &[TensorBinding] {
        &self.inputs
    }

    pub fn output(&self) -> &TensorBinding {
        &self.output
    }

    pub fn value_instance(&self, id: ValueInstanceId) -> Option<&ValueInstance> {
        self.value_instances.get(id)
    }

    pub fn value_instances(
        &self,
    ) -> impl ExactSizeIterator<Item = (ValueInstanceId, &ValueInstance)> {
        self.value_instances.iter()
    }

    pub fn operation(&self, id: OperationId) -> Option<&Operation> {
        self.operations.get(id)
    }

    pub fn operations(&self) -> impl ExactSizeIterator<Item = (OperationId, &Operation)> {
        self.operations.iter()
    }

    pub fn action(&self, id: ActionId) -> Option<&Action> {
        self.actions.get(id)
    }

    pub fn actions(&self) -> impl ExactSizeIterator<Item = (ActionId, &Action)> {
        self.actions.iter()
    }

    /// Returns a process-local compilation cache key, not a persistent identity.
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Compares canonical physical bodies without relying on a hash match.
    pub fn same_body(&self, other: &Self) -> bool {
        self.target == other.target
            && self.world_size == other.world_size
            && self.inputs == other.inputs
            && self.value_instances == other.value_instances
            && self.operations == other.operations
            && self.actions == other.actions
            && self.output == other.output
    }
}
