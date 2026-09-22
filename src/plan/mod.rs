mod bindings;
mod builder;
mod error;
mod expression;
mod ir;
mod normalize;
mod scheduled;
mod statement;
pub(crate) use expression::accumulation_rhs;
pub use expression::{AccessIndex, Constant, Expression, TensorAccess, TileWidth, ValueOp};
pub use scheduled::ScheduledConfig;
pub use statement::{IndexExpr, Loop, LoopDomain, LoopKind, Statement};

pub use builder::PhysicalPlanBuilder;
pub use error::PhysicalInvariantError;
pub(crate) use ir::parse_index;
pub use ir::{IrConfig, IrError, lower_ir};

#[cfg(test)]
pub(crate) use normalize::hash_plan;

use std::marker::PhantomData;

use crate::{DType, TargetCapability};

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

            fn get(&self, id: $name) -> Option<&T> {
                self.values.get(id.index())
            }

            fn iter(&self) -> impl ExactSizeIterator<Item = ($name, &T)> {
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct IdVec<I, T> {
    values: Vec<T>,
    marker: PhantomData<I>,
}

impl<I, T> IdVec<I, T> {
    pub(crate) fn new() -> Self {
        Self {
            values: Vec::new(),
            marker: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn from_values(values: Vec<T>) -> Self {
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
/// ABI boundary values use [`Storage::External`]. Values passed between separately
/// scheduled tasks must reside in [`Storage::External`] or [`Storage::Global`],
/// while [`Storage::Shared`] and [`Storage::Register`] are local to a CTA body.
/// A value retains its rank-local tensor shape after promotion; execution
/// concretization allocates storage for its live tile/panel, not that entire
/// shape, and preserves the value's dtype at each operation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Storage {
    /// Storage supplied by the caller through an input or output ABI binding.
    External,

    /// CUDA global memory used to materialize a non-boundary value.
    Global,

    /// CUDA thread-block shared memory local to a CTA body.
    Shared,

    /// CUDA register storage local to a CTA body.
    Register,
}

/// A tensor read or written by operations in a physical plan.
///
/// Input, output, and intermediate tensors share this representation.
/// Their roles are determined by tensor bindings and operation operands.
/// Each value records its dtype, rank-local shape, and storage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueInstance {
    dtype: DType,
    shape: Box<[usize]>,
    storage: Storage,
    name: Option<String>,
    dimensions: Option<Box<[IndexExpr]>>,
    dtype_explicit: bool,
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
            name: None,
            dimensions: None,
            dtype_explicit: true,
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

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub fn dimensions(&self) -> Option<&[IndexExpr]> {
        self.dimensions.as_deref()
    }
    pub fn dtype_is_explicit(&self) -> bool {
        self.dtype_explicit
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorBinding {
    tensor: String,
    value: ValueInstanceId,
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

/// A memory-state change and the reads and computation needed to perform it.
/// Implementation selection belongs to Emit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Operation {
    inflows: Vec<ValueInstanceId>,
    outflows: Vec<ValueInstanceId>,
    expression: Expression,
    zero_init: Vec<ValueInstanceId>,
}

impl Operation {
    pub(crate) fn new(
        inflows: impl IntoIterator<Item = ValueInstanceId>,
        outflows: impl IntoIterator<Item = ValueInstanceId>,
        expression: Expression,
    ) -> Self {
        Self {
            inflows: inflows.into_iter().collect(),
            outflows: outflows.into_iter().collect(),
            expression,
            zero_init: Vec::new(),
        }
    }

    pub fn inflows(&self) -> &[ValueInstanceId] {
        &self.inflows
    }

    pub fn outflows(&self) -> &[ValueInstanceId] {
        &self.outflows
    }

    pub fn expression(&self) -> &Expression {
        &self.expression
    }
    pub fn zero_init(&self) -> &[ValueInstanceId] {
        &self.zero_init
    }
}

/// A validated, canonical physical program that owns its tensor and ABI metadata.
///
/// Plans do not retain a logical source graph or depend on a compiler's lifetime.
#[derive(Clone)]
pub struct PhysicalPlan {
    target: TargetCapability,
    world_size: usize,
    inputs: Box<[TensorBinding]>,
    value_instances: IdVec<ValueInstanceId, ValueInstance>,
    operations: IdVec<OperationId, Operation>,
    statements: Vec<Statement>,
    outputs: Box<[TensorBinding]>,
    mutable_inputs: std::collections::BTreeSet<ValueInstanceId>,
    bindings: std::collections::BTreeMap<String, i64>,
    hash: u64,
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
        self.outputs.first().expect("this path requires an output")
    }
    pub fn outputs(&self) -> &[TensorBinding] {
        &self.outputs
    }
    pub fn mutable_inputs(&self) -> &std::collections::BTreeSet<ValueInstanceId> {
        &self.mutable_inputs
    }
    pub fn bindings(&self) -> &std::collections::BTreeMap<String, i64> {
        &self.bindings
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

    /// Returns the directly owned top-level statements in program order.
    pub fn statements(&self) -> &[Statement] {
        &self.statements
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
            && self.statements == other.statements
            && self.outputs == other.outputs
            && self.mutable_inputs == other.mutable_inputs
            && self.bindings == other.bindings
    }
}
