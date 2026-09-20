use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};

mod schedule;
mod schedules;

pub use schedule::{OperationSchedule, ScheduleError};

mod tensor;
pub use tensor::{
    BroadcastImplementation, PointwiseImplementation, ReduceSumImplementation,
    broadcast_implementations, pointwise_implementations, reduce_sum_implementations,
};

use super::TargetCapability;
use crate::DType;

/// Built-in fusion rules are being rebuilt with the kernel provider pipeline.
pub fn fusion_rules(_target: TargetCapability) -> &'static [&'static dyn crate::FusionRule] {
    &[]
}

/// Enumerates concrete GEMM instances for an already resolved tensor presentation.
pub trait GemmImplementation: ImplementationDefinition {
    /// Match already scheduled operand tiles without replacing their schedule.
    fn enumerate_scheduled(
        &'static self,
        _dtypes: [DType; 3],
        _tiles: [&[usize]; 3],
    ) -> Vec<ImplementationInstance> {
        Vec::new()
    }

    /// Returns no instances for unsupported dtype, rank, or tile extents.
    ///
    /// Callers must supply semantically valid shapes. Implementations may assert
    /// that the K extents match and the output has shape [M, N].
    fn enumerate(
        &'static self,
        dtypes: [DType; 3],
        shapes: [&[usize]; 3],
    ) -> Vec<ImplementationInstance>;
}

/// Enumerates concrete AllGather instances for source and destination shapes.
pub trait AllGatherImplementation: ImplementationDefinition {
    /// The source and destination are rank-local shapes before and after gathering.
    fn enumerate(
        &'static self,
        dtype: DType,
        shapes: [&[usize]; 2],
        shard_axis: usize,
        world_size: usize,
    ) -> Vec<ImplementationInstance>;
}

/// Returns the target's built-in GEMM implementations in deterministic order.
pub fn gemm_implementations(
    target: TargetCapability,
) -> &'static [&'static dyn GemmImplementation] {
    match target {
        TargetCapability::Cuda(target) => schedules::gemm_implementations(target),
    }
}

/// Returns the target's built-in AllGather implementations in deterministic order.
pub fn all_gather_implementations(
    target: TargetCapability,
) -> &'static [&'static dyn AllGatherImplementation] {
    match target {
        TargetCapability::Cuda(target) => schedules::all_gather_implementations(target),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImplementationId(&'static str);

impl ImplementationId {
    pub const fn new(value: &'static str) -> Self {
        Self(value)
    }

    pub fn as_str(self) -> &'static str {
        self.0
    }
}

pub trait ImplementationDefinition: Sync {
    fn id(&self) -> ImplementationId;

    /// Normalize a builder operation without generating device code.
    fn schedule(
        &self,
        _plan: &crate::PhysicalPlan,
        _operation: crate::OperationId,
    ) -> Result<OperationSchedule, ScheduleError> {
        Err(ScheduleError::Unsupported(
            "implementation requires an explicit schedule".into(),
        ))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttributeSet(BTreeMap<&'static str, usize>);

impl AttributeSet {
    pub fn get(&self, name: &str) -> Option<usize> {
        self.0.get(name).copied()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&'static str, usize)> + '_ {
        self.0.iter().map(|(name, value)| (*name, *value))
    }

    pub(crate) fn new(entries: impl IntoIterator<Item = (&'static str, usize)>) -> Self {
        Self(entries.into_iter().collect())
    }
}

#[derive(Clone)]
pub struct ImplementationInstance {
    definition: &'static dyn ImplementationDefinition,
    attributes: AttributeSet,
}

impl ImplementationInstance {
    pub fn definition(&self) -> &'static dyn ImplementationDefinition {
        self.definition
    }

    pub fn id(&self) -> ImplementationId {
        self.definition.id()
    }

    pub fn attributes(&self) -> &AttributeSet {
        &self.attributes
    }

    pub(crate) fn new(
        definition: &'static dyn ImplementationDefinition,
        attributes: AttributeSet,
    ) -> Self {
        Self {
            definition,
            attributes,
        }
    }
}

impl fmt::Debug for ImplementationInstance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImplementationInstance")
            .field("id", &self.id())
            .field("attributes", &self.attributes)
            .finish()
    }
}

impl PartialEq for ImplementationInstance {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id() && self.attributes == other.attributes
    }
}

impl Eq for ImplementationInstance {}

impl PartialOrd for ImplementationInstance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ImplementationInstance {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.id(), &self.attributes).cmp(&(other.id(), &other.attributes))
    }
}

impl Hash for ImplementationInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
        self.attributes.hash(state);
    }
}

pub(crate) fn expression_instance(target: TargetCapability) -> ImplementationInstance {
    match target {
        TargetCapability::Cuda(target) => schedules::expression_instance(target),
    }
}
