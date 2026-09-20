//! Enumeration contracts for contiguous tensor operations.
use super::{ImplementationDefinition, ImplementationInstance};
use crate::{DType, TargetCapability};

pub trait PointwiseImplementation: ImplementationDefinition {
    fn input_count(&self) -> usize;
    fn requires_scalar(&self) -> bool;

    /// Operands are ordered inputs followed by one output. All shapes must
    /// match; scalar division takes its immediate RHS separately. Computation
    /// is FP32, with conversion only on load and at the declared output.
    fn enumerate(
        &'static self,
        dtypes: &[DType],
        shapes: &[&[usize]],
        scalar: Option<f32>,
    ) -> Vec<ImplementationInstance>;
}

pub trait ReduceSumImplementation: ImplementationDefinition {
    /// Axis 1 of [M, N] produces [M]. Accumulation and output are FP32.
    fn enumerate(
        &'static self,
        dtypes: [DType; 2],
        shapes: [&[usize]; 2],
        axis: usize,
    ) -> Vec<ImplementationInstance>;
}

pub trait BroadcastImplementation: ImplementationDefinition {
    /// Insert axis 1: [M] -> [M, N]. This materializes a dtype-preserving copy.
    fn enumerate(
        &'static self,
        dtype: DType,
        shapes: [&[usize]; 2],
        axis: usize,
    ) -> Vec<ImplementationInstance>;
}

/// Arithmetic, square/root, sigmoid and ReLU definitions. ReLU maps negative
/// inputs to positive zero and preserves NaN, positive infinity and signed zero.
pub fn pointwise_implementations(
    target: TargetCapability,
) -> &'static [&'static dyn PointwiseImplementation] {
    match target {
        TargetCapability::Cuda(target) => super::definitions::pointwise_implementations(target),
    }
}

pub fn reduce_sum_implementations(
    target: TargetCapability,
) -> &'static [&'static dyn ReduceSumImplementation] {
    match target {
        TargetCapability::Cuda(target) => super::definitions::reduce_sum_implementations(target),
    }
}

pub fn broadcast_implementations(
    target: TargetCapability,
) -> &'static [&'static dyn BroadcastImplementation] {
    match target {
        TargetCapability::Cuda(target) => super::definitions::broadcast_implementations(target),
    }
}
