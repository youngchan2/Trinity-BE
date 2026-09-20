//! CUDA-specific implementation definitions and built-in candidate lists.

mod all_gather;
mod broadcast;
mod expression;
mod fusion;
mod gemm;
mod pointwise;
mod reduce_sum;

use super::{AllGatherImplementation, GemmImplementation};
use crate::platform::cuda::CudaTargetCapability;

pub(super) fn expression_instance(_target: CudaTargetCapability) -> super::ImplementationInstance {
    expression::expression_instance()
}
pub(super) fn fusion_rules(
    target: CudaTargetCapability,
) -> &'static [&'static dyn crate::FusionRule] {
    match target {
        CudaTargetCapability::Hopper => fusion::RULES,
    }
}

pub(super) fn gemm_implementations(
    target: CudaTargetCapability,
) -> &'static [&'static dyn GemmImplementation] {
    match target {
        CudaTargetCapability::Hopper => gemm::IMPLEMENTATIONS,
    }
}

pub(super) fn all_gather_implementations(
    target: CudaTargetCapability,
) -> &'static [&'static dyn AllGatherImplementation] {
    match target {
        CudaTargetCapability::Hopper => all_gather::IMPLEMENTATIONS,
    }
}

pub(super) fn pointwise_implementations(
    _: CudaTargetCapability,
) -> &'static [&'static dyn super::PointwiseImplementation] {
    pointwise::IMPLEMENTATIONS
}

pub(super) fn reduce_sum_implementations(
    _: CudaTargetCapability,
) -> &'static [&'static dyn super::ReduceSumImplementation] {
    reduce_sum::IMPLEMENTATIONS
}

pub(super) fn broadcast_implementations(
    _: CudaTargetCapability,
) -> &'static [&'static dyn super::BroadcastImplementation] {
    broadcast::IMPLEMENTATIONS
}
