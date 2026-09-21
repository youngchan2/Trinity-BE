//! Existing implementation enumeration and input schedule normalization only.
//! Device code and rendering are preserved under old/emit-rewrite.

mod broadcast;
mod expression;
mod gemm;
mod nvls;
mod peer;
mod pointwise;
mod reduce_sum;

use super::{AllGatherImplementation, GemmImplementation};
use crate::CudaTargetCapability;

static GEMM: &[&dyn GemmImplementation] = &[&gemm::HOPPER_WGMMA_BF16];
static ALL_GATHER: &[&dyn AllGatherImplementation] = &[
    &nvls::NVLS_ONE_SHOT_PUSH_NBI,
    &peer::PEER_PUSH,
    &peer::PEER_PULL,
];

pub(super) fn expression_instance(_: CudaTargetCapability) -> super::ImplementationInstance {
    expression::expression_instance()
}

pub(super) fn gemm_implementations(
    _: CudaTargetCapability,
) -> &'static [&'static dyn GemmImplementation] {
    GEMM
}

pub(super) fn all_gather_implementations(
    _: CudaTargetCapability,
) -> &'static [&'static dyn AllGatherImplementation] {
    ALL_GATHER
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

struct GatherContext {
    source_rows: usize,
    source_cols: usize,
    target_rows: usize,
    target_cols: usize,
    axis: usize,
    push: bool,
    chunk: usize,
}
