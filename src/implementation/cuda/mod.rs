//! CUDA-specific implementation definitions and built-in candidate lists.

mod emission;
mod fusion;
mod hopper_wgmma;
mod nvls;
mod peer;

static HOPPER_FUSION_RULES: &[&dyn crate::FusionRule] = fusion::RULES;

use super::{AllGatherImplementation, GemmImplementation};
use crate::platform::cuda::CudaTargetCapability;
use hopper_wgmma::HOPPER_WGMMA_BF16;
use nvls::NVLS_ONE_SHOT_PUSH_NBI;
use peer::{PEER_PULL, PEER_PUSH};

static HOPPER_GEMM_IMPLEMENTATIONS: &[&dyn GemmImplementation] = &[&HOPPER_WGMMA_BF16];
static HOPPER_ALL_GATHER_IMPLEMENTATIONS: &[&dyn AllGatherImplementation] =
    &[&NVLS_ONE_SHOT_PUSH_NBI, &PEER_PUSH, &PEER_PULL];

pub(super) fn fusion_rules(
    target: CudaTargetCapability,
) -> &'static [&'static dyn crate::FusionRule] {
    match target {
        CudaTargetCapability::Hopper => HOPPER_FUSION_RULES,
    }
}

pub(super) fn gemm_implementations(
    target: CudaTargetCapability,
) -> &'static [&'static dyn GemmImplementation] {
    match target {
        CudaTargetCapability::Hopper => HOPPER_GEMM_IMPLEMENTATIONS,
    }
}

pub(super) fn all_gather_implementations(
    target: CudaTargetCapability,
) -> &'static [&'static dyn AllGatherImplementation] {
    match target {
        CudaTargetCapability::Hopper => HOPPER_ALL_GATHER_IMPLEMENTATIONS,
    }
}
