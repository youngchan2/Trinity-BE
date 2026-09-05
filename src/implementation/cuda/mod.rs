//! CUDA-specific implementation definitions and built-in candidate lists.

mod hopper_wgmma;
mod nvls;

use super::{AllGatherImplementation, GemmImplementation};
use hopper_wgmma::HOPPER_WGMMA_BF16;
use nvls::NVLS_ONE_SHOT_PUSH_NBI;

pub(super) static HOPPER_GEMM_IMPLEMENTATIONS: &[&dyn GemmImplementation] = &[&HOPPER_WGMMA_BF16];
pub(super) static HOPPER_ALL_GATHER_IMPLEMENTATIONS: &[&dyn AllGatherImplementation] =
    &[&NVLS_ONE_SHOT_PUSH_NBI];
