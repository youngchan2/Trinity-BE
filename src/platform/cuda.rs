use std::fmt;

use super::AsStr;

/// CUDA target capabilities supported by the lowering backend.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CudaTargetCapability {
    Hopper,
    Sm89,
    Sm120,
}

impl fmt::Display for CudaTargetCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hopper => f.write_str("hopper"),
            Self::Sm89 => f.write_str("sm89"),
            Self::Sm120 => f.write_str("sm120"),
        }
    }
}

impl AsStr for CudaTargetCapability {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Hopper => "sm_90a",
            Self::Sm89 => "sm_89",
            Self::Sm120 => "sm_120",
        }
    }
}

impl CudaTargetCapability {
    /// Maximum user-addressable shared memory per CTA, including static and
    /// dynamic allocations. Excludes the CUDA runtime's per-block reservation.
    pub const fn max_shared_memory_per_cta(self) -> usize {
        match self {
            // Hopper provides 228 KiB per SM; CUDA reserves 1 KiB per block.
            Self::Hopper => 227 * 1024,
            // Ada and RTX Blackwell reserve 1 KiB of their 100 KiB per block.
            Self::Sm89 | Self::Sm120 => 99 * 1024,
        }
    }
}
