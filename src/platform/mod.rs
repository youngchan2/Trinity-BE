//! Target selection shared by planning, implementation enumeration and emission.

pub mod cuda;

pub use cuda::CudaTargetCapability;

/// A static compiler identifier, distinct from a target's display name.
pub trait AsStr {
    fn as_str(&self) -> &'static str;
}

/// Platform selection and its concrete target capability.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TargetCapability {
    Cuda(CudaTargetCapability),
}
