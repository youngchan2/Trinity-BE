//! Target selection shared by planning, implementation enumeration and emission.

pub mod cuda;

pub use cuda::CudaTargetCapability;

/// Platform selection and its concrete target capability.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TargetCapability {
    Cuda(CudaTargetCapability),
}
