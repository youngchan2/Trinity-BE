/// CUDA target capabilities supported by the lowering backend.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CudaTargetCapability {
    Hopper,
}
