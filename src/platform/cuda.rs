/// CUDA target capabilities supported by the lowering backend.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CudaTargetCapability {
    Hopper,
}
