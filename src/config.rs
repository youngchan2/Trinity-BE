use crate::platform::{CudaTargetCapability, TargetCapability};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweringConfig {
    target: TargetCapability,
}

impl LoweringConfig {
    pub fn new(target: TargetCapability) -> Self {
        Self { target }
    }

    pub fn target(&self) -> TargetCapability {
        self.target
    }
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self::new(TargetCapability::Cuda(CudaTargetCapability::Hopper))
    }
}
