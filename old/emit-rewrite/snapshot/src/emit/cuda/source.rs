use super::{CudaRequirements, Execution};

/// Target-specific source and allocation requirements. CUDA pointers are supplied
/// by the caller; emission never allocates device memory.
#[derive(Debug, Clone)]
pub struct CudaSource {
    pub(super) code: String,
    pub(super) requirements: CudaRequirements,
    pub(super) execution: Execution,
    pub(super) bodies: Vec<super::Body>,
}

impl CudaSource {
    pub fn bodies(&self) -> &[super::Body] {
        &self.bodies
    }

    pub fn code(&self) -> &str {
        &self.code
    }
    pub fn requirements(&self) -> &CudaRequirements {
        &self.requirements
    }
    /// Inspect domain grids and launches for Streamed, or tile tasks and
    /// readiness/scheduler metadata for Persistent.
    pub fn execution(&self) -> &Execution {
        &self.execution
    }
}
