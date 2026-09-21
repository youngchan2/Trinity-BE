//! Source and allocation metadata consumed by the existing CUDA compiler.
//! Kernel placement is internal to Emit; the host ABI receives allocation metadata.

use crate::CudaTargetCapability;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BufferBindingRequirement {
    /// Dense index in the launch binding array; local phase values have no slot.
    pub value: usize,
    pub input_names: Vec<String>,
    pub output_name: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: crate::DType,
    pub strides: Vec<usize>,
    pub bytes: usize,
    pub alignment: usize,
    /// Whether this value is a program input/output rather than a Global intermediate.
    pub external: bool,
    pub symmetric: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CudaRequirements {
    pub target: CudaTargetCapability,
    pub world_size: usize,
    pub buffers: Vec<BufferBindingRequirement>,
    /// Zero for streamed execution. Persistent execution needs one symmetric
    /// control allocation per rank, zeroed once before the first invocation.
    pub workspace_bytes: usize,
    pub workspace_alignment: usize,
    pub workspace_symmetric: bool,
    pub cooperative_launch: bool,
    /// Maximum across kernels. Each Streamed launch uses its own requirement.
    pub shared_memory_bytes: usize,
    /// Maximum across kernels (128 for a launch-free identity program).
    pub block_threads: usize,
    pub nvshmem: bool,
    pub nvls: bool,
}

#[derive(Debug, Clone)]
pub struct CudaSource {
    pub(crate) code: String,
    pub(crate) requirements: CudaRequirements,
}

impl CudaSource {
    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn requirements(&self) -> &CudaRequirements {
        &self.requirements
    }
}
