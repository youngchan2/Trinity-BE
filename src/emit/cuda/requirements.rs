use serde::Serialize;

use crate::platform::cuda::CudaTargetCapability;

#[derive(Debug, Clone, Serialize)]
pub struct CudaRequirements {
    pub target: CudaTargetCapability,
    /// CUDA architecture target required to compile the generated code (e.g. `sm_90a`).
    pub cuda_arch: &'static str,
    pub world_size: usize,
    pub buffers: Vec<BufferBindingRequirement>,
    /// Zero for streamed execution. Persistent execution needs one symmetric
    /// control allocation per rank, zeroed once before the first invocation.
    pub workspace_bytes: usize,
    pub workspace_alignment: usize,
    pub workspace_symmetric: bool,
    pub cooperative_launch: bool,
    pub shared_memory_bytes: usize,
    pub block_threads: usize,
    /// Persistent worker lower bound. Streamed execution reports 1 for compatibility;
    /// its launch ABI has no worker-count parameter.
    pub minimum_workers: usize,
    pub nvshmem: bool,
    pub nvls: bool,
}

/// Allocation and launch-binding requirements for a device buffer.
#[derive(Debug, Clone, Serialize)]
pub struct BufferBindingRequirement {
    /// Canonical value ID and index in the launch binding array.
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
