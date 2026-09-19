use serde::Serialize;

use crate::PhysicalPlan;
use crate::emit::BufferBindings;
use crate::platform::{TargetCapability, cuda::CudaTargetCapability};

use super::{BufferBindingRequirement, EmitError, Execution, Resources};

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
    pub shared_memory_bytes: usize,
    pub block_threads: usize,
    pub nvshmem: bool,
    pub nvls: bool,
}

/// Check every value, including Shared/Register values with no launch binding.
pub(super) fn validate_values(plan: &PhysicalPlan) -> Result<(), EmitError> {
    for (_, value) in plan.value_instances() {
        let shape = value.shape();

        // Region stores origins and extents in three axes,
        // padding unused axes with extent one.
        if !(1..=3).contains(&shape.len()) {
            return Err(EmitError::Unsupported("supported ranks are 1, 2, 3".into()));
        }

        if shape.contains(&0) {
            return Err(EmitError::Unsupported("empty matrix".into()));
        }
    }

    Ok(())
}

/// Build buffer and launch requirements for the prepared CUDA program.
pub(super) fn build_cuda_requirements(
    plan: &PhysicalPlan,
    mut bindings: BufferBindings,
    resources: &Resources,
    execution: &Execution,
) -> Result<CudaRequirements, EmitError> {
    let TargetCapability::Cuda(target) = plan.target();
    let shared_memory_bytes = resources.shared_memory_bytes;
    let reserved_shared_memory = execution.reserved_shared_memory_bytes();

    let body_shared_memory_limit = target.max_shared_memory_per_cta() - reserved_shared_memory;
    if shared_memory_bytes > body_shared_memory_limit {
        return Err(EmitError::Unsupported(
            "body exceeds target CTA shared-memory capacity after execution reservations".into(),
        ));
    }

    apply_buffer_requirements(&mut bindings, resources)?;

    let nvshmem = plan.world_size() > 1;
    let workspace_alignment = execution.workspace_alignment();

    Ok(CudaRequirements {
        target,
        world_size: plan.world_size(),
        buffers: bindings.requirements,
        workspace_bytes: execution.workspace_bytes(),
        workspace_alignment,
        workspace_symmetric: nvshmem,
        cooperative_launch: plan.world_size() > 1,
        shared_memory_bytes,
        block_threads: 128,
        nvshmem,
        nvls: resources.nvls,
    })
}

fn apply_buffer_requirements(
    bindings: &mut BufferBindings,
    resources: &Resources,
) -> Result<(), EmitError> {
    for buffer in &mut bindings.requirements {
        buffer.alignment = 16;
    }
    for &value in &resources.symmetric_values {
        let slot = bindings.slot(value)?;
        bindings.requirements[slot].symmetric = true;
    }
    Ok(())
}
