//! CUDA source generation.
//!
//! Backends describe concrete reads/writes and render a CTA body. This module
//! resolves those regions into execution metadata. Streamed and persistent
//! runtimes have separate templates and launch ABIs; CTA operations are shared
//! through binding, tile-coordinate, and input-readiness hooks.

mod backend;
mod error;
mod graph;
mod persistent;
mod render;
mod requirements;
mod source;
mod streamed;

#[cfg(test)]
mod tests;

use serde::Serialize;

use crate::platform::{CudaTargetCapability, TargetCapability};
use crate::{PhysicalPlan, Storage};

pub use backend::{CudaImplementation, OperationEmission, Region, Work};
pub use error::EmitError;
pub use graph::{Dependency, Execution, Stage, Task};
pub(crate) use graph::{full_region, region_shape};
pub use requirements::{BufferBindingRequirement, CudaRequirements};
pub use source::CudaSource;

/// Strict rendering helper for a backend's typed template context.
pub fn render_template<T: Serialize>(source: &str, context: &T) -> Result<String, EmitError> {
    let mut environment = minijinja::Environment::new();

    environment.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    environment.set_auto_escape_callback(|_| minijinja::AutoEscape::None);

    Ok(environment.render_str(source, context)?)
}

pub fn emit(plan: &PhysicalPlan) -> Result<CudaSource, EmitError> {
    let mut buffers = Vec::new();

    // Collect buffer allocation and binding requirements for the plan's value instances.
    for (id, value) in plan.value_instances() {
        if matches!(value.storage(), Storage::Shared | Storage::Register) {
            return Err(EmitError::Unsupported(format!(
                "value {} dtype/storage",
                id.index()
            )));
        }

        let shape = value.shape().to_vec();
        let extent = region_shape(&shape)?;

        // Disallow empty matrices.
        if shape.contains(&0) {
            return Err(EmitError::Unsupported("empty matrix".into()));
        }

        // CuTe's statically specialized global layouts use 32-bit indices.
        // Reject oversized spans before constructing work or generating CUDA.
        if extent[0] > i32::MAX as usize / extent[1] {
            return Err(EmitError::Unsupported(
                "matrix exceeds 32-bit CuTe indexing".into(),
            ));
        }

        buffers.push(BufferBindingRequirement {
            dtype: value.dtype(),
            strides: if shape.len() == 1 {
                vec![1]
            } else {
                vec![shape[1], 1]
            },
            value: id.index(),
            bytes: extent[0] * extent[1] * value.dtype().size_bytes(),
            shape,
            alignment: 16,
            external: value.storage() == Storage::External,
            symmetric: false,
            input_names: plan
                .inputs()
                .iter()
                .filter(|b| b.value() == id)
                .map(|b| b.tensor().to_owned())
                .collect(),
            output_name: (plan.output().value() == id).then(|| plan.output().tensor().to_owned()),
        });
    }

    // flags
    let nvshmem = plan.world_size() > 1;
    let mut nvls = false;

    let mut shared_memory_bytes = 0;
    let mut operations = Vec::new();
    let mut bodies = Vec::new();

    // Generate each action's CUDA body, per-rank work, and resource requirements.
    for (action_id, action) in plan.actions() {
        if action.operations().len() != 1 {
            return Err(EmitError::Unsupported("non-singleton Action".into()));
        }
        let op_id = action.operations()[0];
        let op = plan.operation(action.operations()[0]).unwrap();

        let instance = match op.payload() {
            crate::OperationPayload::Compute(op) => op.implementation(),
            crate::OperationPayload::Communication(op) => op.implementation(),
        };

        let backend = instance.definition().cuda().ok_or_else(|| {
            EmitError::Unsupported(format!(
                "implementation {} has no CUDA backend",
                instance.id().as_str()
            ))
        })?;
        let result = backend.specialize(plan, op_id)?;

        bodies.push(result.body.clone());

        shared_memory_bytes = shared_memory_bytes.max(result.shared_memory_bytes);
        nvls |= result.nvls;

        for value in &result.symmetric_values {
            buffers[value.index()].symmetric = true;
        }

        operations.push((action_id, op_id, result));
    }

    // Build execution tasks and dependencies from the operations' read/write regions.
    let execution = graph::build(plan, &operations)?;

    let minimum_workers = if plan.world_size() > 1
        && execution.tasks.iter().any(|t| {
            t.stages
                .iter()
                .skip(1)
                .any(|s| s.dependencies.iter().any(|d| !t.dependencies.contains(d)))
        }) {
        2
    } else {
        1
    };

    let TargetCapability::Cuda(target) = plan.target();
    let requirements = CudaRequirements {
        target,
        cuda_arch: match target {
            CudaTargetCapability::Hopper => "sm_90a",
        },
        world_size: plan.world_size(),
        buffers,
        workspace_bytes: execution.workspace_bytes(),
        workspace_alignment: if plan.world_size() == 1 {
            1
        } else {
            graph::Workspace::ALIGNMENT
        },
        workspace_symmetric: nvshmem,
        cooperative_launch: plan.world_size() > 1,
        shared_memory_bytes,
        block_threads: 128,
        minimum_workers,
        nvshmem,
        nvls,
    };

    let code = render::program(&requirements, &execution, &bodies)?;

    Ok(CudaSource {
        code,
        requirements,
        execution,
    })
}
