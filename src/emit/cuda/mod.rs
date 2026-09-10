//! CUDA source generation.
//!
//! Backends describe concrete reads/writes and render a CTA body. This module
//! resolves those regions into execution metadata. Streamed and persistent
//! runtimes have separate templates and launch ABIs; CTA operations are shared
//! through binding, tile-coordinate, and input-readiness hooks.

mod access;
mod backend;
mod error;
mod graph;
mod indexing;
mod loop_body;
mod persistent;
mod phase;
mod program;
mod render;
mod requirements;
mod source;
mod streamed;

#[cfg(test)]
mod tests;

use serde::Serialize;

use crate::platform::{CudaTargetCapability, TargetCapability};
use crate::{PhysicalPlan, Storage};

pub use backend::{
    AccumulationBody, AccumulationScope, AccumulationTile, CudaImplementation, CudaPhaseTemplate,
    OperationSchedule, Region, Work,
};
pub use error::EmitError;
pub use phase::{Binding, Body, Code, Phase, Resources, Symbol, SymbolId};

pub use graph::{Dependency, Execution, Stage, Task};
pub use requirements::{BufferBindingRequirement, CudaRequirements};
pub use source::CudaSource;

/// Strict rendering helper for a backend's typed template context.
pub fn render_template<T: Serialize>(source: &str, context: &T) -> Result<String, EmitError> {
    let mut environment = minijinja::Environment::new();

    environment.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    environment.set_auto_escape_callback(|_| minijinja::AutoEscape::None);

    Ok(environment.render_str(source, context)?)
}

/// Generates CUDA source and launch requirements from a physical plan.
///
/// Collects buffer bindings, builds execution metadata and CTA bodies, and
/// renders the program into a [`CudaSource`].
///
/// # Errors
///
/// Returns an [`EmitError`] for unsupported shapes or resource requirements,
/// invalid backend contracts, or failures while building or rendering the program.
pub fn emit(plan: &PhysicalPlan) -> Result<CudaSource, EmitError> {
    let mut buffers = Vec::new();

    // Collect buffer allocation and binding requirements for the plan's value instances.
    for (id, value) in plan.value_instances() {
        let shape = value.shape().to_vec();

        // Region stores origins and extents in three axes,
        // padding unused axes with extent one.
        if !(1..=3).contains(&shape.len()) {
            return Err(EmitError::Unsupported("supported ranks are 1, 2, 3".into()));
        }

        let elements: usize = shape.iter().product();

        // Disallow empty matrices.
        if shape.contains(&0) {
            return Err(EmitError::Unsupported("empty matrix".into()));
        }

        if matches!(value.storage(), Storage::Shared | Storage::Register) {
            continue;
        }

        buffers.push(BufferBindingRequirement {
            dtype: value.dtype(),
            strides: (0..shape.len())
                .map(|i| shape[i + 1..].iter().product())
                .collect(),
            value: buffers.len(),
            bytes: elements * value.dtype().size_bytes(),
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
    let program::ProgramEmission {
        execution,
        bodies,
        resources,
        phases,
    } = program::build(plan)?;

    let shared_memory_bytes = resources.shared_memory_bytes;
    // CUTLASS sm90_common.inl: sm90_smem_capacity_bytes. Leave one aligned
    // block for the persistent wrapper's static control variables.
    if shared_memory_bytes > 232_448 - 128 {
        return Err(EmitError::Unsupported(
            "body exceeds Hopper CTA shared-memory capacity".into(),
        ));
    }

    let nvls = resources.nvls;
    for value in resources.symmetric_values {
        buffers[binding_slot(plan, value)?].symmetric = true;
    }

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
        bodies: phases,
    })
}

/// Dense external/global launch slots; local values never require allocations.
pub(crate) fn binding_slot(
    plan: &PhysicalPlan,
    value: crate::ValueInstanceId,
) -> Result<usize, EmitError> {
    plan.value_instances()
        .filter(|(_, v)| matches!(v.storage(), Storage::External | Storage::Global))
        .position(|(id, _)| id == value)
        .ok_or_else(|| EmitError::Contract("local value has no launch binding".into()))
}
