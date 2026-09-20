//! CUDA program preparation and finalized execution requirements.

use crate::PhysicalPlan;
use crate::emit::BufferBindings;

use super::{
    CudaRequirements, EmitError, Execution,
    body::{self, BodyDefinition, ProgramBodies},
    domain,
    execution::Placement,
    requirements, validation,
};

/// Validated CUDA program ready for source rendering.
pub(in crate::emit) struct CudaPrepared {
    pub(super) requirements: CudaRequirements,
    pub(super) bodies: Vec<BodyDefinition>,
    pub(super) execution: Execution,
}

pub(in crate::emit) fn prepare(
    plan: &PhysicalPlan,
    bindings: BufferBindings,
) -> Result<CudaPrepared, EmitError> {
    requirements::validate_values(plan)?;

    let mut domains = domain::collect(plan);
    let placement = Placement::new(plan, &domains)?;

    let ProgramBodies { resources, bodies } = body::build_bodies(plan, &bindings, &mut domains)?;

    validation::validate(plan, &domains)?;

    let execution = placement.finish(plan, domains)?;

    let requirements =
        requirements::build_cuda_requirements(plan, bindings, &resources, &execution)?;

    Ok(CudaPrepared {
        requirements,
        bodies,
        execution,
    })
}
