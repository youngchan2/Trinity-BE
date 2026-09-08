use serde::Serialize;

use super::graph::Launch;
use super::{BufferBindingRequirement, CudaRequirements, EmitError, Execution};

/// Only operation inputs, resource requirements, and body text are shared.
/// Each execution mode owns its CUDA types, runtime, kernels, and host ABI.
#[derive(Serialize)]
pub(super) struct CommonContext<'a> {
    buffers: &'a [BufferBindingRequirement],
    shared: usize,
    launches: &'a [Launch],
    bodies: &'a [String],
    common_types: &'static str,
}

impl<'a> CommonContext<'a> {
    pub(super) fn new(
        req: &'a CudaRequirements,
        execution: &'a Execution,
        bodies: &'a [String],
    ) -> Self {
        Self {
            buffers: &req.buffers,
            shared: req.shared_memory_bytes,
            launches: &execution.launches,
            bodies,
            common_types: include_str!("types.cuh"),
        }
    }
}

pub(super) fn program(
    req: &CudaRequirements,
    execution: &Execution,
    bodies: &[String],
) -> Result<String, EmitError> {
    if req.world_size == 1 {
        super::streamed::program(req, execution, bodies)
    } else {
        super::persistent::program(req, execution, bodies)
    }
}

pub(super) fn render<T: Serialize>(
    templates: &[(&str, &str)],
    context: &T,
) -> Result<String, EmitError> {
    let mut environment = minijinja::Environment::new();

    environment.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    environment.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
    environment.add_template("headers", include_str!("templates/headers.cu.j2"))?;

    for &(name, source) in templates {
        environment.add_template(name, source)?;
    }

    Ok(environment.get_template("program")?.render(context)?)
}
