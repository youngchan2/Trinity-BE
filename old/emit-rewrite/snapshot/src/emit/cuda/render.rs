//! Render a prepared CUDA program and its device bodies.
use serde::Serialize;
use std::fmt::Write;

use crate::emit::{EmittedSource, prepare::PreparedProgram};

use super::{
    BufferBindingRequirement, CudaRequirements, CudaSource, EmitError, Execution,
    body::BodyDefinition, prepare::CudaPrepared,
};

impl PreparedProgram for CudaPrepared {
    fn render(self: Box<Self>) -> Result<EmittedSource, EmitError> {
        let Self {
            requirements,
            bodies,
            execution,
        } = *self;

        let rendered_bodies = bodies
            .iter()
            .enumerate()
            .map(|(id, body)| render_body(id, body))
            .collect::<Result<Vec<_>, _>>()?;

        let code = program(&requirements, &execution, &rendered_bodies)?;

        Ok(EmittedSource::Cuda(CudaSource {
            code,
            requirements,
            execution,
            bodies: bodies
                .into_iter()
                .map(|definition| definition.body)
                .collect(),
        }))
    }
}

/// Coordinate transport belongs to the wrapper; device bodies receive values.
fn render_body(id: usize, definition: &BodyDefinition) -> Result<String, EmitError> {
    let mut code = format!(
        "template<class Runtime> __device__ bool operation_{id}(Bindings const& bindings, std::int64_t const* coordinates, void* memory, Runtime const& runtime) {{\n"
    );

    for (i, v) in definition.parameters.iter().enumerate() {
        writeln!(code, "std::int64_t {v}=coordinates[{i}];").unwrap();
    }

    code.push_str(&definition.body.render()?);
    code.push_str("return true; }\n");

    Ok(code)
}

/// Only operation inputs, resource requirements, and body text are shared.
/// Each execution mode owns its CUDA types, runtime, kernels, and host ABI.
#[derive(Serialize)]
pub(super) struct CommonContext<'a> {
    buffers: &'a [BufferBindingRequirement],
    shared: usize,
    bodies: &'a [String],
    common_types: &'static str,
    host_abi: &'static str,
    requirements_literal: String,
    execution_mode: usize,
}

impl<'a> CommonContext<'a> {
    pub(super) fn new(req: &'a CudaRequirements, bodies: &'a [String]) -> Self {
        Self {
            buffers: &req.buffers,
            shared: req.shared_memory_bytes,
            bodies,
            common_types: include_str!("types.cuh"),
            host_abi: include_str!("abi.h"),
            requirements_literal: serde_json::to_string(&serde_json::to_string(req).unwrap())
                .unwrap(),
            execution_mode: if req.world_size == 1 { 1 } else { 2 },
        }
    }
}

pub(super) fn program(
    req: &CudaRequirements,
    execution: &Execution,
    bodies: &[String],
) -> Result<String, EmitError> {
    match execution {
        Execution::Streamed(e) => super::streamed::render::program(req, e, bodies),
        Execution::Persistent(e) => super::persistent::render::program(req, e, bodies),
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

/// Strict rendering helper for a backend's typed template context.
pub fn render_template<T: Serialize>(source: &str, context: &T) -> Result<String, EmitError> {
    let mut environment = minijinja::Environment::new();

    environment.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    environment.set_auto_escape_callback(|_| minijinja::AutoEscape::None);

    Ok(environment.render_str(source, context)?)
}
