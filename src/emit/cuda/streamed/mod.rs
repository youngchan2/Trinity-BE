use serde::Serialize;

use super::render::{CommonContext, render};
use super::{CudaRequirements, EmitError, Execution};

#[derive(Serialize)]
struct Context<'a> {
    #[serde(flatten)]
    common: CommonContext<'a>,
    tiles: Vec<[usize; 3]>,
    types: &'static str,
    runtime: &'static str,
}

pub(super) fn program(
    req: &CudaRequirements,
    execution: &Execution,
    bodies: &[String],
) -> Result<String, EmitError> {
    // The device only needs coordinates for blockIdx,
    // without task IDs or readiness tables.
    let context = Context {
        common: CommonContext::new(req, execution, bodies),
        tiles: execution.tasks.iter().map(|task| task.coordinate).collect(),
        types: include_str!("types.cuh"),
        runtime: include_str!("runtime.cuh"),
    };

    render(
        &[
            ("program", include_str!("program.cu.j2")),
            ("kernels", include_str!("kernels.cu.j2")),
            ("host", include_str!("host.cu.j2")),
        ],
        &context,
    )
}
