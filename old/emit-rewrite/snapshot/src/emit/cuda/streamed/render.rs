//! Streamed kernel wrappers and host source rendering.
use serde::Serialize;
use std::{collections::BTreeMap, fmt::Write};

use super::super::{
    CudaRequirements, EmitError,
    access::fail,
    render::{CommonContext, render},
};
use super::{StreamedExecution, StreamedLaunch};

#[derive(Serialize)]
struct Kernel {
    body: usize,
    coordinates: String,
}
#[derive(Serialize)]
struct Context<'a> {
    #[serde(flatten)]
    common: CommonContext<'a>,
    kernels: Vec<Kernel>,
    launches: &'a [StreamedLaunch],
    types: &'static str,
    runtime: &'static str,
}

pub(in crate::emit::cuda) fn program(
    req: &CudaRequirements,
    execution: &StreamedExecution,
    bodies: &[String],
) -> Result<String, EmitError> {
    let mut kernels = Vec::new();
    for (task, grid) in execution.tasks.iter().zip(&execution.grids) {
        let mut code = format!(
            "  std::int64_t coordinates[{}]{{}};\n",
            task.domain.len().max(1)
        );
        let mut names = BTreeMap::new();
        for (i, d) in task.domain.iter().enumerate() {
            let divisor: u64 = grid.axes[i + 1..].iter().product();
            let start = d.start.cpp(&names).map_err(fail)?;
            let stop = d.stop.cpp(&names).map_err(fail)?;
            let step = d.step.cpp(&names).map_err(fail)?;
            writeln!(code,"  std::int64_t iteration_{i}=(block/{divisor}LL)%{}LL;\n  if(iteration_{i}>=(({stop})-({start}))/({step})) return;\n  coordinates[{i}]=({start})+iteration_{i}*({step});",grid.axes[i]).unwrap();
            names.insert(d.variable.clone(), format!("coordinates[{i}]"));
        }
        kernels.push(Kernel {
            body: task.body,
            coordinates: code,
        });
    }
    let context = Context {
        common: CommonContext::new(req, bodies),
        kernels,
        launches: &execution.launches,
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
