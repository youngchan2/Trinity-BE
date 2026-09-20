//! Persistent task tables, kernel wrappers, and host source rendering.
use serde::Serialize;
use std::collections::BTreeMap;

use super::super::{
    CudaRequirements, EmitError,
    render::{CommonContext, render},
};
use super::{Dependency, PersistentExecution, Workspace};

#[derive(Serialize)]
struct Range {
    begin: usize,
    count: usize,
}

#[derive(Serialize)]
struct RenderTask {
    body: usize,
    slot: usize,
    argument: usize,
    dependencies: Range,
    stages: Range,
}

#[derive(Serialize)]
struct Context<'a> {
    #[serde(flatten)]
    common: CommonContext<'a>,
    world: usize,
    tasks_per_rank: usize,
    nvls: bool,
    tasks: Vec<RenderTask>,
    arguments: Vec<i64>,
    dependencies: Vec<Dependency>,
    stages: Vec<Range>,
    outputs: Vec<Range>,
    workspace: Workspace,
    types: &'static str,
    runtime: &'static str,
}

pub(in crate::emit::cuda) fn program(
    req: &CudaRequirements,
    execution: &PersistentExecution,
    bodies: &[String],
) -> Result<String, EmitError> {
    let mut dependencies = Vec::new();
    let mut range = |values: &[Dependency]| {
        let result = Range {
            begin: dependencies.len(),
            count: values.len(),
        };
        dependencies.extend_from_slice(values);
        result
    };

    let mut stages = Vec::new();
    let mut tasks = Vec::new();
    let mut arguments = Vec::new();
    let mut argument_offsets = BTreeMap::new();
    for (body_task, task) in execution.tasks.iter().zip(&execution.schedule) {
        let stage_range = Range {
            begin: stages.len(),
            count: task.stages.len(),
        };

        stages.extend(task.stages.iter().map(|s| range(&s.dependencies)));
        let bindings = body_task.bindings()?;
        let coordinates: Vec<_> = body_task
            .domain
            .iter()
            .map(|d| bindings[&d.variable])
            .collect();
        // Rank and scheduler slot do not change the Body coordinate ABI.
        let argument = *argument_offsets
            .entry(coordinates.clone())
            .or_insert_with(|| {
                let offset = arguments.len();
                arguments.extend(coordinates);
                offset
            });
        tasks.push(RenderTask {
            body: body_task.body,
            slot: task.slot,
            argument,
            dependencies: range(&task.dependencies),
            stages: stage_range,
        });
    }

    let outputs = execution
        .output_dependencies
        .iter()
        .map(|d| range(d))
        .collect();

    let context = Context {
        world: req.world_size,
        common: CommonContext::new(req, bodies),
        tasks_per_rank: execution.tasks_per_rank,
        nvls: req.nvls,
        tasks,
        arguments,
        dependencies,
        stages,
        outputs,
        workspace: execution.workspace(),
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
