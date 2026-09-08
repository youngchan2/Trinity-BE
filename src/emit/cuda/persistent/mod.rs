use super::graph::Workspace;
use super::render::{CommonContext, render};
use super::{CudaRequirements, Dependency, EmitError, Execution};
use serde::Serialize;

#[derive(Serialize)]
struct Range {
    begin: usize,
    count: usize,
}
#[derive(Serialize)]
struct RenderTask {
    operation: usize,
    slot: usize,
    x: usize,
    y: usize,
    z: usize,
    dependencies: Range,
    stages: Range,
}
#[derive(Serialize)]
struct Context<'a> {
    #[serde(flatten)]
    common: CommonContext<'a>,
    world: usize,
    tasks_per_rank: usize,
    minimum_workers: usize,
    nvls: bool,
    tasks: Vec<RenderTask>,
    dependencies: Vec<Dependency>,
    stages: Vec<Range>,
    outputs: Vec<Range>,
    workspace: Workspace,
    types: &'static str,
    runtime: &'static str,
}

pub(super) fn program(
    req: &CudaRequirements,
    execution: &Execution,
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
    for task in &execution.tasks {
        let stage_range = Range {
            begin: stages.len(),
            count: task.stages.len(),
        };
        stages.extend(task.stages.iter().map(|s| range(&s.dependencies)));
        tasks.push(RenderTask {
            operation: task.operation,
            slot: task.slot,
            x: task.coordinate[0],
            y: task.coordinate[1],
            z: task.coordinate[2],
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
        common: CommonContext::new(req, execution, bodies),
        tasks_per_rank: execution.tasks_per_rank,
        minimum_workers: req.minimum_workers,
        nvls: req.nvls,
        tasks,
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
