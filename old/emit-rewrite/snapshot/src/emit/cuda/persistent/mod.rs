//! Persistent execution metadata and task construction.
pub(super) mod access;
pub(super) mod graph;
pub(super) mod render;

use super::{
    EmitError, Task, Work,
    domain::{TaskDomain, coordinates},
};
use crate::PhysicalPlan;
use serde::Serialize;

/// Reserve one 128-byte block for static scheduler variables and padding before
/// the aligned dynamic storage in kernels.cu.j2. Includes runtime.cuh flags.
pub(super) const SHARED_MEMORY_RESERVE_BYTES: usize = 128;

/// Persistent placement, in rank-major task/metadata/work order.
#[derive(Debug, Clone)]
pub struct PersistentExecution {
    pub tasks: Vec<Task>,
    pub schedule: Vec<TaskScheduling>,
    pub work: Vec<Work>,
    pub tasks_per_rank: usize,
    pub output_dependencies: Vec<Vec<Dependency>>,
}

/// Runtime scheduling data parallel to `PersistentExecution::tasks`.
#[derive(Debug, Clone, Serialize)]
pub struct TaskScheduling {
    pub rank: usize,
    pub slot: usize,
    pub dependencies: Vec<Dependency>,
    pub stages: Vec<Stage>,
    pub ordered_collective: bool,
}

/// Input readiness checked before a stage load, such as a GEMM K-stage.
#[derive(Debug, Clone, Serialize)]
pub struct Stage {
    /// Sorted, deduplicated producer tokens; local external inputs are omitted.
    pub dependencies: Vec<Dependency>,
}

/// A task-completion or external-input token, ready when it holds the current epoch.
///
/// The publishing rank may differ from the data's destination rank (peer push).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Dependency {
    /// Rank whose control workspace contains the token.
    pub rank: usize,
    /// Task slot, or `PersistentExecution::tasks_per_rank` for external-input readiness.
    pub slot: usize,
}

/// Layout of each rank's symmetric control workspace for persistent execution.
#[derive(Debug, Clone, Serialize)]
pub(super) struct Workspace {
    /// Base and array alignment in bytes.
    pub alignment: usize,
    /// Header capacity in bytes, starting at offset zero.
    pub header_bytes: usize,
    /// Byte offset of `u64` task tokens plus one external-input token.
    pub tokens: usize,
    /// Byte offset of `u32` task states.
    pub states: usize,
    /// Byte offset of the ready queue of `u32` task slots.
    pub queue: usize,
    /// Total allocation size, including alignment padding.
    pub bytes: usize,
}

impl Workspace {
    /// Control workspace alignment shared with the CUDA ABI.
    pub const ALIGNMENT: usize = 8;
    /// Reserved queue header capacity, independent of allocation alignment.
    pub const HEADER_BYTES: usize = 128;
}

impl PersistentExecution {
    /// Compute aligned control workspace offsets and size.
    pub(super) fn workspace(&self) -> Workspace {
        let mut cursor = Workspace::HEADER_BYTES;
        let alignment_mask = Workspace::ALIGNMENT - 1;

        let mut array = |count: usize, width: usize| {
            // Round cursor up to the next 8-byte boundary, keeping aligned offsets unchanged.
            cursor = (cursor + alignment_mask) & !alignment_mask;
            let start = cursor;
            cursor += count * width;
            start
        };

        let tokens = array(self.tasks_per_rank + 1, 8);
        let states = array(self.tasks_per_rank, 4);
        let queue = array(self.tasks_per_rank, 4);
        let bytes = (cursor + alignment_mask) & !alignment_mask;

        Workspace {
            alignment: Workspace::ALIGNMENT,
            header_bytes: Workspace::HEADER_BYTES,
            tokens,
            states,
            queue,
            bytes,
        }
    }
}

pub(super) fn build(
    plan: &PhysicalPlan,
    domains: &[TaskDomain],
) -> Result<PersistentExecution, EmitError> {
    let mut tasks = Vec::new();
    let mut schedule = Vec::new();
    let mut effects = Vec::new();

    for rank in 0..plan.world_size() {
        let mut slot = 0;
        for domain in domains {
            coordinates(&domain.task.domain, &mut |bindings, steps| {
                let effect = access::describe(plan, &domain.statements, bindings, steps, rank)?;

                schedule.push(TaskScheduling {
                    rank,
                    slot,
                    dependencies: Vec::new(),
                    stages: effect
                        .work
                        .stages
                        .iter()
                        .map(|_| Stage {
                            dependencies: Vec::new(),
                        })
                        .collect(),
                    ordered_collective: effect.work.ordered_collective,
                });

                tasks.push(domain.task.bind(bindings)?);
                effects.push(effect);

                slot += 1;

                Ok(())
            })?;
        }
    }

    graph::resolve(plan, tasks, schedule, effects)
}
