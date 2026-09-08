//! CUDA tasks and dependencies derived from a [`PhysicalPlan`].
//!
//! Each singleton [`Action`](crate::Action) expands into tile or chunk tasks
//! (`WorkItem`s in the architecture). Streamed execution launches them as blocks
//! in stream order; persistent execution schedules them on interchangeable Worker
//! CTAs using readiness tokens. This metadata belongs to emission.

use std::collections::BTreeSet;

use serde::Serialize;

use super::{EmitError, OperationEmission, Region};
use crate::{ActionId, OperationId, PhysicalPlan, ValueInstanceId};

/// Execution metadata for all ranks of one physical plan.
#[derive(Debug, Clone)]
pub struct Execution {
    /// Rank-major tasks, indexed by `rank * tasks_per_rank + slot`.
    pub tasks: Vec<Task>,
    /// Task count per rank, equal across ranks.
    pub tasks_per_rank: usize,
    /// Per-rank final-output producer tokens checked at persistent kernel exit.
    pub output_dependencies: Vec<Vec<Dependency>>,
    /// Operation task ranges in topological order, shared across ranks.
    pub launches: Vec<Launch>,
}

/// One operation's contiguous range of rank-local task slots.
///
/// Streamed execution uses one kernel with [`count`](Self::count) blocks;
/// persistent execution dispatches these tasks to its Worker grid.
#[derive(Debug, Clone, Serialize)]
pub struct Launch {
    /// Canonical operation index in the physical plan.
    pub operation: usize,
    /// First rank-local task slot.
    pub begin: usize,
    /// Task count in the range, equal across ranks.
    pub count: usize,
}

/// One CTA's execution of an operation at a tile or chunk coordinate.
///
/// A GEMM task owns its output tile across all K stages through the final store.
#[derive(Debug, Clone, Serialize)]
pub struct Task {
    pub rank: usize,
    /// Rank-local ID indexing the task's completion token and scheduler state.
    pub slot: usize,
    /// Canonical index of the owning Action.
    pub action: usize,
    /// Canonical operation index selecting the generated device body.
    pub operation: usize,
    /// Backend coordinates: WGMMA tile indices, or communication element/chunk
    /// offsets and an optional source rank.
    pub coordinate: [usize; 3],
    /// Per-CTA shared scratch in bytes; kernels reserve the maximum across tasks.
    pub shared_memory_bytes: usize,
    /// Tokens required before task entry, including the first stage's tokens.
    pub dependencies: Vec<Dependency>,
    /// Input-readiness checkpoints in execution order.
    pub stages: Vec<Stage>,
    /// World-team collective, dependent on the previous collective on every rank.
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
    /// Task slot, or `Execution::tasks_per_rank` for external-input readiness.
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

impl Execution {
    /// Control workspace size in bytes; zero for single-rank streamed execution.
    pub fn workspace_bytes(&self) -> usize {
        if self.output_dependencies.len() == 1 {
            0
        } else {
            self.workspace().bytes
        }
    }

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

/// A produced region and its readiness token, which may reside on different ranks.
#[derive(Clone, Copy)]
struct Producer {
    region: Region,
    token: Dependency,
}

/// Resolve backend regions into task, stage and output dependencies.
///
/// # Errors
///
/// Returns [`EmitError::Contract`] for invalid regions, coverage or cycles.
pub(super) fn build(
    plan: &PhysicalPlan,
    operations: &[(ActionId, OperationId, OperationEmission)],
) -> Result<Execution, EmitError> {
    // Total task count per rank.
    let count: usize = operations.iter().map(|(_, _, e)| e.work[0].len()).sum();

    let mut producers =
        vec![vec![Vec::<Producer>::new(); plan.value_instances().len()]; plan.world_size()];

    for rank in 0..plan.world_size() {
        // Add producers for this rank's PhysicalPlan inputs.
        for binding in plan.inputs() {
            let value = plan.value_instance(binding.value()).unwrap();

            producers[rank][binding.value().index()].push(Producer {
                // Full input region on this rank.
                region: Region::new(
                    binding.value(),
                    rank,
                    [0, 0],
                    [value.shape()[0], value.shape()[1]],
                ),
                // Shared input-readiness token, after the task slots.
                token: Dependency { rank, slot: count },
            });
        }

        let mut slot = 0;
        for (_, _, emission) in operations {
            for work in &emission.work[rank] {
                for &region in &work.writes {
                    validate_region(plan, region)?;
                    producers[region.rank][region.value.index()].push(Producer {
                        region,
                        token: Dependency { rank, slot },
                    });
                }
                slot += 1;
            }
        }
    }

    // Task write regions must cover each rank-local tensor exactly once.
    // External inputs are registered as full-tensor regions above.
    for rank in &producers {
        for (pieces, (value_id, value)) in rank.iter().zip(plan.value_instances()) {
            let value_id = value_id.index();
            let shape = value.shape();

            let mut area = 0usize;

            for (i, piece) in pieces.iter().enumerate() {
                area += piece.region.extent[0] * piece.region.extent[1];
                // Reject multiple writers to the same tensor elements.
                if pieces[..i]
                    .iter()
                    .any(|other| overlaps(piece.region, other.region))
                {
                    return Err(EmitError::Contract(format!(
                        "overlapping writes to value {value_id}"
                    )));
                }
            }

            // In-bounds, non-overlapping regions must leave no elements unwritten.
            if area != shape[0] * shape[1] {
                return Err(EmitError::Contract(format!(
                    "incomplete writes to value {value_id}"
                )));
            }
        }
    }

    let mut tasks = Vec::new();
    let mut output_dependencies = Vec::new();
    let mut launches = Vec::new();

    // Turn backend work into execution tasks, using producers to connect each
    // task's reads to the readiness tokens it must wait for.
    for (rank, rank_producers) in producers.iter().enumerate() {
        // Rank-local task index, also used as its completion-token slot.
        let mut slot = 0;
        let mut previous_collective = None;

        for (action, operation, emission) in operations {
            // Launch ranges are identical across ranks; record them once.
            if rank == 0 {
                launches.push(Launch {
                    operation: operation.index(),
                    begin: slot,
                    count: emission.work[rank].len(),
                });
            }

            for work in &emission.work[rank] {
                let mut dependencies = resolve(plan, &producers, &work.reads, rank, count)?;
                // Wait for the previous collective to finish on every rank.
                if work.ordered_collective {
                    if let Some(previous) = previous_collective {
                        dependencies.extend((0..plan.world_size()).map(|rank| Dependency {
                            rank,
                            slot: previous,
                        }));
                    }
                    previous_collective = Some(slot);
                }

                // Resolve inputs needed at each stage, e.g. each GEMM K-stage.
                let stages = work
                    .stages
                    .iter()
                    .map(|reads| {
                        Ok(Stage {
                            dependencies: resolve(plan, &producers, reads, rank, count)?,
                        })
                    })
                    .collect::<Result<Vec<_>, EmitError>>()?;

                // The first stage's inputs must be ready before task entry.
                if let Some(first) = stages.first() {
                    dependencies.extend(&first.dependencies);
                }

                dependencies.sort_unstable();
                dependencies.dedup();

                tasks.push(Task {
                    rank,
                    slot,
                    action: action.index(),
                    operation: operation.index(),
                    coordinate: work.coordinate,
                    shared_memory_bytes: emission.shared_memory_bytes,
                    dependencies,
                    stages,
                    ordered_collective: work.ordered_collective,
                });
                slot += 1;
            }
        }

        // Persistent kernel exit waits for all final-output producers on this rank.
        output_dependencies.push(
            rank_producers[plan.output().value().index()]
                .iter()
                .map(|p| p.token)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        );
    }

    // Entry and future-stage dependencies must form an acyclic global graph.
    // This catches a malformed backend before it can strand a resident worker.
    let mut successors = vec![Vec::new(); tasks.len()];
    let mut indegrees = vec![0usize; tasks.len()];
    for (index, task) in tasks.iter().enumerate() {
        let dependencies: BTreeSet<_> = task
            .dependencies
            .iter()
            .chain(task.stages.iter().flat_map(|s| &s.dependencies))
            .copied()
            .collect();
        for dependency in dependencies {
            if dependency.slot == count {
                continue;
            }
            let source = dependency.rank * count + dependency.slot;
            successors[source].push(index);
            indegrees[index] += 1;
        }
    }

    let mut ready: Vec<_> = indegrees
        .iter()
        .enumerate()
        .filter_map(|(i, n)| (*n == 0).then_some(i))
        .collect();

    let mut visited = 0;
    while let Some(task) = ready.pop() {
        visited += 1;
        for &next in &successors[task] {
            indegrees[next] -= 1;
            if indegrees[next] == 0 {
                ready.push(next);
            }
        }
    }

    if visited != tasks.len() {
        return Err(EmitError::Contract("cyclic execution dependencies".into()));
    }

    Ok(Execution {
        tasks,
        tasks_per_rank: count,
        output_dependencies,
        launches,
    })
}

fn validate_region(plan: &PhysicalPlan, region: Region) -> Result<(), EmitError> {
    let value = plan
        .value_instance(region.value)
        .ok_or_else(|| EmitError::Contract("unknown value".into()))?;

    if region.rank >= plan.world_size()
        || region.extent.contains(&0)
        || (0..2).any(|axis| region.origin[axis] + region.extent[axis] > value.shape()[axis])
    {
        return Err(EmitError::Contract("out-of-bounds execution region".into()));
    }

    Ok(())
}

fn resolve(
    plan: &PhysicalPlan,
    producers: &[Vec<Vec<Producer>>],
    reads: &[Region],
    local_rank: usize,
    external_slot: usize,
) -> Result<Vec<Dependency>, EmitError> {
    let mut dependencies = BTreeSet::new();
    for &read in reads {
        validate_region(plan, read)?;
        for producer in &producers[read.rank][read.value.index()] {
            if overlaps(read, producer.region)
                && !(producer.token.rank == local_rank && producer.token.slot == external_slot)
            {
                dependencies.insert(producer.token);
            }
        }
    }
    Ok(dependencies.into_iter().collect())
}

fn overlaps(a: Region, b: Region) -> bool {
    (0..2).all(|axis| {
        a.origin[axis] < b.origin[axis] + b.extent[axis]
            && b.origin[axis] < a.origin[axis] + a.extent[axis]
    })
}

pub(crate) fn full_region(plan: &PhysicalPlan, value: ValueInstanceId, rank: usize) -> Region {
    let shape = plan.value_instance(value).unwrap().shape();
    Region::new(value, rank, [0, 0], [shape[0], shape[1]])
}
