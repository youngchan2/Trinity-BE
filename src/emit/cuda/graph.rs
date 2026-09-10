//! CUDA tasks and dependencies derived from a [`PhysicalPlan`].
//!
//! Lowering derives CTA bodies and coordinate-specific tasks from the ordered
//! [`Statement`](crate::Statement) program. A body may contain several statements,
//! and a loop statement may produce multiple bodies and tasks. Streamed execution
//! launches tasks as blocks in stream order; persistent execution schedules them
//! on interchangeable Worker CTAs using readiness tokens. This metadata belongs
//! to emission.

use std::collections::BTreeSet;

use serde::Serialize;

use super::access::{Effects, fail};
use super::{EmitError, Region, Work};
use crate::{PhysicalPlan, ValueInstanceId};

/// Execution metadata for all ranks of one physical plan.
#[derive(Debug, Clone)]
pub struct Execution {
    /// Access metadata, in the same rank-major order as tasks.
    pub work: Vec<Work>,
    /// Rank-major tasks, indexed by `rank * tasks_per_rank + slot`.
    pub tasks: Vec<Task>,
    /// Task count per rank, equal across ranks.
    pub tasks_per_rank: usize,
    /// Per-rank final-output producer tokens checked at persistent kernel exit.
    pub output_dependencies: Vec<Vec<Dependency>>,
    /// Task-set ranges in execution order, shared across ranks.
    pub launches: Vec<Launch>,
}

/// One task set's contiguous range of rank-local task slots.
///
/// Streamed execution uses one kernel with [`count`](Self::count) blocks;
/// persistent execution dispatches these tasks to its Worker grid.
#[derive(Debug, Clone, Serialize)]
pub struct Launch {
    /// Top-level Statement provenance; nested parallel sets can share it.
    pub statement: usize,
    /// Leaf task family beneath the parallel-loop collection, independent of body specialization.
    pub task_set: usize,
    /// Bodies specialized for tasks in this set (for example, clipped tails).
    pub bodies: Vec<usize>,
    /// Generated device body ID.
    pub body: usize,
    /// First rank-local task slot.
    pub begin: usize,
    /// Task count in the range, equal across ranks.
    pub count: usize,
}

/// One CTA's invocation of a generated body at bound loop coordinates.
///
/// A GEMM task owns its output tile across all K stages through the final store.
#[derive(Debug, Clone, Serialize)]
pub struct Task {
    pub rank: usize,
    /// Rank-local ID indexing the task's completion token and scheduler state.
    pub slot: usize,
    /// Position in the plan's top-level Statement list, for provenance.
    /// Execution dispatch uses `body` and `arguments`.
    pub statement: usize,
    /// Leaf task family beneath the parallel-loop collection, independent of body specialization.
    pub task_set: usize,
    /// Device body dispatch ID; a body may contain several operations.
    pub body: usize,
    /// Bound lexical parallel-loop coordinates for this invocation.
    pub arguments: std::collections::BTreeMap<String, i64>,
    /// Index in this body's argument table, independent of task slot and tile geometry.
    pub argument: usize,
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

#[derive(Clone, Copy)]
struct AccessToken {
    region: Region,
    token: Dependency,
}

pub(super) fn overlaps(a: Region, b: Region) -> bool {
    a.value == b.value
        && a.rank == b.rank
        && (0..3).all(|i| {
            a.origin[i] < b.origin[i] + b.extent[i] && b.origin[i] < a.origin[i] + a.extent[i]
        })
}
/// Exact rectangular subtraction; disjoint pieces retain the previous version.
fn subtract(region: Region, cut: Region) -> Vec<Region> {
    if !overlaps(region, cut) {
        return vec![region];
    }
    let mut core = region;
    let mut pieces = Vec::new();
    for i in 0..3 {
        let lo = core.origin[i].max(cut.origin[i]);
        let hi = (core.origin[i] + core.extent[i]).min(cut.origin[i] + cut.extent[i]);
        if core.origin[i] < lo {
            let mut p = core;
            p.extent[i] = lo - core.origin[i];
            pieces.push(p);
            core.extent[i] -= lo - core.origin[i];
            core.origin[i] = lo;
        }
        if core.origin[i] + core.extent[i] > hi {
            let mut p = core;
            p.origin[i] = hi;
            p.extent[i] = core.origin[i] + core.extent[i] - hi;
            pieces.push(p);
            core.extent[i] = hi - core.origin[i];
        }
    }
    pieces
}
fn validate_region(plan: &PhysicalPlan, r: Region) -> Result<(), EmitError> {
    let value = plan
        .value_instance(r.value)
        .ok_or_else(|| fail("unknown value"))?;
    let shape = region_shape(value.shape())?;
    if r.rank >= plan.world_size()
        || r.extent.contains(&0)
        || (0..3).any(|i| {
            r.origin[i]
                .checked_add(r.extent[i])
                .is_none_or(|end| end > shape[i])
        })
    {
        return Err(fail("out-of-bounds execution region"));
    }
    Ok(())
}
fn cover(read: Region, versions: &[AccessToken]) -> Result<Vec<Dependency>, EmitError> {
    let mut remaining = vec![read];
    let mut deps = BTreeSet::new();
    for p in versions {
        if overlaps(read, p.region) {
            deps.insert(p.token);
            remaining = remaining
                .into_iter()
                .flat_map(|r| subtract(r, p.region))
                .collect();
        }
    }
    if !remaining.is_empty() {
        return Err(fail(format!(
            "read before production or incomplete writes to value {}",
            read.value.index()
        )));
    }
    Ok(deps.into_iter().collect())
}
fn concurrent(a: &Task, b: &Task) -> bool {
    a.arguments
        .iter()
        .any(|(name, value)| b.arguments.get(name).is_some_and(|other| other != value))
}
fn add_dependency(
    tasks: &mut [Task],
    task: usize,
    dependency: Dependency,
    stage: Option<usize>,
    count: usize,
) {
    if dependency
        == (Dependency {
            rank: tasks[task].rank,
            slot: tasks[task].slot,
        })
        || dependency.rank == tasks[task].rank && dependency.slot == count
    {
        return;
    }
    match stage {
        Some(stage) => tasks[task].stages[stage].dependencies.push(dependency),
        None => tasks[task].dependencies.push(dependency),
    }
}

/// Resolve lexical versions and stage readiness for every kind of body together.
pub(super) fn resolve(
    plan: &PhysicalPlan,
    mut tasks: Vec<Task>,
    effects: Vec<Effects>,
    paths: Vec<Vec<usize>>,
) -> Result<Execution, EmitError> {
    let count = tasks.len() / plan.world_size();
    let mut versions =
        vec![vec![Vec::<AccessToken>::new(); plan.value_instances().len()]; plan.world_size()];
    let mut readers = versions.clone();
    let rewritten: BTreeSet<_> = plan
        .value_instances()
        .filter(|(id, _)| {
            plan.operations()
                .filter(|(_, op)| op.outputs().contains(id))
                .count()
                > 1
        })
        .map(|(id, _)| id)
        .collect();
    for (rank, values) in versions.iter_mut().enumerate() {
        for input in plan.inputs() {
            if values[input.value().index()].is_empty() {
                values[input.value().index()].push(AccessToken {
                    region: full_region(plan, input.value(), rank),
                    token: Dependency { rank, slot: count },
                });
            }
        }
    }
    let mut timeline = Vec::new();
    for (task, effect) in effects.iter().enumerate() {
        for (event, _) in effect.events.iter().enumerate() {
            timeline.push((task, event));
        }
    }
    timeline.sort_by(|&(a, ae), &(b, be)| (&paths[a], ae, a).cmp(&(&paths[b], be, b)));
    for (task, event_index) in timeline {
        let event = &effects[task].events[event_index];
        let region = event.region;
        validate_region(plan, region)?;
        let rank = region.rank;
        let value = region.value.index();
        let token = Dependency {
            rank: tasks[task].rank,
            slot: tasks[task].slot,
        };
        if !event.write {
            for dependency in cover(region, &versions[rank][value])? {
                if dependency.slot < count {
                    let prior = dependency.rank * count + dependency.slot;
                    if prior != task && concurrent(&tasks[prior], &tasks[task]) {
                        return Err(fail("read/write conflict between parallel iterations"));
                    }
                }
                add_dependency(&mut tasks, task, dependency, event.stage, count);
            }
            if rewritten.contains(&region.value)
                && !readers[rank][value]
                    .iter()
                    .any(|r| r.region == region && r.token == token)
            {
                readers[rank][value].push(AccessToken { region, token });
            }
        } else {
            for prior in versions[rank][value].iter().chain(&readers[rank][value]) {
                if !overlaps(region, prior.region) || prior.token == token {
                    continue;
                }
                if prior.token.slot < count {
                    let index = prior.token.rank * count + prior.token.slot;
                    if concurrent(&tasks[index], &tasks[task]) || paths[index] == paths[task] {
                        return Err(fail("overlapping accesses between parallel work"));
                    }
                }
                add_dependency(&mut tasks, task, prior.token, None, count);
            }
            let previous = std::mem::take(&mut versions[rank][value]);
            versions[rank][value] = previous
                .into_iter()
                .flat_map(|p| {
                    subtract(p.region, region)
                        .into_iter()
                        .map(move |region| AccessToken { region, ..p })
                })
                .collect();
            versions[rank][value].push(AccessToken { region, token });
            readers[rank][value] = std::mem::take(&mut readers[rank][value])
                .into_iter()
                .flat_map(|p| {
                    subtract(p.region, region)
                        .into_iter()
                        .map(move |region| AccessToken { region, ..p })
                })
                .collect();
        }
    }
    for rank in 0..plan.world_size() {
        let mut previous = None;
        for task in tasks.iter_mut().skip(rank * count).take(count) {
            if task.ordered_collective {
                if let Some(slot) = previous {
                    task.dependencies
                        .extend((0..plan.world_size()).map(|rank| Dependency { rank, slot }));
                }
                previous = Some(task.slot);
            }
            for stage in &mut task.stages {
                stage.dependencies.sort_unstable();
                stage.dependencies.dedup();
            }
            if let Some(first) = task.stages.first() {
                task.dependencies.extend(&first.dependencies);
            }
            task.dependencies.sort_unstable();
            task.dependencies.dedup();
        }
    }
    let mut output_dependencies = Vec::new();
    for (rank, values) in versions.iter().enumerate() {
        output_dependencies.push(cover(
            full_region(plan, plan.output().value(), rank),
            &values[plan.output().value().index()],
        )?);
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

    // A streamed launch has no inter-CTA ordering. A task set must therefore be
    // parallel after its internal, same-task accesses have been absorbed.
    for task in &tasks {
        for dep in task
            .dependencies
            .iter()
            .chain(task.stages.iter().flat_map(|s| &s.dependencies))
        {
            if plan.world_size() == 1
                && dep.slot < count
                && tasks[dep.rank * count + dep.slot].task_set == task.task_set
            {
                return Err(fail("a task set requires inter-task synchronization"));
            }
        }
    }

    // A shared launch order must respect every rank's entry and future-stage
    // prerequisites. Prefer ready work of the same body without sorting through
    // a dependency. A body may consequently have more than one launch range.
    let mut successors = vec![BTreeSet::new(); count];
    let mut indegrees = vec![0; count];
    for task in &tasks {
        for dep in task
            .dependencies
            .iter()
            .chain(task.stages.iter().flat_map(|s| &s.dependencies))
        {
            if dep.slot < count && dep.slot != task.slot && successors[dep.slot].insert(task.slot) {
                indegrees[task.slot] += 1;
            }
        }
    }
    let mut ready = BTreeSet::new();
    for (slot, &degree) in indegrees.iter().enumerate() {
        if degree == 0 {
            ready.insert((tasks[slot].task_set, tasks[slot].body, slot));
        }
    }
    let mut order = Vec::new();
    while let Some((_, _, slot)) = ready.pop_first() {
        order.push(slot);
        for &next in &successors[slot] {
            indegrees[next] -= 1;
            if indegrees[next] == 0 {
                ready.insert((tasks[next].task_set, tasks[next].body, next));
            }
        }
    }
    if order.len() != count {
        return Err(fail("no common acyclic launch order across ranks"));
    }
    let mut remap = vec![0; count];
    for (slot, &old) in order.iter().enumerate() {
        remap[old] = slot;
    }
    let remap_dep = |d: &mut Dependency| {
        if d.slot < count {
            d.slot = remap[d.slot];
        }
    };
    let mut reordered = Vec::new();
    let mut work = Vec::new();
    for rank in 0..plan.world_size() {
        for (slot, &old) in order.iter().enumerate() {
            let mut task = tasks[rank * count + old].clone();
            task.slot = slot;
            for d in &mut task.dependencies {
                remap_dep(d);
            }
            task.dependencies.sort_unstable();
            for stage in &mut task.stages {
                for d in &mut stage.dependencies {
                    remap_dep(d);
                }
                stage.dependencies.sort_unstable();
            }
            reordered.push(task);
            work.push(effects[rank * count + old].work.clone());
        }
    }
    for ds in &mut output_dependencies {
        for d in ds.iter_mut() {
            remap_dep(d);
        }
        ds.sort_unstable();
        ds.dedup();
    }
    let mut launches: Vec<Launch> = Vec::new();
    for task in reordered.iter().take(count) {
        if let Some(last) = launches.last_mut()
            && last.task_set == task.task_set
        {
            last.count += 1;
            if !last.bodies.contains(&task.body) {
                last.bodies.push(task.body);
            }
        } else {
            launches.push(Launch {
                statement: task.statement,
                task_set: task.task_set,
                bodies: vec![task.body],
                body: task.body,
                begin: task.slot,
                count: 1,
            });
        }
    }
    Ok(Execution {
        tasks: reordered,
        work,
        tasks_per_rank: count,
        output_dependencies,
        launches,
    })
}

pub(crate) fn full_region(plan: &PhysicalPlan, value: ValueInstanceId, rank: usize) -> Region {
    Region::new(
        value,
        rank,
        [0; 3],
        region_shape(plan.value_instance(value).unwrap().shape()).expect("validated tensor rank"),
    )
}
pub(crate) fn region_shape(shape: &[usize]) -> Result<[usize; 3], EmitError> {
    if !(1..=3).contains(&shape.len()) {
        return Err(fail("supported tensor ranks are 1, 2, 3"));
    }
    let mut out = [1; 3];
    out[..shape.len()].copy_from_slice(shape);
    Ok(out)
}
