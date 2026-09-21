//! Persistent task dependencies and token placement.
//!
//! Common validation has already checked logical region legality. This path
//! connects physical entry/stage accesses to completion tokens, then orders
//! concrete tasks for interchangeable Worker CTAs. It never creates launches.

use std::collections::BTreeSet;

use super::super::{
    EmitError, Region, Task,
    access::fail,
    regions::{full_region, overlaps, subtract, validate_region},
};
use super::{Dependency, PersistentExecution, TaskScheduling, access::Effects};
use crate::PhysicalPlan;

#[derive(Clone, Copy)]
struct AccessToken {
    region: Region,
    token: Dependency,
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
fn add_dependency(
    tasks: &mut [TaskScheduling],
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
pub(in crate::emit::cuda) fn resolve(
    plan: &PhysicalPlan,
    tasks: Vec<Task>,
    mut schedule: Vec<TaskScheduling>,
    effects: Vec<Effects>,
) -> Result<PersistentExecution, EmitError> {
    let paths: Vec<_> = tasks.iter().map(|t| &t.path).collect();
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
            rank: schedule[task].rank,
            slot: schedule[task].slot,
        };
        if !event.write {
            for dependency in cover(region, &versions[rank][value])? {
                if dependency.slot < count {
                    let prior = dependency.rank * count + dependency.slot;
                    if prior != task && tasks[prior].concurrent(&tasks[task]) {
                        return Err(fail("read/write conflict between parallel iterations"));
                    }
                }
                add_dependency(&mut schedule, task, dependency, event.stage, count);
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
                    if tasks[index].concurrent(&tasks[task]) || paths[index] == paths[task] {
                        return Err(fail("overlapping accesses between parallel work"));
                    }
                }
                add_dependency(&mut schedule, task, prior.token, None, count);
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
        for task in schedule.iter_mut().skip(rank * count).take(count) {
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
    for (index, task) in schedule.iter().enumerate() {
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

    // Preserve the common rank-local topological order used by admission and
    // collective dispatch. This order does not define streamed kernel launches.
    let mut successors = vec![BTreeSet::new(); count];
    let mut indegrees = vec![0; count];
    for task in &schedule {
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
            ready.insert((tasks[slot].path.clone(), tasks[slot].body, slot));
        }
    }
    let mut order = Vec::new();
    while let Some((_, _, slot)) = ready.pop_first() {
        order.push(slot);
        for &next in &successors[slot] {
            indegrees[next] -= 1;
            if indegrees[next] == 0 {
                ready.insert((tasks[next].path.clone(), tasks[next].body, next));
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
    let mut reordered_schedule = Vec::new();
    let mut work = Vec::new();
    for rank in 0..plan.world_size() {
        for (slot, &old) in order.iter().enumerate() {
            let mut task = schedule[rank * count + old].clone();
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
            reordered_schedule.push(task);
            reordered.push(tasks[rank * count + old].clone());
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
    Ok(PersistentExecution {
        tasks: reordered,
        work,
        tasks_per_rank: count,
        output_dependencies,
        schedule: reordered_schedule,
    })
}
