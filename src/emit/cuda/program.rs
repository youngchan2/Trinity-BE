//! Assemble CUDA bodies, tasks, dependencies, and resources for a physical plan.
use super::access::{self, Effects, fail};
use super::{EmitError, Execution, Stage, Task};
use crate::{LoopKind, PhysicalPlan, Statement};
use std::collections::BTreeMap;

pub(super) struct TaskDomain {
    pub statements: Vec<Statement>,
    pub task_set: usize,
    pub bindings: Vec<BTreeMap<String, i64>>,
    pub steps: BTreeMap<String, i64>,
    pub path: Vec<usize>,
    signature: Vec<[usize; 3]>,
}

struct Pending {
    body: usize,
    argument: usize,
    statement: usize,
    effects: Vec<Effects>,
}

#[derive(Default)]
struct Builder {
    bodies: Vec<TaskDomain>,
    tasks: Vec<Pending>,
    sets: BTreeMap<Vec<usize>, usize>,
}

impl Builder {
    fn sequence(
        &mut self,
        plan: &PhysicalPlan,
        statements: &[Statement],
        bindings: &BTreeMap<String, i64>,
        steps: &BTreeMap<String, i64>,
        path: &[usize],
        owner: usize,
    ) -> Result<(), EmitError> {
        let mut begin = 0;
        while begin < statements.len() {
            let mut nested = path.to_vec();
            nested.push(begin);

            if let Statement::Loop(l) = &statements[begin]
                && l.kind == LoopKind::Parallel
            {
                let step = l.domain.step.evaluate(bindings).map_err(fail)?;
                for value in l.domain.values(bindings).map_err(fail)? {
                    let mut b = bindings.clone();
                    let mut s = steps.clone();
                    b.insert(l.domain.variable.clone(), value);
                    s.insert(l.domain.variable.clone(), step);
                    self.sequence(plan, &l.body, &b, &s, &nested, owner)?;
                }
                begin += 1;
                continue;
            }

            let end = (begin + 1..statements.len())
                .find(|&i| matches!(&statements[i], Statement::Loop(l) if l.kind == LoopKind::Parallel))
                .unwrap_or(statements.len());

            let part = &statements[begin..end];
            let effects = (0..plan.world_size())
                .map(|rank| access::describe(plan, part, bindings, steps, rank))
                .collect::<Result<Vec<_>, _>>()?;

            let signature = effects[0]
                .events
                .iter()
                .map(|e| e.region.extent)
                .chain(std::iter::once([effects[0].work.stages.len(), 0, 0]))
                .collect::<Vec<_>>();

            let next_set = self.sets.len();
            let task_set = *self.sets.entry(nested.clone()).or_insert(next_set);
            let body = self
                .bodies
                .iter()
                .position(|b| {
                    b.statements == part
                        && b.steps == *steps
                        && b.path == nested
                        && b.signature == signature
                })
                .unwrap_or_else(|| {
                    self.bodies.push(TaskDomain {
                        statements: part.to_vec(),
                        task_set,
                        bindings: Vec::new(),
                        steps: steps.clone(),
                        path: nested,
                        signature,
                    });
                    self.bodies.len() - 1
                });

            let argument = self.bodies[body].bindings.len();

            self.bodies[body].bindings.push(bindings.clone());
            self.tasks.push(Pending {
                body,
                argument,
                statement: owner,
                effects,
            });
            begin = end;
        }
        Ok(())
    }
}

pub(super) struct ProgramEmission {
    pub execution: Execution,
    pub bodies: Vec<String>,
    pub resources: super::Resources,
    pub phases: Vec<super::Body>,
}

pub(super) fn build(plan: &PhysicalPlan) -> Result<ProgramEmission, EmitError> {
    let mut builder = Builder::default();

    for (i, statement) in plan.statements().iter().enumerate() {
        builder.sequence(
            plan,
            std::slice::from_ref(statement),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &[i],
            i,
        )?;
    }

    let mut code = Vec::new();
    let mut phases = Vec::new();
    let mut shared = 0;
    let mut symmetric = Vec::new();
    let mut nvls = false;
    let mut body_shared = Vec::new();

    for (id, body) in builder.bodies.iter().enumerate() {
        let (text, phase) = super::loop_body::render(plan, id, body)?;
        let resources = phase.resources();
        phases.push(phase);
        code.push(text);
        shared = shared.max(resources.shared_memory_bytes);
        body_shared.push(resources.shared_memory_bytes);
        symmetric.extend(resources.symmetric_values);
        nvls |= resources.nvls;
    }

    let mut tasks = Vec::new();
    let mut effects = Vec::new();
    let mut paths = Vec::new();
    for rank in 0..plan.world_size() {
        for (slot, p) in builder.tasks.iter().enumerate() {
            let body = &builder.bodies[p.body];
            let work = &p.effects[rank].work;
            tasks.push(Task {
                rank,
                slot,
                statement: p.statement,
                task_set: body.task_set,
                body: p.body,
                arguments: body.bindings[p.argument].clone(),
                argument: p.argument,
                coordinate: work.coordinate,
                shared_memory_bytes: body_shared[p.body],
                dependencies: Vec::new(),
                stages: work
                    .stages
                    .iter()
                    .map(|_| Stage {
                        dependencies: Vec::new(),
                    })
                    .collect(),
                ordered_collective: work.ordered_collective,
            });
            effects.push(p.effects[rank].clone());
            paths.push(body.path.clone());
        }
    }

    symmetric.sort();
    symmetric.dedup();

    Ok(ProgramEmission {
        execution: super::graph::resolve(plan, tasks, effects, paths)?,
        bodies: code,
        phases,
        resources: super::Resources {
            shared_memory_bytes: shared,
            symmetric_values: symmetric,
            nvls,
        },
    })
}
