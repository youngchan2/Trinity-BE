//! Ordered task domains and coordinate traversal.
use std::collections::BTreeMap;

use serde::Serialize;

use super::{EmitError, access::fail};
use crate::{IndexExpr, LoopDomain, LoopKind, PhysicalPlan, Statement};

/// Apply one body to every coordinate in an ordered parallel-loop domain.
///
/// Streamed tasks retain the entire domain. Persistent tasks use singleton
/// ranges, preserving the original steps (including `elem` indexing). Completion
/// means all computations and final stores in this domain have finished.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Task {
    pub body: usize,
    /// Top-level Statement provenance, independent of body identity.
    pub statement: usize,
    /// Lexical location of the ordered body within the Statement tree.
    pub path: Vec<usize>,
    /// Outermost parallel loop first; sequential loops remain in the body.
    pub domain: Vec<LoopDomain>,
}

impl Task {
    /// Coordinates of a singleton task. Whole domains have no single binding.
    pub fn bindings(&self) -> Result<BTreeMap<String, i64>, EmitError> {
        let mut bindings = BTreeMap::new();
        for d in &self.domain {
            let (start, stop, step) = d.bounds(&bindings).map_err(fail)?;
            if stop - start != step {
                return Err(fail("task does not have singleton coordinates"));
            }
            bindings.insert(d.variable.clone(), start);
        }
        Ok(bindings)
    }

    pub(super) fn bind(&self, bindings: &BTreeMap<String, i64>) -> Result<Self, EmitError> {
        let mut task = self.clone();
        for d in &mut task.domain {
            let start = bindings[&d.variable];
            let step = d.step.evaluate(bindings).map_err(fail)?;
            d.start = IndexExpr::Constant(start);
            d.stop = IndexExpr::Constant(
                start
                    .checked_add(step)
                    .ok_or_else(|| fail("coordinate overflow"))?,
            );
            d.step = IndexExpr::Constant(step);
        }
        Ok(task)
    }

    pub(super) fn concurrent(&self, other: &Self) -> bool {
        self.domain.iter().any(|a| {
            other
                .domain
                .iter()
                .any(|b| a.variable == b.variable && a.start != b.start)
        })
    }
}

/// Full execution domain and source statements used during preparation.
pub(super) struct TaskDomain {
    pub task: Task,
    pub statements: Vec<Statement>,
}

pub(super) fn collect(plan: &PhysicalPlan) -> Vec<TaskDomain> {
    let mut domains = Vec::new();
    for (owner, statement) in plan.statements().iter().enumerate() {
        partition(
            std::slice::from_ref(statement),
            &[],
            &[owner],
            owner,
            &mut domains,
        );
    }
    domains
}

fn partition(
    statements: &[Statement],
    domain: &[LoopDomain],
    path: &[usize],
    owner: usize,
    out: &mut Vec<TaskDomain>,
) {
    let mut begin = 0;
    while begin < statements.len() {
        let mut nested = path.to_vec();
        nested.push(begin);
        if let Statement::Loop(l) = &statements[begin]
            && l.kind == LoopKind::Parallel
        {
            let mut child = domain.to_vec();
            child.push(l.domain.clone());
            partition(&l.body, &child, &nested, owner, out);
            begin += 1;
        } else {
            let end = (begin + 1..statements.len()).find(|&i| matches!(&statements[i], Statement::Loop(l) if l.kind == LoopKind::Parallel)).unwrap_or(statements.len());
            out.push(TaskDomain {
                task: Task {
                    body: 0,
                    statement: owner,
                    path: nested,
                    domain: domain.to_vec(),
                },
                statements: statements[begin..end].to_vec(),
            });
            begin = end;
        }
    }
}

/// Visit bound coordinates without materializing a coordinate table.
pub(super) fn coordinates(
    domains: &[LoopDomain],
    visit: &mut impl FnMut(&BTreeMap<String, i64>, &BTreeMap<String, i64>) -> Result<(), EmitError>,
) -> Result<(), EmitError> {
    fn walk(
        domains: &[LoopDomain],
        b: &mut BTreeMap<String, i64>,
        s: &mut BTreeMap<String, i64>,
        visit: &mut impl FnMut(&BTreeMap<String, i64>, &BTreeMap<String, i64>) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let Some((d, rest)) = domains.split_first() else {
            return visit(b, s);
        };
        let (start, stop, step) = d.bounds(b).map_err(fail)?;
        s.insert(d.variable.clone(), step);
        for value in (start..stop).step_by(step as usize) {
            b.insert(d.variable.clone(), value);
            walk(rest, b, s, visit)?;
        }
        b.remove(&d.variable);
        s.remove(&d.variable);
        Ok(())
    }
    walk(domains, &mut BTreeMap::new(), &mut BTreeMap::new(), visit)
}
