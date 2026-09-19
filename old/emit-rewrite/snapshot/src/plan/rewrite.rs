//! Transactional fusion of adjacent task sets; serial scopes remain in the tree.
use std::collections::BTreeMap;

use super::{
    Expression, LoopKind, PhysicalInvariantError, PhysicalPlan, PhysicalPlanBuilder, Statement,
};
use crate::FusionRewrite;

pub(crate) fn rewrite_statements(
    plan: &PhysicalPlan,
    producer: usize,
    proposal: &FusionRewrite,
) -> Result<Option<PhysicalPlan>, PhysicalInvariantError> {
    let mut builder = PhysicalPlanBuilder {
        target: plan.target,
        world_size: plan.world_size,
        inputs: plan.inputs.to_vec(),
        values: plan.value_instances.clone(),
        operations: plan.operations.clone(),
        statements: plan.statements.clone(),
    };
    let mut left = builder.statements[producer].clone();
    let mut right = builder.statements[producer + 1].clone();
    let right_ids = right.operations();
    let pointwise = right_ids.len() == 1 && crate::fusion::pointwise(plan, right_ids[0]);
    if pointwise {
        // A pointwise consumer adopts the producer's tile and fragment traversal.
        // Its original loads must address precisely its own output coordinates.
        let Some(last) = left.operations().last().copied() else {
            return Ok(None);
        };
        let Some(index) = plan
            .operation(last)
            .and_then(|op| op.expression())
            .and_then(Expression::list)
            .and_then(|xs| xs.get(3))
            .cloned()
        else {
            return Ok(None);
        };
        let op = &mut builder.operations.values[right_ids[0].index()];
        fn retile(e: &mut Expression, index: &Expression) {
            if let Expression::List(xs) = e {
                if matches!(
                    xs.first().and_then(Expression::atom),
                    Some("load" | "store")
                ) {
                    *xs.last_mut().unwrap() = index.clone();
                }
                for child in xs {
                    retile(child, index);
                }
            }
        }
        retile(op.expression.as_mut().unwrap(), &index);
        op.coordinates = plan.operation(last).unwrap().coordinates().to_vec();
        fn append(left: &mut Statement, right: Statement) -> bool {
            let Statement::Loop(l) = left else {
                return false;
            };
            if l.kind != LoopKind::Parallel {
                return false;
            }
            if l.body.len() == 1
                && matches!(&l.body[0], Statement::Loop(n) if n.kind == LoopKind::Parallel)
            {
                append(&mut l.body[0], right)
            } else {
                l.body.push(right);
                true
            }
        }
        if !append(&mut left, Statement::Operation(right_ids[0])) {
            return Ok(None);
        }
    } else {
        // Fuse equal parallel domains and retain the complete sequential bodies.
        fn merge(
            left: &mut Statement,
            right: &mut Statement,
            names: &mut BTreeMap<String, String>,
        ) -> bool {
            let (Statement::Loop(a), Statement::Loop(b)) = (left, right) else {
                return false;
            };
            if a.kind != LoopKind::Parallel || b.kind != LoopKind::Parallel {
                return false;
            }
            b.domain.start.rename(names);
            b.domain.stop.rename(names);
            b.domain.step.rename(names);
            if a.domain.start != b.domain.start
                || a.domain.stop != b.domain.stop
                || a.domain.step != b.domain.step
            {
                return false;
            }
            names.insert(b.domain.variable.clone(), a.domain.variable.clone());
            if a.body.len() == 1
                && b.body.len() == 1
                && matches!(&a.body[0], Statement::Loop(l) if l.kind == LoopKind::Parallel)
                && matches!(&b.body[0], Statement::Loop(l) if l.kind == LoopKind::Parallel)
            {
                merge(&mut a.body[0], &mut b.body[0], names)
            } else {
                if a.body
                    .iter()
                    .chain(&b.body)
                    .any(|s| matches!(s, Statement::Loop(l) if l.kind == LoopKind::Parallel))
                {
                    return false;
                }
                fn rename(s: &mut Statement, names: &BTreeMap<String, String>) {
                    if let Statement::Loop(l) = s {
                        l.domain.start.rename(names);
                        l.domain.stop.rename(names);
                        l.domain.step.rename(names);
                        for s in &mut l.body {
                            rename(s, names);
                        }
                    }
                }
                for s in &mut b.body {
                    rename(s, names);
                }
                a.body.extend(b.body.iter().cloned());
                true
            }
        }
        let mut names = BTreeMap::new();
        if !merge(&mut left, &mut right, &mut names) {
            return Ok(None);
        }
        for id in right_ids {
            let op = &mut builder.operations.values[id.index()];
            if let Some(e) = &mut op.expression {
                e.rename_indices(&names);
            }
            for c in &mut op.coordinates {
                c.rename(&names);
            }
        }
    }
    builder.statements[producer] = left;
    builder.statements.remove(producer + 1);
    for &(value, storage) in proposal.storage_updates() {
        builder.values.values[value.index()].storage = storage;
    }
    builder
        .finalize(plan.output.tensor.clone(), plan.output.value)
        .map(Some)
}
