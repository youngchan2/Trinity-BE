//! Normalization shared by every builder input, before intrinsic validation.
use super::*;
use crate::implementation::{OperationSchedule, ScheduleError};

pub(crate) fn atom(value: impl ToString) -> Expression {
    Expression::Atom(value.to_string())
}
pub(crate) fn expr(op: &str, args: impl IntoIterator<Item = Expression>) -> Expression {
    Expression::List(std::iter::once(atom(op)).chain(args).collect())
}
pub(crate) fn view(plan: &PhysicalPlan, id: ValueInstanceId) -> Expression {
    let value = plan.value_instance(id).unwrap();
    let role = if plan.inputs().iter().any(|b| b.value() == id) {
        "input"
    } else if plan.output().value() == id {
        "output"
    } else {
        "tensor"
    };
    expr(
        "view",
        [
            expr(role, [atom(value.name().unwrap())]),
            expr(
                "layout",
                value
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(axis, size)| expr("axis", [atom(format!("a{axis}")), atom(size)])),
            ),
        ],
    )
}
pub(crate) fn index(parts: impl IntoIterator<Item = Expression>) -> Expression {
    expr(
        "keyed_index",
        parts
            .into_iter()
            .enumerate()
            .map(|(axis, part)| expr("slot", [atom(format!("a{axis}")), part])),
    )
}
pub(crate) fn tile(name: &str, width: usize) -> Expression {
    expr("tile", [atom(name), atom(width)])
}
pub(crate) fn clipped_tile(name: &str, width: usize) -> Expression {
    expr("clipped_tile", [atom(name), atom(width)])
}
pub(crate) fn load(plan: &PhysicalPlan, id: ValueInstanceId, parts: Vec<Expression>) -> Expression {
    expr("load", [view(plan, id), index(parts)])
}
pub(crate) fn store(
    plan: &PhysicalPlan,
    id: ValueInstanceId,
    rhs: Expression,
    parts: Vec<Expression>,
) -> Expression {
    expr("store", [view(plan, id), rhs, index(parts)])
}
pub(crate) fn dimension(
    name: &str,
    stop: usize,
    step: usize,
    kind: LoopKind,
) -> (LoopKind, LoopDomain) {
    (
        kind,
        LoopDomain {
            variable: name.into(),
            start: IndexExpr::Constant(0),
            stop: IndexExpr::Constant(stop as i64),
            step: IndexExpr::Constant(step as i64),
        },
    )
}
pub(crate) fn variable(name: &str) -> IndexExpr {
    IndexExpr::Variable(name.into())
}
pub(crate) fn quotient(name: &str, step: usize) -> IndexExpr {
    IndexExpr::Div(
        Box::new(variable(name)),
        Box::new(IndexExpr::Constant(step as i64)),
    )
}

pub(super) fn normalize(plan: &mut PhysicalPlan) -> Result<(), PhysicalInvariantError> {
    // Assign deterministic internal names in program order. ABI aliases stay in bindings.
    let mut order = Vec::new();
    for binding in plan.inputs.iter() {
        if !order.contains(&binding.value) {
            order.push(binding.value);
        }
    }
    for id in plan.statements.iter().flat_map(Statement::operations) {
        for value in plan.operations.values[id.index()]
            .inputs
            .iter()
            .chain(&plan.operations.values[id.index()].outputs)
        {
            if !order.contains(value) {
                order.push(*value);
            }
        }
    }
    let mut used: std::collections::BTreeSet<String> = plan
        .value_instances
        .iter()
        .filter_map(|(_, v)| v.name.clone())
        .collect();
    for (position, id) in order.into_iter().enumerate() {
        let value = &mut plan.value_instances.values[id.index()];
        if value.name.is_none() {
            let mut name = format!("__value{position}");
            while used.contains(&name) {
                name.push('_');
            }
            used.insert(name.clone());
            value.name = Some(name);
        }
    }
    fn visit(
        plan: &mut PhysicalPlan,
        statement: &mut Statement,
    ) -> Result<(), PhysicalInvariantError> {
        match statement {
            Statement::Loop(l) => {
                for child in &mut l.body {
                    visit(plan, child)?;
                }
            }
            Statement::Operation(id) => {
                let op = &plan.operations.values[id.index()];
                if op.expression.is_some() || !op.coordinates.is_empty() {
                    return Ok(());
                }
                let instance = match op.payload() {
                    OperationPayload::Compute(c) => c.implementation(),
                    OperationPayload::Communication(c) => c.implementation(),
                };
                let OperationSchedule {
                    dimensions,
                    expression,
                    coordinates,
                } = instance
                    .definition()
                    .schedule(plan, *id)
                    .map_err(|e| PhysicalInvariantError::InvalidProgram(e.to_string()))?;
                if matches!(op.payload(), OperationPayload::Compute(_)) && expression.is_none() {
                    return Err(PhysicalInvariantError::InvalidProgram(
                        "compute schedule needs an expression".into(),
                    ));
                }
                let op = &mut plan.operations.values[id.index()];
                op.expression = expression;
                op.coordinates = coordinates;
                for (kind, domain) in dimensions.into_iter().rev() {
                    *statement = Statement::Loop(Loop {
                        kind,
                        domain,
                        body: vec![statement.clone()],
                    });
                }
            }
        }
        Ok(())
    }
    let mut statements = std::mem::take(&mut plan.statements);
    for statement in &mut statements {
        visit(plan, statement)?;
    }
    plan.statements = statements;
    normalize_operands(plan)?;
    Ok(())
}

pub(crate) fn unsupported(message: &str) -> ScheduleError {
    ScheduleError::Unsupported(message.into())
}

/// Bind notation to the declared operands once, so both frontends produce the
/// same ordered operand list. An accumulator's initial destination is implicit.
fn normalize_operands(plan: &mut PhysicalPlan) -> Result<(), PhysicalInvariantError> {
    fn tensor(e: &Expression) -> Option<&str> {
        e.list()?.get(1)?.list()?.get(1)?.atom()
    }
    fn loads(e: &Expression, names: &mut Vec<String>) {
        if e.operator() == Some("load") {
            if let Some(name) = e.list().and_then(|a| a.get(1)).and_then(tensor)
                && !names.iter().any(|n| n == name)
            {
                names.push(name.into());
            }
        } else if let Some(xs) = e.list() {
            for e in &xs[1..] {
                loads(e, names);
            }
        }
    }
    fn has_matmul(e: &Expression) -> bool {
        e.operator() == Some("@") || e.list().is_some_and(|xs| xs.iter().any(has_matmul))
    }
    fn visit(
        plan: &mut PhysicalPlan,
        statements: &[Statement],
        accum: Option<&str>,
    ) -> Result<(), PhysicalInvariantError> {
        let fail = |s: &str| PhysicalInvariantError::InvalidProgram(s.into());
        for statement in statements {
            match statement {
                Statement::Loop(l) => visit(
                    plan,
                    &l.body,
                    if l.kind == LoopKind::Sequential && l.body.len() == 1 {
                        Some(&l.domain.variable)
                    } else {
                        None
                    },
                )?,
                Statement::Operation(id) => {
                    let op = &plan.operations.values[id.index()];
                    if !matches!(op.payload, OperationPayload::Compute(_)) {
                        continue;
                    }
                    let e = op
                        .expression
                        .as_ref()
                        .ok_or_else(|| fail("compute expression is missing"))?;
                    let a = e
                        .list()
                        .filter(|a| a.len() == 4 && a[0].atom() == Some("store"))
                        .ok_or_else(|| fail("expected store expression"))?;
                    if op.outputs.len() != 1 {
                        return Err(fail("store requires one output"));
                    }
                    let name = tensor(&a[1]).ok_or_else(|| fail("invalid output view"))?;
                    if plan.value_instance(op.outputs[0]).and_then(|v| v.name()) != Some(name) {
                        return Err(fail("store expression differs from declared output"));
                    }
                    let accumulator = accum.and_then(|v| super::accumulation_rhs(e, v));
                    let rhs = accumulator.unwrap_or(&a[2]);
                    let mut names = Vec::new();
                    loads(rhs, &mut names);
                    let inputs = names
                        .iter()
                        .map(|name| {
                            plan.value_instances
                                .iter()
                                .find(|(_, v)| v.name() == Some(name.as_str()))
                                .map(|(id, _)| id)
                                .ok_or_else(|| fail("unknown expression operand"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let declared: std::collections::BTreeSet<_> = op
                        .inputs
                        .iter()
                        .copied()
                        .filter(|id| accumulator.is_none() || !op.outputs.contains(id))
                        .collect();
                    if declared != inputs.iter().copied().collect() {
                        return Err(fail("expression loads differ from declared inputs"));
                    }
                    // GEMM's diagnostic tile coordinates come from its output view,
                    // independently of whether the frontend supplied full M/N loops.
                    let coordinates = if has_matmul(rhs) {
                        let layout = a[1]
                            .list()
                            .and_then(|v| v.get(2))
                            .and_then(Expression::list)
                            .ok_or_else(|| fail("invalid output layout"))?;
                        let index = a[3].list().ok_or_else(|| fail("invalid output index"))?;
                        let mut coordinates = Vec::new();
                        for axis in layout.iter().skip(1) {
                            let axis = axis
                                .list()
                                .and_then(|v| v.get(1))
                                .and_then(Expression::atom)
                                .ok_or_else(|| fail("invalid output axis"))?;
                            let part = index.iter().skip(1).find_map(|slot| {
                                let s = slot.list()?;
                                (s.get(1)?.atom() == Some(axis)).then(|| s.get(2)).flatten()
                            });
                            coordinates.push(match part {
                                Some(p) if p.operator() == Some("tile") => quotient(
                                    p.list()
                                        .and_then(|p| p.get(1))
                                        .and_then(Expression::atom)
                                        .ok_or_else(|| fail("invalid tile"))?,
                                    128,
                                ),
                                _ => IndexExpr::Constant(0),
                            });
                        }
                        if coordinates.len() == 3 {
                            coordinates.remove(0);
                        }
                        coordinates
                    } else {
                        op.coordinates.clone()
                    };
                    let op = &mut plan.operations.values[id.index()];
                    op.inputs = inputs;
                    op.coordinates = coordinates;
                }
            }
        }
        Ok(())
    }
    visit(plan, &plan.statements.clone(), None)
}
