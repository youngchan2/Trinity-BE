//! Name, operand, and ID normalization for physical plans.
use super::{
    Expression, IdVec, LoopKind, OperationId, PhysicalInvariantError, PhysicalPlan, Statement,
    ValueInstanceId,
};
use std::collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

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
            .inflows
            .iter()
            .chain(&plan.operations.values[id.index()].outflows)
        {
            if !order.contains(value) {
                order.push(*value);
            }
        }
    }
    // Providers may inspect every declared value, including unused scratch.
    for (id, _) in plan.value_instances.iter() {
        if !order.contains(&id) {
            order.push(id);
        }
    }

    let mut used: BTreeSet<String> = plan
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

    normalize_operands(plan)
}

/// Bind notation to the declared operands once, so both frontends produce the
/// same ordered operand list. Recognized reductions start at zero; their initial
/// destination load is not an inflow.
fn normalize_operands(plan: &mut PhysicalPlan) -> Result<(), PhysicalInvariantError> {
    fn visit(
        plan: &mut PhysicalPlan,
        statements: &[Statement],
        accum: Option<&str>,
        explicit_initialization: bool,
    ) -> Result<(), PhysicalInvariantError> {
        let fail = |s: &str| PhysicalInvariantError::InvalidProgram(s.into());
        for statement in statements {
            match statement {
                Statement::Region(body) => visit(plan, body, None, true)?,
                Statement::Loop(l) => visit(
                    plan,
                    &l.body,
                    if !explicit_initialization
                        && l.kind == LoopKind::Sequential
                        && l.body.len() == 1
                    {
                        Some(&l.domain.variable)
                    } else {
                        None
                    },
                    explicit_initialization,
                )?,
                Statement::Operation(id) => {
                    let op = &plan.operations.values[id.index()];
                    let e = &op.expression;
                    let (destination, rhs, accumulator) = match e {
                        Expression::Store { destination, value } if value.is_pure() => {
                            let accumulator = accum.and_then(|v| super::accumulation_rhs(e, v));
                            (
                                destination,
                                accumulator.unwrap_or(value),
                                accumulator.is_some(),
                            )
                        }
                        Expression::AllGather { destination, .. } => (destination, e, false),
                        _ => {
                            return Err(fail("expected store with pure computation or all_gather"));
                        }
                    };
                    if op.outflows.as_slice() != [destination.value] {
                        return Err(fail("expression destination differs from declared outflow"));
                    }
                    let inflows: Vec<_> = rhs
                        .reads()
                        .into_iter()
                        .filter(|id| !op.zero_init.contains(id))
                        .collect();

                    let declared: BTreeSet<_> = op
                        .inflows
                        .iter()
                        .copied()
                        .filter(|id| {
                            (!accumulator || !op.outflows.contains(id))
                                && !op.zero_init.contains(id)
                        })
                        .collect();

                    if declared != inflows.iter().copied().collect() {
                        return Err(fail("expression reads differ from declared inflows"));
                    }

                    let op = &mut plan.operations.values[id.index()];
                    op.inflows = inflows;
                }
            }
        }

        Ok(())
    }

    visit(plan, &plan.statements.clone(), None, false)
}

/// Canonicalize declaration IDs without reordering executable statements.
pub(super) fn canonicalize(mut plan: PhysicalPlan) -> PhysicalPlan {
    let symbols = plan.symbols();
    let order: Vec<_> = plan
        .statements
        .iter()
        .flat_map(Statement::operations)
        .collect();
    let mut value_order = Vec::new();
    let mut value_ids = vec![usize::MAX; plan.value_instances.len()];
    let assign = |id: ValueInstanceId, order: &mut Vec<usize>, ids: &mut [usize]| {
        if ids[id.index()] == usize::MAX {
            ids[id.index()] = order.len();
            order.push(id.index());
        }
    };
    for binding in &plan.inputs {
        assign(binding.value, &mut value_order, &mut value_ids);
    }
    for id in &order {
        let op = &plan.operations.values[id.index()];
        for &value in op.inflows.iter().chain(&op.outflows) {
            assign(value, &mut value_order, &mut value_ids);
        }
    }
    for i in 0..plan.value_instances.len() {
        assign(
            ValueInstanceId::from_index(i),
            &mut value_order,
            &mut value_ids,
        );
    }
    let mut operation_ids = vec![0; plan.operations.len()];
    let mut operations = Vec::new();
    for &id in &order {
        operation_ids[id.index()] = operations.len();
        let mut op = plan.operations.values[id.index()].clone();
        for value in op
            .inflows
            .iter_mut()
            .chain(&mut op.outflows)
            .chain(&mut op.zero_init)
        {
            *value = ValueInstanceId::from_index(value_ids[value.index()]);
        }
        op.expression.remap_values(&value_ids);
        operations.push(op);
    }
    fn visit(
        statement: &mut Statement,
        ids: &[usize],
        operations: &mut [super::Operation],
        names: &BTreeMap<String, String>,
        next: &mut usize,
        symbols: &BTreeSet<String>,
    ) {
        match statement {
            Statement::Region(body) => {
                for statement in body {
                    visit(statement, ids, operations, names, next, symbols);
                }
            }
            Statement::Operation(id) => {
                *id = OperationId::from_index(ids[id.index()]);
                let expr = &mut operations[id.index()].expression;
                expr.rename_indices(names);
            }
            Statement::Loop(l) => {
                l.domain.start.rename(names);
                l.domain.stop.rename(names);
                l.domain.step.rename(names);
                let mut nested = names.clone();
                while symbols.contains(&format!("lv{}", *next)) {
                    *next += 1;
                }
                let variable = format!("lv{}", *next);
                *next += 1;
                nested.insert(l.domain.variable.clone(), variable.clone());
                l.domain.variable = variable;
                for statement in &mut l.body {
                    visit(statement, ids, operations, &nested, next, symbols);
                }
            }
        }
    }
    let mut next = 0;
    for statement in &mut plan.statements {
        visit(
            statement,
            &operation_ids,
            &mut operations,
            &BTreeMap::new(),
            &mut next,
            &symbols,
        );
    }
    plan.operations = IdVec::from_values(operations);
    plan.value_instances = IdVec::from_values(
        value_order
            .iter()
            .map(|&i| plan.value_instances.values[i].clone())
            .collect(),
    );
    for input in &mut plan.inputs {
        input.value = ValueInstanceId::from_index(value_ids[input.value.index()]);
    }
    for output in &mut plan.outputs {
        output.value = ValueInstanceId::from_index(value_ids[output.value.index()]);
    }
    plan.mutable_inputs = plan
        .mutable_inputs
        .iter()
        .map(|v| ValueInstanceId::from_index(value_ids[v.index()]))
        .collect();
    plan.hash = hash_plan(&plan);
    plan
}

pub(crate) fn hash_plan(plan: &PhysicalPlan) -> u64 {
    let mut hasher = DefaultHasher::new();
    plan.target.hash(&mut hasher);
    plan.world_size.hash(&mut hasher);
    plan.inputs.hash(&mut hasher);
    plan.value_instances.hash(&mut hasher);
    plan.operations.hash(&mut hasher);
    plan.statements.hash(&mut hasher);
    plan.outputs.hash(&mut hasher);
    plan.mutable_inputs.hash(&mut hasher);
    plan.bindings.hash(&mut hasher);
    hasher.finish()
}
