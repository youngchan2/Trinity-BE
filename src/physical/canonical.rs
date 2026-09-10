use super::plan::{IdVec, OperationId, PhysicalPlan, Statement, ValueInstanceId};
use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

/// Canonicalize declaration IDs without reordering executable statements.
pub(super) fn canonicalize(mut plan: PhysicalPlan) -> PhysicalPlan {
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
        for &value in op.inputs.iter().chain(&op.outputs) {
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
        for value in op.inputs.iter_mut().chain(&mut op.outputs) {
            *value = ValueInstanceId::from_index(value_ids[value.index()]);
        }
        operations.push(op);
    }
    fn visit(
        statement: &mut Statement,
        ids: &[usize],
        operations: &mut [super::Operation],
        names: &BTreeMap<String, String>,
        next: &mut usize,
    ) {
        match statement {
            Statement::Operation(id) => {
                *id = OperationId::from_index(ids[id.index()]);
                for coordinate in &mut operations[id.index()].coordinates {
                    coordinate.rename(names);
                }
                if let Some(expr) = &mut operations[id.index()].expression {
                    expr.rename_indices(names);
                    expr.normalize_axes();
                }
            }
            Statement::Loop(l) => {
                l.domain.start.rename(names);
                l.domain.stop.rename(names);
                l.domain.step.rename(names);
                let mut nested = names.clone();
                let variable = format!("lv{}", *next);
                *next += 1;
                nested.insert(l.domain.variable.clone(), variable.clone());
                l.domain.variable = variable;
                for statement in &mut l.body {
                    visit(statement, ids, operations, &nested, next);
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
    plan.output.value = ValueInstanceId::from_index(value_ids[plan.output.value.index()]);
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
    plan.output.hash(&mut hasher);
    hasher.finish()
}
