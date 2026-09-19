use super::builder::PhysicalPlanBuilder;
use super::canonical::canonicalize;
use super::error::PhysicalInvariantError;
use super::types::{
    OperationPayload, PhysicalPlan, Statement, Storage, TensorBinding, ValueInstanceId,
};
use std::collections::BTreeSet;

pub(super) fn finalize(
    branch: PhysicalPlanBuilder,
    output: TensorBinding,
) -> Result<PhysicalPlan, PhysicalInvariantError> {
    let PhysicalPlanBuilder {
        target,
        world_size,
        mut inputs,
        values,
        operations,
        statements,
    } = branch;
    if world_size == 0 {
        return Err(PhysicalInvariantError::InvalidWorldSize);
    }
    validate_name(&output.tensor)?;
    validate_value_id(output.value, values.len(), "output binding")?;
    let (input_values, _) = validate_inputs(&inputs, values.len())?;
    validate_membership(&statements, operations.len())?;
    for (_, op) in operations.iter() {
        for &id in op.inputs.iter().chain(&op.outputs) {
            validate_value_id(id, values.len(), "operation operand")?;
        }
        for id in &op.outputs {
            if input_values.contains(id) {
                return Err(PhysicalInvariantError::BoundaryInputHasProducer { value: id.index() });
            }
        }
    }
    inputs.sort_by(|a, b| a.tensor.cmp(&b.tensor));
    let mut plan = PhysicalPlan {
        target,
        world_size,
        inputs: inputs.into_boxed_slice(),
        value_instances: values,
        operations,
        statements,
        output,
        hash: 0,
    };
    super::normalize::normalize(&mut plan)?;
    validate_program(&plan, &input_values)?;
    Ok(canonicalize(plan))
}

fn validate_program(
    plan: &PhysicalPlan,
    inputs: &BTreeSet<ValueInstanceId>,
) -> Result<(), PhysicalInvariantError> {
    let fail = |s: &str| PhysicalInvariantError::InvalidProgram(s.into());
    for (id, value) in plan.value_instances.iter() {
        let boundary = inputs.contains(&id) || id == plan.output.value;
        if boundary && value.storage != Storage::External {
            return Err(PhysicalInvariantError::InvalidBoundaryStorage { value: id.index() });
        }
        if !boundary && value.storage == Storage::External {
            return Err(PhysicalInvariantError::UnboundExternalValue { value: id.index() });
        }
        if matches!(value.storage, Storage::Shared | Storage::Register) {
            let owners: BTreeSet<_> = plan
                .statements
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    s.operations().iter().any(|op| {
                        let op = &plan.operations.values[op.index()];
                        op.inputs.contains(&id) || op.outputs.contains(&id)
                    })
                })
                .map(|(i, _)| i)
                .collect();
            if owners.len() > 1 {
                return Err(PhysicalInvariantError::CrossStatementStorage {
                    value: id.index(),
                    storage: value.storage,
                });
            }
        }
    }
    fn visit(
        plan: &PhysicalPlan,
        statements: &[Statement],
        available: &mut BTreeSet<ValueInstanceId>,
        scope: &mut BTreeSet<String>,
        accumulation: Option<&str>,
    ) -> Result<(), PhysicalInvariantError> {
        let fail = |s: &str| PhysicalInvariantError::InvalidProgram(s.into());
        for statement in statements {
            match statement {
                Statement::Loop(l) => {
                    if l.body.is_empty() {
                        return Err(fail("empty Loop body"));
                    }
                    if !scope.insert(l.domain.variable.clone()) {
                        return Err(fail("shadowed loop variable"));
                    }
                    let accumulation = if l.kind == super::LoopKind::Sequential && l.body.len() == 1
                    {
                        Some(l.domain.variable.as_str())
                    } else {
                        None
                    };
                    visit(plan, &l.body, available, scope, accumulation)?;
                    scope.remove(&l.domain.variable);
                }
                Statement::Operation(id) => {
                    let op = &plan.operations.values[id.index()];
                    let accumulator = if matches!(op.payload(), OperationPayload::Compute(_)) {
                        let e = op
                            .expression
                            .as_ref()
                            .ok_or_else(|| fail("compute operation needs expression"))?;
                        let a = e.list().ok_or_else(|| fail("expected store expression"))?;
                        if e.operator() != Some("store") || a.len() != 4 || op.outputs.len() != 1 {
                            return Err(fail("operation must contain one store"));
                        }
                        accumulation
                            .is_some_and(|var| super::loops::accumulation_rhs(e, var).is_some())
                    } else {
                        false
                    };
                    for input in &op.inputs {
                        if !(available.contains(input) || accumulator && op.outputs.contains(input))
                        {
                            return Err(PhysicalInvariantError::MissingProducer {
                                value: input.index(),
                            });
                        }
                    }
                    available.extend(op.outputs.iter().copied());
                }
            }
        }
        Ok(())
    }
    let mut available = inputs.clone();
    visit(
        plan,
        &plan.statements,
        &mut available,
        &mut BTreeSet::new(),
        None,
    )?;
    if !available.contains(&plan.output.value) {
        return Err(fail("output has no producer"));
    }
    Ok(())
}

fn validate_inputs(
    inputs: &[TensorBinding],
    value_count: usize,
) -> Result<(BTreeSet<ValueInstanceId>, BTreeSet<String>), PhysicalInvariantError> {
    let mut values = BTreeSet::new();
    let mut names = BTreeSet::new();
    for input in inputs {
        validate_name(&input.tensor)?;
        validate_value_id(input.value, value_count, "input binding")?;
        if !names.insert(input.tensor.clone()) {
            return Err(PhysicalInvariantError::DuplicateInputTensor {
                tensor: input.tensor.clone(),
            });
        }

        // Multiple names may designate one canonical value. Runtime binding
        // validation requires all aliases to reference exactly the same region.
        values.insert(input.value);
    }
    Ok((values, names))
}

fn validate_name(name: &str) -> Result<(), PhysicalInvariantError> {
    if name.is_empty() {
        Err(PhysicalInvariantError::EmptyTensorName)
    } else {
        Ok(())
    }
}

fn validate_membership(
    statements: &[Statement],
    operation_count: usize,
) -> Result<Vec<usize>, PhysicalInvariantError> {
    let mut membership = vec![None; operation_count];
    for (statement_index, statement) in statements.iter().enumerate() {
        let operations = statement.operations();
        if operations.is_empty() {
            return Err(PhysicalInvariantError::EmptyStatement {
                statement: statement_index,
            });
        }
        let mut seen = BTreeSet::new();
        for operation in operations {
            if operation.index() >= operation_count {
                return Err(PhysicalInvariantError::InvalidOperationId {
                    operation: operation.index(),
                });
            }
            if !seen.insert(operation) {
                return Err(PhysicalInvariantError::DuplicateOperationInStatement {
                    statement: statement_index,
                    operation: operation.index(),
                });
            }
            if membership[operation.index()]
                .replace(statement_index)
                .is_some()
            {
                return Err(PhysicalInvariantError::DuplicateStatementMembership {
                    operation: operation.index(),
                });
            }
        }
    }
    membership
        .into_iter()
        .enumerate()
        .map(|(operation, statement)| {
            statement.ok_or(PhysicalInvariantError::MissingStatementMembership { operation })
        })
        .collect()
}

fn validate_value_id(
    value: ValueInstanceId,
    value_count: usize,
    context: &'static str,
) -> Result<(), PhysicalInvariantError> {
    if value.index() < value_count {
        Ok(())
    } else {
        Err(PhysicalInvariantError::InvalidValueId {
            value: value.index(),
            context,
        })
    }
}
