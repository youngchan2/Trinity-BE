use super::normalize::{canonicalize, normalize};
use super::{
    Expression, IdVec, Operation, OperationId, PhysicalInvariantError, PhysicalPlan, Statement,
    Storage, TensorBinding, ValueInstance, ValueInstanceId,
};
use crate::{DType, TargetCapability};
use std::collections::BTreeSet;

/// Registers values and operations for a physical plan.
///
/// `build` takes the ordered statements, normalizes names and operands,
/// validates the program, and canonicalizes IDs.
/// Callers supply loops and computation expressions; `build` does not generate them.
/// Callers are responsible for valid loop ranges and memory accesses.
/// IDs belong to this builder (or its clones) and may be remapped by `build`.
#[derive(Clone)]
pub struct PhysicalPlanBuilder {
    pub(super) target: TargetCapability,
    pub(super) world_size: usize,
    pub(super) inputs: Vec<TensorBinding>,
    pub(super) values: IdVec<ValueInstanceId, ValueInstance>,
    pub(super) operations: IdVec<OperationId, Operation>,
}

impl PhysicalPlanBuilder {
    pub fn new(target: TargetCapability, world_size: usize) -> Self {
        Self {
            target,
            world_size,
            inputs: Vec::new(),
            values: IdVec::new(),
            operations: IdVec::new(),
        }
    }

    /// Adds a [`ValueInstance`] with the given dtype, shape, and storage.
    /// Returns its ID for use in operation operands and tensor bindings.
    pub fn add_value(
        &mut self,
        dtype: DType,
        shape: impl IntoIterator<Item = usize>,
        storage: Storage,
    ) -> ValueInstanceId {
        self.values.push(ValueInstance::new(dtype, shape, storage))
    }

    /// Registers an operation with its expression and returns its ID.
    /// The expression describes computation or communication; Emit selects its implementation.
    pub fn add_operation(
        &mut self,
        inflows: impl IntoIterator<Item = ValueInstanceId>,
        outflows: impl IntoIterator<Item = ValueInstanceId>,
        expression: Expression,
    ) -> OperationId {
        self.operations
            .push(Operation::new(inflows, outflows, expression))
    }

    pub fn add_named_value(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        shape: impl IntoIterator<Item = usize>,
        storage: Storage,
    ) -> ValueInstanceId {
        let id = self.add_value(dtype, shape, storage);
        self.values.values[id.index()].name = Some(name.into());

        id
    }

    pub fn bind_input(&mut self, tensor: impl Into<String>, value: ValueInstanceId) {
        self.inputs.push(TensorBinding::new(tensor, value));
    }

    /// Builds a plan with the given top-level statements in execution order.
    /// Normalizes names and operands, validates the program, and canonicalizes IDs.
    ///
    /// Query the returned plan's bindings and graph for canonical IDs rather than
    /// reusing IDs obtained before building the plan.
    pub fn build(
        self,
        statements: Vec<Statement>,
        output_tensor: impl Into<String>,
        output_value: ValueInstanceId,
    ) -> Result<PhysicalPlan, PhysicalInvariantError> {
        self.build_program(
            statements,
            vec![TensorBinding::new(output_tensor, output_value)],
            Default::default(),
            Default::default(),
        )
    }

    pub(super) fn build_program(
        self,
        statements: Vec<Statement>,
        outputs: Vec<TensorBinding>,
        mutable_inputs: BTreeSet<ValueInstanceId>,
        bindings: std::collections::BTreeMap<String, i64>,
    ) -> Result<PhysicalPlan, PhysicalInvariantError> {
        let PhysicalPlanBuilder {
            target,
            world_size,
            mut inputs,
            values,
            mut operations,
        } = self;

        validate_world_size(world_size)?;
        for output in &outputs {
            validate_name(&output.tensor)?;
            validate_value_id(output.value, values.len(), "output binding")?;
        }
        validate_membership(&statements, operations.len())?;

        let (input_values, _) = validate_inputs(&inputs, values.len())?;

        for (_, op) in operations.iter() {
            for access in op.expression.accesses() {
                validate_value_id(access.value, values.len(), "expression operand")?;
                access
                    .validate_view(values.values[access.value.index()].shape())
                    .map_err(PhysicalInvariantError::InvalidProgram)?;
            }
            for &id in op.inflows.iter().chain(&op.outflows) {
                validate_value_id(id, values.len(), "operation operand")?;
            }

            for id in &op.outflows {
                if input_values.contains(id) && !mutable_inputs.contains(id) {
                    return Err(PhysicalInvariantError::BoundaryInputHasProducer {
                        value: id.index(),
                    });
                }
            }
        }

        // An explicit base-shape view and the default view are the same access.
        for op in &mut operations.values {
            op.expression.map_accesses(&mut |access| {
                if access.view_shape.as_deref() == Some(values.values[access.value.index()].shape())
                {
                    access.view_shape = None;
                }
            });
        }

        inputs.sort_by(|a, b| a.tensor.cmp(&b.tensor));

        let mut plan = PhysicalPlan {
            target,
            world_size,
            inputs: inputs.into_boxed_slice(),
            value_instances: values,
            operations,
            statements,
            outputs: outputs.into_boxed_slice(),
            mutable_inputs,
            bindings,
            hash: 0,
        };

        normalize(&mut plan)?;
        validate_program(&plan, &input_values)?;

        Ok(canonicalize(plan))
    }
}

fn validate_world_size(world_size: usize) -> Result<(), PhysicalInvariantError> {
    if world_size == 0 {
        Err(PhysicalInvariantError::InvalidWorldSize)
    } else {
        Ok(())
    }
}

fn validate_program(
    plan: &PhysicalPlan,
    inputs: &BTreeSet<ValueInstanceId>,
) -> Result<(), PhysicalInvariantError> {
    let fail = |s: &str| PhysicalInvariantError::InvalidProgram(s.into());

    for (id, value) in plan.value_instances.iter() {
        let boundary = inputs.contains(&id) || plan.outputs.iter().any(|o| o.value == id);

        // Input and output bindings must refer to values in External storage.
        if boundary && value.storage != Storage::External {
            return Err(PhysicalInvariantError::InvalidBoundaryStorage { value: id.index() });
        }

        // Every External value must have an input or output binding.
        if !boundary && value.storage == Storage::External {
            return Err(PhysicalInvariantError::UnboundExternalValue { value: id.index() });
        }

        // Shared and Register values must stay within one top-level statement.
        validate_storage_scope(plan, id, value.storage)?;
    }

    let mut available = inputs.clone();

    validate_statements(plan, &plan.statements, &mut available, &mut BTreeSet::new())?;

    if plan.outputs.iter().any(|o| !available.contains(&o.value)) {
        return Err(fail("output has no producer"));
    }

    Ok(())
}

fn validate_statements(
    plan: &PhysicalPlan,
    statements: &[Statement],
    available: &mut BTreeSet<ValueInstanceId>,
    scope: &mut BTreeSet<String>,
) -> Result<(), PhysicalInvariantError> {
    for statement in statements {
        match statement {
            Statement::Region(body) => validate_statements(plan, body, available, scope)?,
            Statement::Loop(l) => {
                validate_loop(plan, l, available, scope)?;
            }
            Statement::Operation(id) => {
                let op = &plan.operations.values[id.index()];
                validate_operation(op, available, scope)?;
            }
        }
    }
    Ok(())
}

fn validate_loop(
    plan: &PhysicalPlan,
    l: &super::Loop,
    available: &mut BTreeSet<ValueInstanceId>,
    scope: &mut BTreeSet<String>,
) -> Result<(), PhysicalInvariantError> {
    let fail = |s: &str| PhysicalInvariantError::InvalidProgram(s.into());

    if l.body.is_empty() {
        return Err(fail("empty Loop body"));
    }

    if !scope.insert(l.domain.variable.clone()) {
        return Err(fail("shadowed loop variable"));
    }

    validate_statements(plan, &l.body, available, scope)?;
    scope.remove(&l.domain.variable);

    Ok(())
}

fn validate_operation(
    op: &super::Operation,
    available: &mut BTreeSet<ValueInstanceId>,
    scope: &BTreeSet<String>,
) -> Result<(), PhysicalInvariantError> {
    for access in op.expression.accesses() {
        for index in &access.indices {
            if let super::AccessIndex::Slice { width, .. }
            | super::AccessIndex::Tile { width, .. }
            | super::AccessIndex::ClippedTile { width, .. } = index
            {
                if let super::TileWidth::Symbol(name) = width {
                    if !super::expression::is_symbol(name) || scope.contains(name) {
                        return Err(PhysicalInvariantError::InvalidProgram(format!(
                            "tile width {name} must name a configuration symbol, not a loop coordinate"
                        )));
                    }
                } else {
                    width
                        .resolve(&Default::default())
                        .map_err(PhysicalInvariantError::InvalidProgram)?;
                }
            }
        }
        for variable in access
            .indices
            .iter()
            .filter_map(super::AccessIndex::variable)
        {
            if !scope.contains(variable) {
                return Err(PhysicalInvariantError::InvalidProgram(format!(
                    "unbound index {variable}"
                )));
            }
        }
    }

    for inflow in &op.inflows {
        if !available.contains(inflow) && !op.zero_init.contains(inflow) {
            return Err(PhysicalInvariantError::MissingProducer {
                value: inflow.index(),
            });
        }
    }

    available.extend(op.outflows.iter().copied());

    Ok(())
}

fn validate_storage_scope(
    plan: &PhysicalPlan,
    value: ValueInstanceId,
    storage: Storage,
) -> Result<(), PhysicalInvariantError> {
    // Only Shared and Register values are restricted to one top-level statement.
    if !matches!(storage, Storage::Shared | Storage::Register) {
        return Ok(());
    }

    // Find top-level statements that read or write this value, including nested operations.
    let mut owners = plan.statements.iter().filter(|statement| {
        statement.operations().iter().any(|id| {
            let op = &plan.operations.values[id.index()];
            op.inflows.contains(&value) || op.outflows.contains(&value)
        })
    });

    // A second owner crosses the allowed scope; nth(1) stops at that second match.
    if owners.nth(1).is_some() {
        Err(PhysicalInvariantError::CrossStatementStorage {
            value: value.index(),
            storage,
        })
    } else {
        Ok(())
    }
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
