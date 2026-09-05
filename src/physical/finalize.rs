use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::builder::PhysicalPlanBuilder;
use super::canonical::canonicalize;
use super::error::PhysicalInvariantError;
use super::plan::{
    Action, ActionId, IdVec, Operation, OperationId, OperationPayload, PhysicalPlan, Storage,
    TensorBinding, ValueInstance, ValueInstanceId,
};

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
        action_operations,
    } = branch;

    if world_size == 0 {
        return Err(PhysicalInvariantError::InvalidWorldSize);
    }

    validate_name(&output.tensor)?;
    validate_value_id(output.value, values.len(), "output binding")?;

    let (input_values, input_names) = validate_inputs(&inputs, values.len())?;
    let (producers, consumers) = validate_edges(&operations, values.len())?;
    validate_unique_operations(&operations, &values.values)?;
    validate_boundaries(&values.values, &input_values, output.value, &producers)?;
    ensure_operation_dag(&operations.values, &producers)?;

    let operation_actions = validate_membership(&action_operations, operations.len())?;
    ensure_action_dag(
        &operations.values,
        &producers,
        &operation_actions,
        action_operations.len(),
    )?;
    validate_cross_action_storage(&values.values, &producers, &consumers, &operation_actions)?;

    let actions = derive_actions(
        &action_operations.values,
        &operations.values,
        &producers,
        &consumers,
        &operation_actions,
        output.value,
    );
    inputs.sort_by(|lhs, rhs| lhs.tensor.cmp(&rhs.tensor));
    debug_assert_eq!(inputs.len(), input_names.len());

    let plan = PhysicalPlan {
        target,
        world_size,
        inputs: inputs.into_boxed_slice(),
        value_instances: values,
        operations,
        actions: IdVec::from_values(actions),
        output,
        hash: 0,
    };
    Ok(canonicalize(plan, &producers, &operation_actions))
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
        if !values.insert(input.value) {
            return Err(PhysicalInvariantError::DuplicateInputValue {
                value: input.value.index(),
            });
        }
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

fn validate_edges(
    operations: &IdVec<OperationId, Operation>,
    value_count: usize,
) -> Result<ProducerConsumerMap, PhysicalInvariantError> {
    let mut producers = vec![None; value_count];
    let mut consumers = vec![Vec::new(); value_count];

    for (operation_id, operation) in operations.iter() {
        for (input_index, input) in operation.inputs.iter().copied().enumerate() {
            validate_value_id(input, value_count, "operation input")?;
            consumers[input.index()].push((operation_id, input_index));
        }
        for (output_index, output) in operation.outputs.iter().copied().enumerate() {
            validate_value_id(output, value_count, "operation output")?;
            if producers[output.index()]
                .replace((operation_id, output_index))
                .is_some()
            {
                return Err(PhysicalInvariantError::DuplicateProducer {
                    value: output.index(),
                });
            }
        }
    }
    Ok((producers, consumers))
}

type ProducerConsumerMap = (
    Vec<Option<(OperationId, usize)>>,
    Vec<Vec<(OperationId, usize)>>,
);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OperationSignature {
    payload: OperationPayload,
    inputs: Vec<ValueInstanceId>,
    outputs: Vec<ValueInstance>,
}

fn validate_unique_operations(
    operations: &IdVec<OperationId, Operation>,
    values: &[ValueInstance],
) -> Result<(), PhysicalInvariantError> {
    let mut seen = BTreeMap::new();
    for (operation_id, operation) in operations.iter() {
        let signature = OperationSignature {
            payload: operation.payload.clone(),
            inputs: operation.inputs.clone(),
            outputs: operation
                .outputs
                .iter()
                .map(|output| values[output.index()].clone())
                .collect(),
        };
        if let Some(first) = seen.insert(signature, operation_id) {
            return Err(PhysicalInvariantError::DuplicateOperation {
                first: first.index(),
                second: operation_id.index(),
            });
        }
    }
    Ok(())
}

fn validate_boundaries(
    values: &[ValueInstance],
    inputs: &BTreeSet<ValueInstanceId>,
    output: ValueInstanceId,
    producers: &[Option<(OperationId, usize)>],
) -> Result<(), PhysicalInvariantError> {
    for (index, value) in values.iter().enumerate() {
        let id = ValueInstanceId::from_index(index);
        let is_input = inputs.contains(&id);
        let is_output = id == output;
        if is_input && producers[index].is_some() {
            return Err(PhysicalInvariantError::BoundaryInputHasProducer { value: index });
        }
        if !is_input && producers[index].is_none() {
            return Err(PhysicalInvariantError::MissingProducer { value: index });
        }
        if (is_input || is_output) && value.storage != Storage::External {
            return Err(PhysicalInvariantError::InvalidBoundaryStorage { value: index });
        }
        if value.storage == Storage::External && !is_input && !is_output {
            return Err(PhysicalInvariantError::UnboundExternalValue { value: index });
        }
    }
    Ok(())
}

fn validate_membership(
    actions: &IdVec<ActionId, Vec<OperationId>>,
    operation_count: usize,
) -> Result<Vec<ActionId>, PhysicalInvariantError> {
    let mut membership = vec![None; operation_count];
    for (action_id, operations) in actions.iter() {
        if operations.is_empty() {
            return Err(PhysicalInvariantError::EmptyAction {
                action: action_id.index(),
            });
        }
        let mut seen = BTreeSet::new();
        for operation in operations {
            if operation.index() >= operation_count {
                return Err(PhysicalInvariantError::InvalidOperationId {
                    operation: operation.index(),
                });
            }
            if !seen.insert(*operation) {
                return Err(PhysicalInvariantError::DuplicateOperationInAction {
                    action: action_id.index(),
                    operation: operation.index(),
                });
            }
            if membership[operation.index()].replace(action_id).is_some() {
                return Err(PhysicalInvariantError::DuplicateActionMembership {
                    operation: operation.index(),
                });
            }
        }
    }
    membership
        .into_iter()
        .enumerate()
        .map(|(operation, action)| {
            action.ok_or(PhysicalInvariantError::MissingActionMembership { operation })
        })
        .collect()
}

fn ensure_operation_dag(
    operations: &[Operation],
    producers: &[Option<(OperationId, usize)>],
) -> Result<(), PhysicalInvariantError> {
    let mut successors = vec![BTreeSet::new(); operations.len()];
    let mut indegree = vec![0; operations.len()];
    for (consumer_index, operation) in operations.iter().enumerate() {
        let mut dependencies = BTreeSet::new();
        for input in &operation.inputs {
            if let Some((producer, _)) = producers[input.index()] {
                dependencies.insert(producer.index());
            }
        }
        indegree[consumer_index] = dependencies.len();
        for producer in dependencies {
            successors[producer].insert(consumer_index);
        }
    }
    ensure_dag(
        &successors,
        indegree,
        PhysicalInvariantError::OperationCycle,
    )
}

fn ensure_action_dag(
    operations: &[Operation],
    producers: &[Option<(OperationId, usize)>],
    operation_actions: &[ActionId],
    action_count: usize,
) -> Result<(), PhysicalInvariantError> {
    let mut successors = vec![BTreeSet::new(); action_count];
    let mut predecessors = vec![BTreeSet::new(); action_count];
    for (consumer_index, operation) in operations.iter().enumerate() {
        let consumer = operation_actions[consumer_index].index();
        for input in &operation.inputs {
            let Some((producer, _)) = producers[input.index()] else {
                continue;
            };
            let producer = operation_actions[producer.index()].index();
            if producer != consumer && successors[producer].insert(consumer) {
                predecessors[consumer].insert(producer);
            }
        }
    }
    let indegree = predecessors.into_iter().map(|set| set.len()).collect();
    ensure_dag(&successors, indegree, PhysicalInvariantError::ActionCycle)
}

fn ensure_dag(
    successors: &[BTreeSet<usize>],
    mut indegree: Vec<usize>,
    error: PhysicalInvariantError,
) -> Result<(), PhysicalInvariantError> {
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect::<VecDeque<_>>();
    let mut visited = 0;
    while let Some(node) = ready.pop_front() {
        visited += 1;
        for successor in &successors[node] {
            indegree[*successor] -= 1;
            if indegree[*successor] == 0 {
                ready.push_back(*successor);
            }
        }
    }
    if visited == successors.len() {
        Ok(())
    } else {
        Err(error)
    }
}

fn validate_cross_action_storage(
    values: &[ValueInstance],
    producers: &[Option<(OperationId, usize)>],
    consumers: &[Vec<(OperationId, usize)>],
    operation_actions: &[ActionId],
) -> Result<(), PhysicalInvariantError> {
    for (value_index, value) in values.iter().enumerate() {
        if matches!(value.storage, Storage::External | Storage::Global) {
            continue;
        }
        let Some((producer, _)) = producers[value_index] else {
            continue;
        };
        let producer_action = operation_actions[producer.index()];
        if consumers[value_index]
            .iter()
            .any(|(consumer, _)| operation_actions[consumer.index()] != producer_action)
        {
            return Err(PhysicalInvariantError::CrossActionStorage {
                value: value_index,
                storage: value.storage,
            });
        }
    }
    Ok(())
}

fn derive_actions(
    pending: &[Vec<OperationId>],
    operations: &[Operation],
    producers: &[Option<(OperationId, usize)>],
    consumers: &[Vec<(OperationId, usize)>],
    operation_actions: &[ActionId],
    plan_output: ValueInstanceId,
) -> Vec<Action> {
    pending
        .iter()
        .enumerate()
        .map(|(action_index, action_operations)| {
            let action_id = ActionId::from_index(action_index);
            let mut inputs = BTreeSet::new();
            let mut outputs = BTreeSet::new();
            for operation_id in action_operations {
                let operation = &operations[operation_id.index()];
                for input in &operation.inputs {
                    let producer_action = producers[input.index()]
                        .map(|(producer, _)| operation_actions[producer.index()]);
                    if producer_action != Some(action_id) {
                        inputs.insert(*input);
                    }
                }
                for output in &operation.outputs {
                    let leaves_action = consumers[output.index()]
                        .iter()
                        .any(|(consumer, _)| operation_actions[consumer.index()] != action_id);
                    if *output == plan_output || leaves_action {
                        outputs.insert(*output);
                    }
                }
            }
            Action {
                operations: action_operations.clone(),
                inputs: inputs.into_iter().collect(),
                outputs: outputs.into_iter().collect(),
            }
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
