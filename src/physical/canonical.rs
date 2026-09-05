use std::collections::BTreeSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::plan::{
    Action, ActionId, IdVec, Operation, OperationId, OperationPayload, PhysicalPlan, TensorBinding,
    ValueInstance, ValueInstanceId,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OperationOrderKey {
    payload: OperationPayload,
    inputs: Vec<usize>,
    outputs: Vec<ValueInstance>,
}

pub(super) fn canonicalize(
    plan: PhysicalPlan,
    producers: &[Option<(OperationId, usize)>],
    operation_actions: &[ActionId],
) -> PhysicalPlan {
    let mut value_order = Vec::with_capacity(plan.value_instances.len());
    let mut value_remap = vec![usize::MAX; plan.value_instances.len()];

    let mut input_values = plan
        .inputs
        .iter()
        .map(|input| (input.tensor.as_str(), input.value.index()))
        .collect::<Vec<_>>();
    input_values.sort_by(|lhs, rhs| lhs.0.cmp(rhs.0));
    for (_, value) in input_values {
        assign_value(value, &mut value_order, &mut value_remap);
    }

    let operation_order =
        canonical_operation_order(&plan, producers, &mut value_order, &mut value_remap);
    debug_assert_eq!(operation_order.len(), plan.operations.len());
    debug_assert_eq!(value_order.len(), plan.value_instances.len());

    let operation_remap = remap(&operation_order, plan.operations.len());
    let values = value_order
        .iter()
        .map(|value| plan.value_instances.values[*value].clone())
        .collect::<Vec<_>>();
    let operations = operation_order
        .iter()
        .map(|operation| remap_operation(&plan.operations.values[*operation], &value_remap))
        .collect::<Vec<_>>();

    let action_depths = action_depths(&plan, producers, operation_actions);
    let action_keys = plan
        .actions
        .values
        .iter()
        .enumerate()
        .map(|(action, value)| {
            (
                action_depths[action],
                canonical_action_operations(value, &operation_remap),
            )
        })
        .collect::<Vec<_>>();
    let mut action_order = (0..plan.actions.len()).collect::<Vec<_>>();
    action_order.sort_by(|lhs, rhs| action_keys[*lhs].cmp(&action_keys[*rhs]));

    let actions = action_order
        .iter()
        .map(|action| {
            remap_action(
                &plan.actions.values[*action],
                &operation_remap,
                &value_remap,
            )
        })
        .collect::<Vec<_>>();

    let mut inputs = plan
        .inputs
        .iter()
        .map(|binding| TensorBinding {
            tensor: binding.tensor.clone(),
            value: ValueInstanceId::from_index(value_remap[binding.value.index()]),
        })
        .collect::<Vec<_>>();
    inputs.sort_by(|lhs, rhs| lhs.tensor.cmp(&rhs.tensor));
    let output = TensorBinding {
        tensor: plan.output.tensor,
        value: ValueInstanceId::from_index(value_remap[plan.output.value.index()]),
    };

    let mut canonical = PhysicalPlan {
        target: plan.target,
        world_size: plan.world_size,
        inputs: inputs.into_boxed_slice(),
        value_instances: IdVec::from_values(values),
        operations: IdVec::from_values(operations),
        actions: IdVec::from_values(actions),
        output,
        hash: 0,
    };
    canonical.hash = hash_plan(&canonical);
    canonical
}

fn canonical_operation_order(
    plan: &PhysicalPlan,
    producers: &[Option<(OperationId, usize)>],
    value_order: &mut Vec<usize>,
    value_remap: &mut [usize],
) -> Vec<usize> {
    let mut successors = vec![BTreeSet::new(); plan.operations.len()];
    let mut indegree = vec![0; plan.operations.len()];
    for (consumer, operation) in plan.operations.values.iter().enumerate() {
        let dependencies = operation
            .inputs
            .iter()
            .filter_map(|input| producers[input.index()].map(|(producer, _)| producer.index()))
            .collect::<BTreeSet<_>>();
        indegree[consumer] = dependencies.len();
        for producer in dependencies {
            successors[producer].insert(consumer);
        }
    }

    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(operation, degree)| (*degree == 0).then_some(operation))
        .collect::<Vec<_>>();
    let mut order = Vec::with_capacity(plan.operations.len());

    while !ready.is_empty() {
        let mut keyed = ready
            .into_iter()
            .map(|operation| (operation, operation_order_key(operation, plan, value_remap)))
            .collect::<Vec<_>>();
        keyed.sort_by(|lhs, rhs| lhs.1.cmp(&rhs.1));

        for (operation, _) in &keyed {
            order.push(*operation);
            for output in &plan.operations.values[*operation].outputs {
                assign_value(output.index(), value_order, value_remap);
            }
        }

        let mut next = Vec::new();
        for (operation, _) in keyed {
            for successor in &successors[operation] {
                indegree[*successor] -= 1;
                if indegree[*successor] == 0 {
                    next.push(*successor);
                }
            }
        }
        ready = next;
    }

    order
}

fn operation_order_key(
    operation: usize,
    plan: &PhysicalPlan,
    value_remap: &[usize],
) -> OperationOrderKey {
    let operation = &plan.operations.values[operation];
    OperationOrderKey {
        payload: operation.payload.clone(),
        inputs: operation
            .inputs
            .iter()
            .map(|input| {
                let canonical = value_remap[input.index()];
                debug_assert_ne!(canonical, usize::MAX);
                canonical
            })
            .collect(),
        outputs: operation
            .outputs
            .iter()
            .map(|output| plan.value_instances.values[output.index()].clone())
            .collect(),
    }
}

fn assign_value(value: usize, order: &mut Vec<usize>, remap: &mut [usize]) {
    debug_assert_eq!(remap[value], usize::MAX);
    remap[value] = order.len();
    order.push(value);
}

fn remap_operation(operation: &Operation, value_remap: &[usize]) -> Operation {
    Operation {
        inputs: operation
            .inputs
            .iter()
            .map(|value| ValueInstanceId::from_index(value_remap[value.index()]))
            .collect(),
        outputs: operation
            .outputs
            .iter()
            .map(|value| ValueInstanceId::from_index(value_remap[value.index()]))
            .collect(),
        payload: operation.payload.clone(),
    }
}

fn action_depths(
    plan: &PhysicalPlan,
    producers: &[Option<(OperationId, usize)>],
    operation_actions: &[ActionId],
) -> Vec<usize> {
    let mut predecessors = vec![BTreeSet::new(); plan.actions.len()];
    for (consumer_index, operation) in plan.operations.values.iter().enumerate() {
        let consumer = operation_actions[consumer_index].index();
        for input in &operation.inputs {
            let Some((producer, _)) = producers[input.index()] else {
                continue;
            };
            let producer = operation_actions[producer.index()].index();
            if producer != consumer {
                predecessors[consumer].insert(producer);
            }
        }
    }
    let mut depths = vec![None; plan.actions.len()];
    for action in 0..plan.actions.len() {
        action_depth(action, &predecessors, &mut depths);
    }
    depths.into_iter().map(Option::unwrap).collect()
}

fn action_depth(
    action: usize,
    predecessors: &[BTreeSet<usize>],
    depths: &mut [Option<usize>],
) -> usize {
    if let Some(depth) = depths[action] {
        return depth;
    }
    let depth = predecessors[action]
        .iter()
        .map(|predecessor| action_depth(*predecessor, predecessors, depths) + 1)
        .max()
        .unwrap_or(0);
    depths[action] = Some(depth);
    depth
}

fn canonical_action_operations(action: &Action, operation_remap: &[usize]) -> Vec<usize> {
    let mut operations = action
        .operations
        .iter()
        .map(|operation| operation_remap[operation.index()])
        .collect::<Vec<_>>();
    operations.sort_unstable();
    operations
}

fn remap_action(action: &Action, operation_remap: &[usize], value_remap: &[usize]) -> Action {
    let mut operations = action
        .operations
        .iter()
        .map(|operation| OperationId::from_index(operation_remap[operation.index()]))
        .collect::<Vec<_>>();
    operations.sort_unstable();
    let mut inputs = action
        .inputs
        .iter()
        .map(|value| ValueInstanceId::from_index(value_remap[value.index()]))
        .collect::<Vec<_>>();
    inputs.sort_unstable();
    let mut outputs = action
        .outputs
        .iter()
        .map(|value| ValueInstanceId::from_index(value_remap[value.index()]))
        .collect::<Vec<_>>();
    outputs.sort_unstable();
    Action {
        operations,
        inputs,
        outputs,
    }
}

fn remap(order: &[usize], count: usize) -> Vec<usize> {
    let mut remap = vec![0; count];
    for (new, old) in order.iter().copied().enumerate() {
        remap[old] = new;
    }
    remap
}

pub(crate) fn hash_plan(plan: &PhysicalPlan) -> u64 {
    let mut hasher = DefaultHasher::new();
    plan.target.hash(&mut hasher);
    plan.world_size.hash(&mut hasher);
    plan.inputs.hash(&mut hasher);
    plan.value_instances.hash(&mut hasher);
    plan.operations.hash(&mut hasher);
    plan.actions.hash(&mut hasher);
    plan.output.hash(&mut hasher);
    hasher.finish()
}
