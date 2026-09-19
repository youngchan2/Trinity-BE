use std::collections::BTreeMap;
use std::sync::Arc;

use egg::Id;

use crate::analysis::{LogicalOperation, TensorShape, can_materialize, matmul_input_presentation};
use crate::{ExtractionContext, Layout, LogicalNode, ProgramCandidate};

mod error;

pub use error::LoweringError;
use trinity_lowering::{
    CommunicationKind, CommunicationOperation, ComputeOperation, ImplementationInstance,
    LoweringConfig, OperationPayload, PhysicalPlan, PhysicalPlanBuilder, Storage, TargetCapability,
    ValueInstanceId, all_gather_implementations, gemm_implementations,
};

#[derive(Clone)]
struct LoweringBranch {
    physical: PhysicalPlanBuilder,
    direct_values: BTreeMap<Id, ValueInstanceId>,
    transitions: BTreeMap<(Id, Layout), ValueInstanceId>,
}

impl LoweringBranch {
    fn new(target: TargetCapability, world_size: usize) -> Self {
        Self {
            physical: PhysicalPlanBuilder::new(target, world_size),
            direct_values: BTreeMap::new(),
            transitions: BTreeMap::new(),
        }
    }
}

/// Enumerates verified physical implementations for one logical candidate.
///
/// A valid candidate for which no currently implemented definition applies
/// produces an empty vector.
pub fn lower<'eg>(
    source: Arc<ProgramCandidate<'eg>>,
    context: &ExtractionContext,
    config: &LoweringConfig,
) -> Result<Vec<PhysicalPlan>, LoweringError> {
    let graph = &source.logical_graph;
    let mut initial = LoweringBranch::new(config.target(), context.world_size());

    // Initialize direct physical values before enumerating implementation branches.
    for node in graph.nodes() {
        // Decode the selected e-node once for support and boundary handling.
        let operation = graph
            .operation(node)
            .expect("an extracted logical node must decode to an operation");

        let dtype = graph
            .dtype(node)
            .expect("a supported logical tensor operation must have an inferred dtype");

        // Apply the selected layout to obtain this rank's exact local shape.
        let shape = {
            let global_shape = graph
                .tensor_shape(node)
                .expect("an extracted logical tensor operation must have an inferred shape");

            let Some(shape) =
                try_concrete_local_shape(global_shape, node.layout(), context.world_size())
            else {
                return Ok(Vec::new());
            };

            shape
        };

        // Pass only logical operations with a physical lowering implemented currently.
        let abi_input_tensor = match &operation {
            LogicalOperation::Load { tensor, .. } => Some(*tensor),
            // Matmul results are produced by the plan rather than supplied through its input ABI.
            LogicalOperation::Matmul { .. } => None,
            operation => unimplemented!("physical lowering is not implemented for {operation:?}"),
        };

        // Select storage for the value at the node's extracted layout.
        let storage = {
            let is_direct_output =
                node.eclass() == graph.root() && node.layout() == context.required_output_layout();

            // Keep ABI boundaries.
            if abi_input_tensor.is_some() || is_direct_output {
                Storage::External
            } else {
                Storage::Global
            }
        };

        // Allocate the value at its extracted layout and record its direct presentation.
        let value = initial.physical.add_value(dtype, shape, storage);
        initial.direct_values.insert(node.eclass(), value);

        // Expose logical loads as named inputs in the plan ABI.
        if let Some(tensor) = abi_input_tensor {
            initial.physical.bind_input(tensor.to_string(), value);
        }
    }

    let mut branches = vec![initial];

    // Expand branches with every applicable physical implementation of each logical operation.
    for output in graph.nodes() {
        let operation = graph
            .operation(output)
            .expect("a prevalidated logical node must decode to an operation");

        // Loads are represented by ABI bindings, so skip physical operation enumeration.
        if matches!(&operation, LogicalOperation::Load { .. }) {
            continue;
        }

        // Infer the single presentation required at each input edge by the selected layouts.
        let inputs = operation.inputs();
        let input_presentations = inputs
            .iter()
            .enumerate()
            .map(|(input_index, _)| {
                infer_input_presentation(graph, &operation, output, input_index)
            })
            .collect::<Vec<_>>();

        // Enumerate compute implementations for the resulting concrete local tensor shapes.
        let instances = enumerate_compute_implementations(
            graph,
            &operation,
            output,
            &input_presentations,
            config.target(),
            context.world_size(),
        );

        if instances.is_empty() {
            return Ok(Vec::new());
        }

        let mut next = Vec::new();

        // Resolve the fixed input presentations into branches before adding compute instances.
        for branch in branches {
            let input_branches = resolve_input_values(
                branch,
                &source,
                context,
                config,
                &inputs,
                &input_presentations,
            );

            for (branch, input_values) in input_branches {
                for instance in &instances {
                    let mut instance_branch = branch.clone();

                    let output_value = instance_branch.direct_values[&output.eclass()];
                    let physical_operation = instance_branch.physical.add_operation(
                        input_values.iter().copied(),
                        [output_value],
                        OperationPayload::Compute(ComputeOperation::new(instance.clone())),
                    );

                    instance_branch
                        .physical
                        .add_statement(trinity_lowering::Statement::Operation(physical_operation));
                    next.push(instance_branch);
                }
            }
        }

        branches = next;
        if branches.is_empty() {
            return Ok(Vec::new());
        }
    }

    let output_node = graph
        .node(graph.root())
        .expect("an extracted logical graph must contain its root");

    let mut plans = Vec::new();

    // Resolve the root value in the required ABI output layout, then finalize unique plans.
    for branch in branches {
        for (resolved, output) in resolve_value_instance(
            branch,
            &source,
            context,
            config,
            output_node,
            context.required_output_layout(),
        ) {
            let plan = resolved
                .physical
                .finalize(context.output_tensor(), output)?;
            if !plans.iter().any(|existing| plan.same_body(existing)) {
                plans.push(plan);
            }
        }
    }

    Ok(plans)
}

fn infer_input_presentation(
    graph: &crate::LogicalGraph<'_>,
    operation: &LogicalOperation<'_, '_>,
    output: &LogicalNode<'_>,
    input_index: usize,
) -> Layout {
    match operation {
        LogicalOperation::Matmul { lhs, rhs } => {
            let lhs_shape = graph
                .tensor_shape(lhs)
                .expect("an extracted matmul lhs must have an inferred shape");

            let rhs_shape = graph
                .tensor_shape(rhs)
                .expect("an extracted matmul rhs must have an inferred shape");

            let output_shape = graph
                .tensor_shape(output)
                .expect("an extracted matmul output must have an inferred shape");

            matmul_input_presentation(
                lhs_shape,
                rhs_shape,
                lhs.layout(),
                rhs.layout(),
                output_shape,
                output.layout(),
                input_index,
            )
            .expect("an extracted matmul input must have a unique presentation")
        }
        _ => unreachable!(
            "only operations admitted during branch initialization require input presentations"
        ),
    }
}

/// Enumerates compute implementations applicable to one fixed logical presentation.
fn enumerate_compute_implementations(
    graph: &crate::LogicalGraph<'_>,
    operation: &LogicalOperation<'_, '_>,
    output: &LogicalNode<'_>,
    input_presentations: &[Layout],
    target: TargetCapability,
    world_size: usize,
) -> Vec<ImplementationInstance> {
    let inputs = operation.inputs();

    let input_shapes = inputs
        .iter()
        .zip(input_presentations)
        .map(|(input, presentation)| {
            let tensor_shape = graph
                .tensor_shape(input)
                .expect("a supported input must have an inferred shape");

            try_concrete_local_shape(tensor_shape, presentation, world_size)
                .expect("an inferred presentation must have a concrete local shape")
        })
        .collect::<Vec<_>>();

    let output_shape = {
        let global_shape = graph
            .tensor_shape(output)
            .expect("a prevalidated direct output must have an inferred shape");

        // Apply the selected layout to convert the global shape into this rank's local shape.
        try_concrete_local_shape(global_shape, output.layout(), world_size)
            .expect("the direct output shape was validated during branch initialization")
    };

    let input_dtypes = inputs
        .iter()
        .map(|input| {
            graph
                .dtype(input)
                .expect("a supported input must have an inferred dtype")
        })
        .collect::<Vec<_>>();

    let output_dtype = graph
        .dtype(output)
        .expect("a supported logical tensor operation must have an inferred dtype");

    match operation {
        LogicalOperation::Matmul { .. } => match (input_dtypes.as_slice(), input_shapes.as_slice())
        {
            ([lhs_dtype, rhs_dtype], [lhs_shape, rhs_shape]) => gemm_implementations(target)
                .iter()
                .flat_map(|implementation| {
                    implementation.enumerate(
                        [*lhs_dtype, *rhs_dtype, output_dtype],
                        [lhs_shape, rhs_shape, &output_shape],
                    )
                })
                .collect(),
            _ => unreachable!("a matmul must expose exactly two inputs"),
        },
        _ => {
            unreachable!("only operations admitted during branch initialization reach enumeration")
        }
    }
}

/// Resolves each logical input to a physical value at its required layout.
///
/// The required layouts are fixed before this function is called. Multiple
/// results arise only when a required materialization has more than one
/// applicable communication implementation.
fn resolve_input_values<'eg>(
    branch: LoweringBranch,
    source: &Arc<ProgramCandidate<'eg>>,
    context: &ExtractionContext,
    config: &LoweringConfig,
    inputs: &[&LogicalNode<'eg>],
    required_layouts: &[Layout],
) -> Vec<(LoweringBranch, Vec<ValueInstanceId>)> {
    let mut branches = vec![(branch, Vec::with_capacity(inputs.len()))];
    for (input, required_layout) in inputs.iter().zip(required_layouts) {
        let mut next = Vec::new();
        for (branch, values) in branches {
            for (resolved_branch, value) in
                resolve_value_instance(branch, source, context, config, input, required_layout)
            {
                let mut resolved_values = values.clone();
                resolved_values.push(value);
                next.push((resolved_branch, resolved_values));
            }
        }
        branches = next;
    }
    branches
}

/// Resolves a physical value instance at the required layout.
fn resolve_value_instance<'eg>(
    branch: LoweringBranch,
    source: &Arc<ProgramCandidate<'eg>>,
    context: &ExtractionContext,
    config: &LoweringConfig,
    node: &LogicalNode<'eg>,
    required_layout: &Layout,
) -> Vec<(LoweringBranch, ValueInstanceId)> {
    if node.layout() == required_layout {
        let value = branch
            .direct_values
            .get(&node.eclass())
            .copied()
            .expect("every extracted logical node must have a direct physical value");

        return vec![(branch, value)];
    }

    if let Some(value) = branch
        .transitions
        .get(&(node.eclass(), required_layout.clone()))
        .copied()
    {
        return vec![(branch, value)];
    }

    if !can_materialize(node.layout(), required_layout) || !required_layout.is_replicated() {
        return Vec::new();
    }

    let shards = node.layout().shard_axes().collect::<Vec<_>>();
    let [shard_axis] = shards.as_slice() else {
        return Vec::new();
    };

    let graph = &source.logical_graph;

    let global_shape = graph
        .tensor_shape(node)
        .expect("a prevalidated logical value must have an inferred shape");

    let source_shape = try_concrete_local_shape(global_shape, node.layout(), context.world_size())
        .expect("a direct logical value must have a concrete local shape");

    let target_shape =
        try_concrete_local_shape(global_shape, required_layout, context.world_size())
            .expect("a reachable presentation must have a concrete local shape");

    let dtype = graph
        .dtype(node)
        .expect("a supported logical value must have an inferred dtype");

    let instances = all_gather_implementations(config.target())
        .iter()
        .flat_map(|implementation| {
            implementation.enumerate(
                dtype,
                [&source_shape, &target_shape],
                *shard_axis,
                context.world_size(),
            )
        })
        .collect::<Vec<_>>();

    let mut branches = Vec::new();
    for instance in instances {
        let mut resolved = branch.clone();
        let direct = resolved.direct_values[&node.eclass()];
        let is_output =
            node.eclass() == graph.root() && required_layout == context.required_output_layout();

        let target = resolved.physical.add_value(
            dtype,
            target_shape.iter().copied(),
            if is_output {
                Storage::External
            } else {
                Storage::Global
            },
        );

        let operation = resolved.physical.add_operation(
            [direct],
            [target],
            OperationPayload::Communication(CommunicationOperation::new(
                CommunicationKind::AllGather,
                instance,
            )),
        );

        resolved
            .physical
            .add_statement(trinity_lowering::Statement::Operation(operation));
        resolved
            .transitions
            .insert((node.eclass(), required_layout.clone()), target);

        branches.push((resolved, target));
    }
    branches
}

fn try_concrete_local_shape(
    shape: &TensorShape,
    layout: &Layout,
    world_size: usize,
) -> Option<Vec<usize>> {
    let mut shape = shape.concrete_dimensions().unwrap_or_else(|| {
        unimplemented!("physical lowering for symbolic tensor shapes is not implemented")
    });

    if world_size == 0 || layout.pending_reductions().next().is_some() {
        return None;
    }

    for axis in layout.shard_axes() {
        if axis >= shape.len() || !shape[axis].is_multiple_of(world_size) {
            return None;
        }
        shape[axis] /= world_size;
    }
    Some(shape)
}

#[cfg(test)]
mod tests;
