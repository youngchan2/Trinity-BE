use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;
use crate::{
    BoundaryPlacementContract, CudaTargetCapability, DType, DistributedPlanRequest,
    ExtractionContext, Placement, SaturatedProgram, SaturationLimits, SourceProgram, TensorBinding,
    TensorMetadata, extract_candidates, saturate,
};

fn gemm_program(
    m: usize,
    k: usize,
    n: usize,
    weight_placement: Placement,
    output_placement: Placement,
    dtype: DType,
    world_size: usize,
) -> SaturatedProgram {
    let source = SourceProgram::parse(
        "
        (store (output Y)
            (@
                (load (input X) (index fulltile fulltile))
                (load (input W) (index fulltile fulltile)))
            (index fulltile fulltile))
        ",
    )
    .unwrap();
    let metadata = BTreeMap::from([
        (
            "X".to_owned(),
            TensorMetadata {
                shape: vec![m, k],
                dtype,
            },
        ),
        (
            "W".to_owned(),
            TensorMetadata {
                shape: vec![k, n],
                dtype,
            },
        ),
        (
            "Y".to_owned(),
            TensorMetadata {
                shape: vec![m, n],
                dtype,
            },
        ),
    ]);
    let boundary = BoundaryPlacementContract::new(
        BTreeMap::from([
            ("X".to_owned(), Placement::Replicated),
            ("W".to_owned(), weight_placement),
        ]),
        BTreeMap::from([("Y".to_owned(), output_placement)]),
    );
    let request = DistributedPlanRequest::try_new(source, metadata, boundary, world_size).unwrap();
    saturate(request, SaturationLimits::default(), &[]).unwrap()
}

fn default_program(
    weight_placement: Placement,
    output_placement: Placement,
    dtype: DType,
) -> SaturatedProgram {
    gemm_program(128, 64, 256, weight_placement, output_placement, dtype, 2)
}

fn find_candidate<'a, 'eg>(
    context: &ExtractionContext,
    candidates: &'a [ProgramCandidate<'eg>],
    output_matches_boundary: bool,
) -> &'a ProgramCandidate<'eg> {
    candidates
        .iter()
        .find(|candidate| {
            candidate
                .logical_graph
                .node(candidate.logical_graph.root())
                .is_some_and(|node| {
                    (node.layout() == context.required_output_layout()) == output_matches_boundary
                })
        })
        .unwrap()
}

fn implementation_ids(plan: &PhysicalPlan) -> Vec<&'static str> {
    plan.operations()
        .map(|(_, operation)| match operation.payload() {
            OperationPayload::Compute(operation) => operation.implementation().id().as_str(),
            OperationPayload::Communication(operation) => operation.implementation().id().as_str(),
        })
        .collect()
}

fn gemm_attribute(plan: &PhysicalPlan, name: &str) -> usize {
    plan.operations()
        .find_map(|(_, operation)| match operation.payload() {
            OperationPayload::Compute(operation) => {
                operation.implementation().attributes().get(name)
            }
            OperationPayload::Communication(_) => None,
        })
        .unwrap()
}

fn chunk_extent(plan: &PhysicalPlan) -> usize {
    plan.operations()
        .find_map(|(_, operation)| match operation.payload() {
            OperationPayload::Communication(operation) => {
                operation.implementation().attributes().get("chunk_extent")
            }
            OperationPayload::Compute(_) => None,
        })
        .unwrap()
}

fn nvls_plan(plans: Vec<PhysicalPlan>) -> PhysicalPlan {
    plans
        .into_iter()
        .find(|plan| implementation_ids(plan).contains(&"nvls.one_shot_push_nbi"))
        .unwrap()
}

#[test]
#[should_panic(expected = "physical lowering is not implemented")]
fn unsupported_logical_operations_are_not_empty_search_results() {
    let source = SourceProgram::parse(
        "
        (store (output Y)
            (+
                (load (input X) (index fulltile fulltile))
                (load (input X) (index fulltile fulltile)))
            (index fulltile fulltile))
        ",
    )
    .unwrap();
    let metadata = BTreeMap::from([
        (
            "X".to_owned(),
            TensorMetadata {
                shape: vec![128, 128],
                dtype: DType::Bf16,
            },
        ),
        (
            "Y".to_owned(),
            TensorMetadata {
                shape: vec![128, 128],
                dtype: DType::Bf16,
            },
        ),
    ]);
    let boundary = BoundaryPlacementContract::new(
        BTreeMap::from([("X".to_owned(), Placement::Replicated)]),
        BTreeMap::from([("Y".to_owned(), Placement::Replicated)]),
    );
    let request = DistributedPlanRequest::try_new(source, metadata, boundary, 1).unwrap();
    let program = saturate(request, SaturationLimits::default(), &[]).unwrap();
    let (context, mut candidates) = extract_candidates(&program).unwrap();

    let _ = lower(
        Arc::new(candidates.pop().unwrap()),
        &context,
        &LoweringConfig::default(),
    );
}

#[test]
fn lowers_a_direct_gemm_with_named_external_boundaries() {
    let program = default_program(Placement::Replicated, Placement::Replicated, DType::Bf16);
    let (context, mut candidates) = extract_candidates(&program).unwrap();
    let plan = lower(
        Arc::new(candidates.pop().unwrap()),
        &context,
        &LoweringConfig::default(),
    )
    .unwrap()
    .pop()
    .unwrap();

    assert_eq!(
        plan.target(),
        TargetCapability::Cuda(CudaTargetCapability::Hopper)
    );
    assert_eq!(plan.world_size(), 2);
    assert_eq!(
        plan.inputs()
            .iter()
            .map(TensorBinding::tensor)
            .collect::<Vec<_>>(),
        vec!["W", "X"]
    );
    assert_eq!(plan.output().tensor(), "Y");
    assert_eq!(
        plan.value_instance(plan.output().value()).unwrap().shape(),
        &[128, 256]
    );
    assert!(plan.inputs().iter().all(|binding| {
        plan.value_instance(binding.value()).unwrap().storage() == Storage::External
    }));
    assert_eq!(plan.statements().len(), 1);
    assert_eq!(implementation_ids(&plan), vec!["hopper.wgmma.bf16"]);
}

#[test]
fn lowers_input_and_output_all_gather_paths_with_local_shapes() {
    let program = default_program(
        Placement::Sharded { tensor_axis: 1 },
        Placement::Replicated,
        DType::Bf16,
    );
    let (context, candidates) = extract_candidates(&program).unwrap();
    let direct_output = find_candidate(&context, &candidates, true).clone();
    let sharded_output = find_candidate(&context, &candidates, false).clone();

    let input_plan = nvls_plan(
        lower(
            Arc::new(direct_output),
            &context,
            &LoweringConfig::default(),
        )
        .unwrap(),
    );
    assert_eq!(
        implementation_ids(&input_plan),
        vec!["nvls.one_shot_push_nbi", "hopper.wgmma.bf16"]
    );
    let weight = input_plan
        .inputs()
        .iter()
        .find(|binding| binding.tensor() == "W")
        .unwrap();
    assert_eq!(
        input_plan.value_instance(weight.value()).unwrap().shape(),
        &[64, 128]
    );

    let output_plan = nvls_plan(
        lower(
            Arc::new(sharded_output),
            &context,
            &LoweringConfig::default(),
        )
        .unwrap(),
    );
    assert_eq!(
        implementation_ids(&output_plan),
        vec!["hopper.wgmma.bf16", "nvls.one_shot_push_nbi"]
    );
    assert_eq!(
        output_plan
            .value_instance(output_plan.output().value())
            .unwrap()
            .shape(),
        &[128, 256]
    );
    assert_eq!(output_plan.statements().len(), 2);
    assert!(
        output_plan
            .statements()
            .iter()
            .all(|statement| statement.operations().len() == 1)
    );
}

#[test]
fn infers_input_presentation_from_the_selected_matmul_layout() {
    let program = default_program(
        Placement::Sharded { tensor_axis: 1 },
        Placement::Replicated,
        DType::Bf16,
    );
    let (context, candidates) = extract_candidates(&program).unwrap();

    let replicated_output = find_candidate(&context, &candidates, true);
    let graph = &replicated_output.logical_graph;
    let output = graph.node(graph.root()).unwrap();
    let operation = graph.operation(output).unwrap();
    assert!(
        infer_input_presentation(graph, &operation, output, 1).is_replicated(),
        "a replicated matmul output must consume the sharded weight through a replicated presentation"
    );

    let sharded_output = find_candidate(&context, &candidates, false);
    let graph = &sharded_output.logical_graph;
    let output = graph.node(graph.root()).unwrap();
    let operation = graph.operation(output).unwrap();
    assert_eq!(
        infer_input_presentation(graph, &operation, output, 1)
            .shard_axes()
            .collect::<Vec<_>>(),
        vec![1],
        "an N-sharded matmul output must consume the N-sharded weight directly"
    );
}

#[test]
fn passes_wgmma_tile_and_nvls_chunk_attributes_to_physical_plans() {
    let program = default_program(
        Placement::Sharded { tensor_axis: 1 },
        Placement::Replicated,
        DType::Bf16,
    );
    let (context, candidates) = extract_candidates(&program).unwrap();
    let candidate = find_candidate(&context, &candidates, true).clone();
    let plans = lower(Arc::new(candidate), &context, &LoweringConfig::default()).unwrap();

    assert_eq!(plans.len(), 3);
    assert!(
        plans
            .iter()
            .all(|plan| gemm_attribute(plan, "tile_m") == 128)
    );
    assert_eq!(chunk_extent(&nvls_plan(plans)), 128);
}

#[test]
fn rejects_shapes_requiring_an_unimplemented_tail_tile() {
    let program = gemm_program(
        192,
        64,
        256,
        Placement::Replicated,
        Placement::Replicated,
        DType::Bf16,
        1,
    );
    let (context, mut candidates) = extract_candidates(&program).unwrap();
    let candidate = Arc::new(candidates.pop().unwrap());
    assert!(
        lower(candidate, &context, &LoweringConfig::default())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn an_unimplemented_dtype_returns_no_plan() {
    let program = default_program(Placement::Replicated, Placement::Replicated, DType::Fp32);
    let (context, mut candidates) = extract_candidates(&program).unwrap();
    assert!(
        lower(
            Arc::new(candidates.pop().unwrap()),
            &context,
            &LoweringConfig::default()
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn physical_plans_outlive_the_candidate_and_egraph() {
    let plans = {
        let program = default_program(
            Placement::Sharded { tensor_axis: 1 },
            Placement::Replicated,
            DType::Bf16,
        );
        let (context, candidates) = extract_candidates(&program).unwrap();
        candidates
            .into_iter()
            .flat_map(|candidate| {
                let source = Arc::new(candidate);
                let weak_source = Arc::downgrade(&source);
                let plans = lower(source, &context, &LoweringConfig::default()).unwrap();
                assert!(weak_source.upgrade().is_none());
                plans
            })
            .collect::<Vec<_>>()
    };

    assert_eq!(plans.len(), 6);
    for plan in plans {
        assert_eq!(plan.output().tensor(), "Y");
        assert_eq!(
            plan.value_instance(plan.output().value()).unwrap().shape(),
            &[128, 256]
        );
        assert_eq!(plan.operations().len(), 2);
        assert_eq!(plan.statements().len(), 2);
        let cloned = plan.clone();
        assert!(plan.same_body(&cloned));
        assert_eq!(plan.hash(), cloned.hash());
    }
}

#[test]
fn enumerates_backends_and_reports_deferred_fusion() {
    let program = default_program(
        Placement::Sharded { tensor_axis: 1 },
        Placement::Replicated,
        DType::Bf16,
    );
    let (context, candidates) = extract_candidates(&program).unwrap();
    for candidate in candidates {
        let plans = lower(Arc::new(candidate), &context, &LoweringConfig::default()).unwrap();
        assert_eq!(plans.len(), 3);
        for source in plans {
            assert_eq!(source.statements().len(), 2);
            let candidates = crate::fuse(&source, crate::fusion_rules(source.target())).unwrap();
            assert!(candidates[0].same_body(&source));
        }
    }
}

#[test]
fn candidate_and_public_builder_share_the_normalized_plan() {
    for world in [1, 2] {
        let program = gemm_program(
            128,
            128,
            128,
            Placement::Replicated,
            Placement::Replicated,
            DType::Bf16,
            world,
        );
        let (context, mut candidates) = extract_candidates(&program).unwrap();
        let candidate = lower(
            Arc::new(candidates.pop().unwrap()),
            &context,
            &LoweringConfig::default(),
        )
        .unwrap()
        .remove(0);
        let target = TargetCapability::Cuda(CudaTargetCapability::Hopper);
        let mut b = PhysicalPlanBuilder::new(target, world);
        let x = b.add_value(DType::Bf16, [128, 128], Storage::External);
        b.bind_input("X", x);
        let w = b.add_value(DType::Bf16, [128, 128], Storage::External);
        b.bind_input("W", w);
        let y = b.add_value(DType::Bf16, [128, 128], Storage::External);
        let instance = gemm_implementations(target)[0]
            .enumerate([DType::Bf16; 3], [&[128, 128]; 3])
            .pop()
            .unwrap();
        let op = b.add_operation(
            [x, w],
            [y],
            OperationPayload::Compute(ComputeOperation::new(instance)),
        );
        b.add_statement(trinity_lowering::Statement::Operation(op));
        let builder = b.finalize("Y", y).unwrap();
        assert!(candidate.same_body(&builder));
        assert_eq!(candidate.hash(), builder.hash());
    }
}
