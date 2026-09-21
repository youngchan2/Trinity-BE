use crate::*;

#[test]
fn lowering_config_carries_the_target_capability() {
    let config = LoweringConfig::new(TargetCapability::Cuda(CudaTargetCapability::Hopper));

    assert_eq!(
        config.target(),
        TargetCapability::Cuda(CudaTargetCapability::Hopper)
    );
    assert_eq!(config, LoweringConfig::default());
}

#[test]
fn enumerates_wgmma_tiles_and_nvls_chunks() {
    let gemm = implementation();
    assert_eq!(gemm.id().as_str(), "hopper.wgmma.bf16");
    assert_eq!(
        gemm.attributes().iter().collect::<Vec<_>>(),
        vec![("tile_k", 64), ("tile_m", 128), ("tile_n", 128)]
    );
    let gather = all_gather_implementations(TargetCapability::Cuda(CudaTargetCapability::Hopper))
        [0]
    .enumerate(DType::Bf16, [&[64, 128], &[64, 256]], 1, 2);
    assert_eq!(gather.len(), 1);
    assert_eq!(gather[0].id().as_str(), "nvls.one_shot_push_nbi");
    assert_eq!(gather[0].attributes().get("chunk_extent"), Some(128));
}

#[test]
fn unsupported_presentations_have_no_implementation_instances() {
    let gemm = gemm_implementations(TargetCapability::Cuda(CudaTargetCapability::Hopper))[0];
    assert!(
        gemm.enumerate([DType::Fp32; 3], [&[128, 64], &[64, 128], &[128, 128]])
            .is_empty()
    );
    assert!(
        gemm.enumerate([DType::Bf16; 3], [&[192, 64], &[64, 128], &[192, 128]])
            .is_empty()
    );
    assert!(
        gemm.enumerate([DType::Bf16; 3], [&[64], &[64, 128], &[128]])
            .is_empty()
    );

    let gather =
        all_gather_implementations(TargetCapability::Cuda(CudaTargetCapability::Hopper))[0];
    for (dtype, source, destination, axis, world_size) in [
        (DType::Fp32, [64, 128], [64, 256], 1, 2),
        (DType::Bf16, [64, 128], [64, 128], 1, 1),
        (DType::Bf16, [64, 128], [64, 256], 2, 2),
        (DType::Bf16, [64, 128], [64, 384], 1, 2),
        (DType::Bf16, [64, 192], [64, 384], 1, 2),
    ] {
        assert!(
            gather
                .enumerate(dtype, [&source, &destination], axis, world_size)
                .is_empty()
        );
    }
}

#[test]
#[should_panic(expected = "matching K extents")]
fn mismatched_gemm_extents_remain_an_invariant_violation() {
    gemm_implementations(TargetCapability::Cuda(CudaTargetCapability::Hopper))[0]
        .enumerate([DType::Bf16; 3], [&[128, 64], &[128, 128], &[128, 128]]);
}

fn implementation() -> ImplementationInstance {
    gemm_implementations(TargetCapability::Cuda(CudaTargetCapability::Hopper))[0]
        .enumerate([DType::Bf16; 3], [&[128, 64], &[64, 128], &[128, 128]])
        .pop()
        .unwrap()
}

fn manual_plan(
    reverse_values: bool,
    reverse_operations: bool,
    intermediate_storage: Storage,
    combined_statement: bool,
) -> Result<PhysicalPlan, PhysicalInvariantError> {
    let mut branch =
        PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
    let shape = [128, 128];
    let (a, b, intermediate, output) = if reverse_values {
        let output = branch.add_value(DType::Bf16, shape, Storage::External);
        let intermediate = branch.add_value(DType::Bf16, shape, intermediate_storage);
        let b = branch.add_value(DType::Bf16, shape, Storage::External);
        let a = branch.add_value(DType::Bf16, shape, Storage::External);
        (a, b, intermediate, output)
    } else {
        let a = branch.add_value(DType::Bf16, shape, Storage::External);
        let b = branch.add_value(DType::Bf16, shape, Storage::External);
        let intermediate = branch.add_value(DType::Bf16, shape, intermediate_storage);
        let output = branch.add_value(DType::Bf16, shape, Storage::External);
        (a, b, intermediate, output)
    };
    branch.bind_input("A", a);
    branch.bind_input("B", b);
    let payload = || OperationPayload::Compute(ComputeOperation::new(implementation()));
    let (first, second) = if reverse_operations {
        let second = branch.add_operation([intermediate, b], [output], payload());
        let first = branch.add_operation([a, b], [intermediate], payload());
        (first, second)
    } else {
        let first = branch.add_operation([a, b], [intermediate], payload());
        let second = branch.add_operation([intermediate, b], [output], payload());
        (first, second)
    };
    if combined_statement {
        branch.add_statement(Statement::Loop(Loop {
            kind: LoopKind::Sequential,
            domain: LoopDomain {
                variable: "once".into(),
                start: IndexExpr::Constant(0),
                stop: IndexExpr::Constant(1),
                step: IndexExpr::Constant(1),
            },
            body: vec![Statement::Operation(first), Statement::Operation(second)],
        }));
    } else {
        branch.add_statement(crate::Statement::Operation(first));
        branch.add_statement(crate::Statement::Operation(second));
    }
    branch.finalize("Y", output)
}

#[test]
fn canonicalizes_branch_insertion_order_and_uses_builtin_hash() {
    let first = manual_plan(false, false, Storage::Global, false).unwrap();
    let second = manual_plan(true, true, Storage::Global, false).unwrap();

    assert!(first.same_body(&second));
    assert_eq!(first.hash(), second.hash());
    assert_eq!(first.hash(), super::plan::hash_plan(&first));
}

#[test]
fn hash_covers_storage_and_statement_graph() {
    let split = manual_plan(false, false, Storage::Global, false).unwrap();
    let combined = manual_plan(false, false, Storage::Global, true).unwrap();
    let shared = manual_plan(false, false, Storage::Shared, true).unwrap();

    assert_ne!(split.hash(), combined.hash());
    assert_ne!(combined.hash(), shared.hash());
    assert!(!split.same_body(&combined));
    assert!(!combined.same_body(&shared));
}

#[test]
fn finalization_rejects_boundary_and_cross_statement_storage_errors() {
    let mut duplicate =
        PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
    let input = duplicate.add_value(DType::Bf16, [1], Storage::External);
    duplicate.bind_input("X", input);
    duplicate.bind_input("X", input);
    assert!(matches!(
        duplicate.finalize("Y", input),
        Err(PhysicalInvariantError::DuplicateInputTensor { .. })
    ));

    let mut crossing =
        PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
    let a = crossing.add_value(DType::Bf16, [128, 128], Storage::External);
    let b = crossing.add_value(DType::Bf16, [128, 128], Storage::External);
    let shared = crossing.add_value(DType::Bf16, [128, 128], Storage::Shared);
    let output = crossing.add_value(DType::Bf16, [128, 128], Storage::External);
    crossing.bind_input("A", a);
    crossing.bind_input("B", b);
    let first = crossing.add_operation(
        [a, b],
        [shared],
        OperationPayload::Compute(ComputeOperation::new(implementation())),
    );
    let second = crossing.add_operation(
        [shared, b],
        [output],
        OperationPayload::Compute(ComputeOperation::new(implementation())),
    );
    crossing.add_statement(crate::Statement::Operation(first));
    crossing.add_statement(crate::Statement::Operation(second));
    assert!(matches!(
        crossing.finalize("Y", output),
        Err(PhysicalInvariantError::CrossStatementStorage {
            storage: Storage::Shared,
            ..
        })
    ));
}

#[test]
fn finalization_rejects_uninitialized_reads_and_missing_statement_membership() {
    let mut cyclic =
        PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
    let first_value = cyclic.add_value(DType::Bf16, [128, 128], Storage::External);
    let second_value = cyclic.add_value(DType::Bf16, [128, 128], Storage::Global);
    let first = cyclic.add_operation(
        [second_value, second_value],
        [first_value],
        OperationPayload::Compute(ComputeOperation::new(implementation())),
    );
    let second = cyclic.add_operation(
        [first_value, first_value],
        [second_value],
        OperationPayload::Compute(ComputeOperation::new(implementation())),
    );
    cyclic.add_statement(crate::Statement::Operation(first));
    cyclic.add_statement(crate::Statement::Operation(second));
    assert!(matches!(
        cyclic.finalize("Y", first_value),
        Err(PhysicalInvariantError::MissingProducer { .. })
    ));

    let mut missing =
        PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
    let input = missing.add_value(DType::Bf16, [128, 128], Storage::External);
    let output = missing.add_value(DType::Bf16, [128, 128], Storage::External);
    missing.bind_input("X", input);
    missing.add_operation(
        [input],
        [output],
        OperationPayload::Compute(ComputeOperation::new(implementation())),
    );
    assert!(matches!(
        missing.finalize("Y", output),
        Err(PhysicalInvariantError::MissingStatementMembership { .. })
    ));
}

#[test]
fn separate_ordered_stores_can_repeat_the_same_computation() {
    let mut branch =
        PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
    let a = branch.add_value(DType::Bf16, [128, 128], Storage::External);
    let b = branch.add_value(DType::Bf16, [128, 128], Storage::External);
    let first_output = branch.add_value(DType::Bf16, [128, 128], Storage::Global);
    let duplicate_output = branch.add_value(DType::Bf16, [128, 128], Storage::Global);
    let output = branch.add_value(DType::Bf16, [128, 128], Storage::External);
    branch.bind_input("A", a);
    branch.bind_input("B", b);

    let payload = || OperationPayload::Compute(ComputeOperation::new(implementation()));
    let first = branch.add_operation([a, b], [first_output], payload());
    let duplicate = branch.add_operation([a, b], [duplicate_output], payload());
    let consumer = branch.add_operation([first_output, b], [output], payload());
    branch.add_statement(crate::Statement::Operation(first));
    branch.add_statement(crate::Statement::Operation(duplicate));
    branch.add_statement(crate::Statement::Operation(consumer));

    let plan = branch.finalize("Y", output).unwrap();
    assert_eq!(plan.operations().len(), 3);
    assert_eq!(
        crate::emit(&plan)
            .unwrap()
            .execution()
            .as_streamed()
            .unwrap()
            .tasks
            .len(),
        3
    );
}

#[test]
fn input_aliases_share_one_canonical_value_and_are_order_independent() {
    let build = |reverse: bool| {
        let mut b =
            PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
        let x = b.add_value(DType::Bf16, [128, 128], Storage::External);

        for name in if reverse {
            ["second", "first"]
        } else {
            ["first", "second"]
        } {
            b.bind_input(name, x);
        }

        b.finalize("output", x).unwrap()
    };

    let a = build(false);
    let b = build(true);

    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.value_instances().count(), 1);
    assert_eq!(a.inputs()[0].value(), a.inputs()[1].value());
    assert_eq!(a.output().value(), a.inputs()[0].value());

    let emitted = crate::emit(&a).unwrap();
    assert_eq!(emitted.requirements().buffers[0].input_names.len(), 2);
}
