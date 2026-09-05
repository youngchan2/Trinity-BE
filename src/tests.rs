use crate::*;

#[test]
fn lowering_config_carries_the_target_capability() {
    let config = LoweringConfig::new(TargetCapability::Hopper);

    assert_eq!(config.target(), TargetCapability::Hopper);
}

#[test]
fn enumerates_wgmma_tiles_and_nvls_chunks() {
    let gemm = implementation();
    assert_eq!(gemm.id().as_str(), "hopper.wgmma.bf16");
    assert_eq!(
        gemm.attributes().iter().collect::<Vec<_>>(),
        vec![("tile_k", 64), ("tile_m", 128), ("tile_n", 128)]
    );
    let gather = all_gather_implementations(TargetCapability::Hopper)[0].enumerate(
        DType::Bf16,
        [&[64, 128], &[64, 256]],
        1,
        2,
    );
    assert_eq!(gather.len(), 1);
    assert_eq!(gather[0].id().as_str(), "nvls.one_shot_push_nbi");
    assert_eq!(gather[0].attributes().get("chunk_extent"), Some(128));
}

#[test]
fn unsupported_presentations_have_no_implementation_instances() {
    let gemm = gemm_implementations(TargetCapability::Hopper)[0];
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

    let gather = all_gather_implementations(TargetCapability::Hopper)[0];
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
    gemm_implementations(TargetCapability::Hopper)[0]
        .enumerate([DType::Bf16; 3], [&[128, 64], &[128, 128], &[128, 128]]);
}

fn implementation() -> ImplementationInstance {
    gemm_implementations(TargetCapability::Hopper)[0]
        .enumerate([DType::Bf16; 3], [&[128, 64], &[64, 128], &[128, 128]])
        .pop()
        .unwrap()
}

fn manual_plan(
    reverse_values: bool,
    reverse_operations: bool,
    intermediate_storage: Storage,
    combined_action: bool,
) -> Result<PhysicalPlan, PhysicalInvariantError> {
    let mut branch = PhysicalPlanBuilder::new(TargetCapability::Hopper, 1);
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
    if combined_action {
        branch.add_action([first, second]);
    } else if reverse_operations {
        branch.add_action([second]);
        branch.add_action([first]);
    } else {
        branch.add_action([first]);
        branch.add_action([second]);
    }
    branch.finalize("Y", output)
}

#[test]
fn canonicalizes_branch_insertion_order_and_uses_builtin_hash() {
    let first = manual_plan(false, false, Storage::Global, false).unwrap();
    let second = manual_plan(true, true, Storage::Global, false).unwrap();

    assert!(first.same_body(&second));
    assert_eq!(first.hash(), second.hash());
    assert_eq!(first.hash(), super::physical::hash_plan(&first));
}

#[test]
fn hash_covers_storage_and_action_graph() {
    let split = manual_plan(false, false, Storage::Global, false).unwrap();
    let combined = manual_plan(false, false, Storage::Global, true).unwrap();
    let shared = manual_plan(false, false, Storage::Shared, true).unwrap();

    assert_ne!(split.hash(), combined.hash());
    assert_ne!(combined.hash(), shared.hash());
    assert!(!split.same_body(&combined));
    assert!(!combined.same_body(&shared));
}

#[test]
fn finalization_rejects_boundary_and_cross_action_storage_errors() {
    let mut duplicate = PhysicalPlanBuilder::new(TargetCapability::Hopper, 1);
    let input = duplicate.add_value(DType::Bf16, [1], Storage::External);
    duplicate.bind_input("X", input);
    duplicate.bind_input("X", input);
    assert!(matches!(
        duplicate.finalize("Y", input),
        Err(PhysicalInvariantError::DuplicateInputTensor { .. })
    ));

    let mut crossing = PhysicalPlanBuilder::new(TargetCapability::Hopper, 1);
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
    crossing.add_action([first]);
    crossing.add_action([second]);
    assert!(matches!(
        crossing.finalize("Y", output),
        Err(PhysicalInvariantError::CrossActionStorage {
            storage: Storage::Shared,
            ..
        })
    ));
}

#[test]
fn finalization_rejects_operation_cycles_and_missing_action_membership() {
    let mut cyclic = PhysicalPlanBuilder::new(TargetCapability::Hopper, 1);
    let first_value = cyclic.add_value(DType::Bf16, [128, 128], Storage::External);
    let second_value = cyclic.add_value(DType::Bf16, [128, 128], Storage::Global);
    let first = cyclic.add_operation(
        [second_value],
        [first_value],
        OperationPayload::Compute(ComputeOperation::new(implementation())),
    );
    let second = cyclic.add_operation(
        [first_value],
        [second_value],
        OperationPayload::Compute(ComputeOperation::new(implementation())),
    );
    cyclic.add_action([first]);
    cyclic.add_action([second]);
    assert!(matches!(
        cyclic.finalize("Y", first_value),
        Err(PhysicalInvariantError::OperationCycle)
    ));

    let mut missing = PhysicalPlanBuilder::new(TargetCapability::Hopper, 1);
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
        Err(PhysicalInvariantError::MissingActionMembership { .. })
    ));
}

#[test]
fn finalization_rejects_duplicate_physical_operations() {
    let mut branch = PhysicalPlanBuilder::new(TargetCapability::Hopper, 1);
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
    branch.add_action([first]);
    branch.add_action([duplicate]);
    branch.add_action([consumer]);

    assert!(matches!(
        branch.finalize("Y", output),
        Err(PhysicalInvariantError::DuplicateOperation { .. })
    ));
}
