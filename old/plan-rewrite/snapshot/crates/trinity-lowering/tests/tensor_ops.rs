#[allow(dead_code)]
mod support;
use support::tensor::*;
use trinity_lowering::*;

#[test]
fn pointwise_candidates_validate_operands_and_immediates() {
    assert_eq!(pointwise_implementations(TARGET).len(), 8);
    for definition in pointwise_implementations(TARGET) {
        let count = definition.input_count() + 1;
        let scalar = definition.requires_scalar().then_some(4096.0);
        for shape in [
            &[1][..],
            &[127],
            &[128],
            &[129],
            &[16, 4096],
            &[16, 16384],
            &[2, 257],
            &[32768, 65536],
            &[65536, 65536],
            &[4294967424],
        ] {
            let shapes = vec![shape; count];
            for dtype in [DType::Bf16, DType::Fp32] {
                assert_eq!(
                    definition
                        .enumerate(&vec![dtype; count], &shapes, scalar)
                        .len(),
                    1
                );
            }
        }
        assert!(
            definition
                .enumerate(&[DType::Fp32], &[&[16]], scalar)
                .is_empty()
        );
        assert!(
            definition
                .enumerate(&vec![DType::Fp32; count], &vec![&[0][..]; count], scalar)
                .is_empty()
        );
        assert!(
            definition
                .enumerate(
                    &vec![DType::Fp32; count],
                    &vec![&[1, 2, 3][..]; count],
                    scalar
                )
                .is_empty()
        );
        assert!(
            definition
                .enumerate(
                    &vec![DType::Fp32; count],
                    &vec![&[16][..]; count],
                    if scalar.is_some() { None } else { Some(2.0) }
                )
                .is_empty()
        );
    }
}

#[test]
fn reduction_and_broadcast_contracts() {
    let reduce = reduce_sum_implementations(TARGET)[0];
    assert_eq!(
        reduce
            .enumerate([DType::Bf16, DType::Fp32], [&[16, 4096], &[16]], 1)
            .len(),
        1
    );
    assert!(
        reduce
            .enumerate([DType::Fp32; 2], [&[16, 4096], &[4096]], 0)
            .is_empty()
    );
    assert!(
        reduce
            .enumerate([DType::Bf16; 2], [&[16, 4096], &[16]], 1)
            .is_empty()
    );
    assert!(
        reduce
            .enumerate([DType::Fp32; 2], [&[16, 4096], &[16, 1]], 1)
            .is_empty()
    );
    let broadcast = broadcast_implementations(TARGET)[0];
    assert_eq!(
        broadcast
            .enumerate(DType::Fp32, [&[16], &[16, 4096]], 1)
            .len(),
        1
    );
    assert!(
        broadcast
            .enumerate(DType::Fp32, [&[16], &[16, 4096]], 0)
            .is_empty()
    );
    assert!(
        broadcast
            .enumerate(DType::Fp32, [&[16], &[17, 4096]], 1)
            .is_empty()
    );
}

#[test]
fn nvls_limits_tile_indices_without_limiting_full_tensor_size() {
    let nvls = all_gather_implementations(TARGET)
        .iter()
        .find(|definition| definition.id().as_str() == "nvls.one_shot_push_nbi")
        .unwrap();
    for axis in [0, 1] {
        let source = [65536, 65536];
        let mut target = source;
        target[axis] *= 2;
        assert_eq!(
            nvls.enumerate(DType::Bf16, [&source, &target], axis, 2)
                .len(),
            1
        );
    }
    // Row strides and packed tile counts are still int in NVSHMEM 3.7.2.
    for (source, target, axis) in [
        ([128, 1usize << 30], [128, 1usize << 31], 1),
        ([1usize << 26, 128], [1usize << 26, 256], 1),
    ] {
        assert!(
            nvls.enumerate(DType::Bf16, [&source, &target], axis, 2)
                .is_empty()
        );
    }
}

#[test]
fn scalar_bits_are_canonical_and_preserve_signed_zero() {
    let plan = |scalar| pointwise("scalar_div", &[16], &[DType::Fp32; 2], Some(scalar), 1);
    assert!(plan(4096.).same_body(&plan(4096.)));
    for scalar in [2., 0., -0.] {
        assert!(!plan(4096.).same_body(&plan(scalar)));
    }
    assert!(!plan(0.).same_body(&plan(-0.)));
    let nan = f32::from_bits(0x7fc00001);
    assert!(plan(nan).same_body(&plan(nan)));
}

#[test]
fn malformed_pointwise_and_reduction_requests_fail_normalization() {
    let imp = pointwise_implementations(TARGET)[0]
        .enumerate(&[DType::Fp32; 3], &[&[16][..]; 3], None)
        .pop()
        .unwrap();
    let mut b = Builder::new(1);
    let x = b.input("X", DType::Fp32, &[16]);
    let out = b.value(DType::Fp32, &[16], Storage::External);
    b.operation(&[x], out, imp);
    assert!(matches!(
        b.plan.finalize("Y", out),
        Err(PhysicalInvariantError::InvalidProgram(_))
    ));

    let imp = reduce_sum_implementations(TARGET)[0]
        .enumerate([DType::Fp32; 2], [&[16, 128], &[16]], 1)
        .pop()
        .unwrap();
    let mut b = Builder::new(1);
    let x = b.input("X", DType::Fp32, &[16, 128]);
    let out = b.value(DType::Bf16, &[16], Storage::External);
    b.operation(&[x], out, imp);
    assert!(matches!(
        b.plan.finalize("Y", out),
        Err(PhysicalInvariantError::InvalidProgram(_))
    ));
}

#[test]
fn bf16_communication_backends_reject_fp32_and_vectors() {
    for definition in all_gather_implementations(TARGET) {
        for (dtype, source, target) in [
            (DType::Fp32, vec![128, 128], vec![128, 256]),
            (DType::Bf16, vec![128], vec![256]),
        ] {
            let implementation = definition
                .enumerate(DType::Bf16, [&[128, 128], &[128, 256]], 1, 2)
                .pop()
                .unwrap();
            let mut b = PhysicalPlanBuilder::new(TARGET, 2);
            let x = b.add_value(dtype, source, Storage::External);
            b.bind_input("X", x);
            let y = b.add_value(dtype, target, Storage::External);
            let op = b.add_operation(
                [x],
                [y],
                OperationPayload::Communication(CommunicationOperation::new(
                    CommunicationKind::AllGather,
                    implementation,
                )),
            );
            b.add_statement(trinity_lowering::Statement::Operation(op));
            assert!(matches!(
                b.finalize("Y", y),
                Err(PhysicalInvariantError::InvalidProgram(_))
            ));
        }
    }
}
