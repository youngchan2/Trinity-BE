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
fn relu_emits_vectors_matrices_and_mixed_dtype_outputs() {
    for world in [1, 2] {
        for input in [DType::Bf16, DType::Fp32] {
            for output in [DType::Bf16, DType::Fp32] {
                for shape in [&[129][..], &[3, 257]] {
                    let plan = pointwise("relu", shape, &[input, output], None, world);
                    let source = emit(&plan).unwrap();
                    assert_eq!(source.requirements().buffers[0].dtype, input);
                    assert_eq!(source.requirements().buffers[1].dtype, output);
                    let tasks = if shape.len() == 1 { 2 } else { 9 };
                    assert_eq!(source.execution().tasks_per_rank, tasks);
                    assert!(
                        source
                            .execution()
                            .tasks
                            .iter()
                            .all(|t| t.dependencies.is_empty() && t.stages.is_empty())
                    );
                }
            }
        }
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
fn vectors_and_matrix_tails_have_exact_coverage_and_dtype_sizes() {
    for world in [1, 2] {
        for shape in [&[1][..], &[127], &[128], &[129], &[2, 257]] {
            let plan = pointwise(
                "add",
                shape,
                &[DType::Bf16, DType::Fp32, DType::Fp32],
                None,
                world,
            );
            let source = emit(&plan).unwrap(); // validates non-overlapping complete coverage
            for buffer in &source.requirements().buffers {
                assert_eq!(buffer.shape, shape);
                assert_eq!(
                    buffer.bytes,
                    shape.iter().product::<usize>() * buffer.dtype.size_bytes()
                );
                assert_eq!(
                    buffer.strides,
                    if shape.len() == 1 {
                        vec![1]
                    } else {
                        vec![shape[1], 1]
                    }
                );
            }
            let expected = if shape.len() == 1 {
                shape[0].div_ceil(128)
            } else {
                shape[0] * shape[1].div_ceil(128)
            };
            assert_eq!(source.execution().tasks_per_rank, expected);
            assert_eq!(source.requirements().workspace_bytes == 0, world == 1);
        }
    }
}

#[test]
fn normalization_dependencies_wait_for_complete_row_and_selected_statistic() {
    let plan = normalization(16, 257, 2);
    let source = emit(&plan).unwrap();
    let operation = |name| {
        plan.operations().find(|(_,op)| matches!(op.payload(), OperationPayload::Compute(c) if c.implementation().id().as_str()==name)).unwrap().0.index()
    };
    let reduce = operation("cuda.reduce_sum");
    let broadcast = operation("cuda.broadcast");
    for task in &source.execution().tasks {
        if task.operation == reduce {
            assert_eq!(task.dependencies.len(), 3); // all three input chunks of this row
            assert!(task.stages.is_empty());
            assert_eq!(task.shared_memory_bytes, 512);
        }
        if task.operation == broadcast {
            assert_eq!(task.dependencies.len(), 1);
        }
        assert!(task.dependencies.iter().all(|d| d.rank == task.rank));
    }
    assert_eq!(
        source
            .requirements()
            .buffers
            .iter()
            .filter(|b| b.shape == [16])
            .count(),
        3
    );
    assert!(
        source
            .requirements()
            .buffers
            .iter()
            .filter(|b| b.shape == [16])
            .all(|b| b.dtype == DType::Fp32)
    );
}

#[test]
fn malformed_pointwise_and_reduction_plans_return_emit_errors() {
    let imp = pointwise_implementations(TARGET)[0]
        .enumerate(&[DType::Fp32; 3], &[&[16][..]; 3], None)
        .pop()
        .unwrap();
    let mut b = Builder::new(1);
    let x = b.input("X", DType::Fp32, &[16]);
    let out = b.value(DType::Fp32, &[16], Storage::External);
    b.operation(&[x], out, imp);
    assert!(matches!(
        emit(&b.finish(out)),
        Err(EmitError::Unsupported(_))
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
        emit(&b.finish(out)),
        Err(EmitError::Unsupported(_))
    ));
}

#[test]
fn communication_to_fp32_reduction_has_transitive_dependencies() {
    let plan = gather_normalization(2);
    let source = emit(&plan).unwrap();
    assert!(
        source
            .execution()
            .tasks
            .iter()
            .any(|t| t.dependencies.iter().any(|d| d.rank != t.rank))
    );
    assert_eq!(
        source.requirements().buffers.last().unwrap().dtype,
        DType::Fp32
    );
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
            b.add_action([op]);
            assert!(matches!(
                emit(&b.finalize("Y", y).unwrap()),
                Err(EmitError::Unsupported(_))
            ));
        }
    }
}
