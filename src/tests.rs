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
    let gemm = gemm_implementations(LoweringConfig::default().target())[0]
        .enumerate([DType::Bf16; 3], [&[128, 64], &[64, 128], &[128, 128]])
        .pop()
        .unwrap();
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

fn value(b: &mut PhysicalPlanBuilder, name: &str, storage: Storage) -> ValueInstanceId {
    b.add_named_value(name, DType::Bf16, [128, 128], storage)
}
fn add(
    b: &mut PhysicalPlanBuilder,
    inflows: [ValueInstanceId; 2],
    output: ValueInstanceId,
) -> OperationId {
    let access = |value| TensorAccess::new(value, [AccessIndex::FullTile, AccessIndex::FullTile]);
    b.add_operation(
        inflows,
        [output],
        Expression::Store {
            destination: access(output),
            value: Box::new(Expression::Add(Box::new([
                Expression::Load(access(inflows[0])),
                Expression::Load(access(inflows[1])),
            ]))),
        },
    )
}

fn builder() -> PhysicalPlanBuilder {
    PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1)
}
fn manual_plan(
    reverse_values: bool,
    reverse_operations: bool,
    storage: Storage,
    combined: bool,
) -> Result<PhysicalPlan, PhysicalInvariantError> {
    let mut b = builder();
    let (a, w, t, y) = if reverse_values {
        let y = value(&mut b, "Y", Storage::External);
        let t = value(&mut b, "T", storage);
        let w = value(&mut b, "W", Storage::External);
        let a = value(&mut b, "A", Storage::External);
        (a, w, t, y)
    } else {
        let a = value(&mut b, "A", Storage::External);
        let w = value(&mut b, "W", Storage::External);
        let t = value(&mut b, "T", storage);
        let y = value(&mut b, "Y", Storage::External);
        (a, w, t, y)
    };
    b.bind_input("A", a);
    b.bind_input("W", w);
    let (first, second) = if reverse_operations {
        let second = add(&mut b, [t, w], y);
        let first = add(&mut b, [a, w], t);
        (first, second)
    } else {
        let first = add(&mut b, [a, w], t);
        let second = add(&mut b, [t, w], y);
        (first, second)
    };
    let statements = if combined {
        vec![Statement::Loop(Loop {
            kind: LoopKind::Sequential,
            domain: LoopDomain {
                variable: "iteration".into(),
                start: IndexExpr::Constant(0),
                stop: IndexExpr::Constant(2),
                step: IndexExpr::Constant(1),
            },
            body: vec![Statement::Operation(first), Statement::Operation(second)],
        })]
    } else {
        vec![Statement::Operation(first), Statement::Operation(second)]
    };
    b.build(statements, "Y", y)
}

#[test]
fn canonicalizes_branch_insertion_order_and_uses_builtin_hash() {
    let a = manual_plan(false, false, Storage::Global, false).unwrap();
    let b = manual_plan(true, true, Storage::Global, false).unwrap();
    assert!(a.same_body(&b));
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.hash(), super::plan::hash_plan(&a));
    assert!(
        a.statements()
            .iter()
            .all(|s| matches!(s, Statement::Operation(_)))
    );
}
#[test]
fn hash_covers_storage_and_statement_graph() {
    let a = manual_plan(false, false, Storage::Global, false).unwrap();
    let b = manual_plan(false, false, Storage::Global, true).unwrap();
    let c = manual_plan(false, false, Storage::Shared, true).unwrap();
    assert_ne!(a.hash(), b.hash());
    assert_ne!(b.hash(), c.hash());
    assert!(!a.same_body(&b));
    assert!(!b.same_body(&c));
}
#[test]
fn finalization_rejects_boundary_and_cross_statement_storage_errors() {
    let mut b = builder();
    let x = value(&mut b, "X", Storage::External);
    b.bind_input("X", x);
    b.bind_input("X", x);
    assert!(matches!(
        b.build(vec![], "Y", x),
        Err(PhysicalInvariantError::DuplicateInputTensor { .. })
    ));
    assert!(matches!(
        manual_plan(false, false, Storage::Shared, false),
        Err(PhysicalInvariantError::CrossStatementStorage {
            storage: Storage::Shared,
            ..
        })
    ));
}
#[test]
fn finalization_rejects_uninitialized_reads_and_missing_statement_membership() {
    let mut b = builder();
    let a = value(&mut b, "A", Storage::External);
    let t = value(&mut b, "T", Storage::Global);
    let first = add(&mut b, [t, t], a);
    let second = add(&mut b, [a, a], t);
    let statements = vec![Statement::Operation(first), Statement::Operation(second)];
    assert!(matches!(
        b.build(statements, "A", a),
        Err(PhysicalInvariantError::MissingProducer { .. })
    ));
    let mut b = builder();
    let x = value(&mut b, "X", Storage::External);
    let y = value(&mut b, "Y", Storage::External);
    b.bind_input("X", x);
    add(&mut b, [x, x], y);
    assert!(matches!(
        b.build(vec![], "Y", y),
        Err(PhysicalInvariantError::MissingStatementMembership { .. })
    ));
}
#[test]
fn separate_ordered_stores_can_repeat_the_same_computation() {
    let mut b = builder();
    let a = value(&mut b, "A", Storage::External);
    let w = value(&mut b, "W", Storage::External);
    let t = value(&mut b, "T", Storage::Global);
    let u = value(&mut b, "U", Storage::Global);
    let y = value(&mut b, "Y", Storage::External);
    b.bind_input("A", a);
    b.bind_input("W", w);
    let first = add(&mut b, [a, w], t);
    let duplicate = add(&mut b, [a, w], u);
    let consumer = add(&mut b, [t, w], y);
    let statements = [first, duplicate, consumer]
        .into_iter()
        .map(Statement::Operation)
        .collect();
    let p = b.build(statements, "Y", y).unwrap();
    assert_eq!(p.operations().len(), 3);
    assert_eq!(p.statements().len(), 3);
}
#[test]
fn input_aliases_share_one_canonical_value_and_are_order_independent() {
    let build = |reverse| {
        let mut b = builder();
        let x = value(&mut b, "X", Storage::External);
        for name in if reverse {
            ["second", "first"]
        } else {
            ["first", "second"]
        } {
            b.bind_input(name, x);
        }
        b.build(vec![], "output", x).unwrap()
    };
    let a = build(false);
    let b = build(true);
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.value_instances().count(), 1);
    assert_eq!(a.inputs()[0].value(), a.inputs()[1].value());
    assert_eq!(a.output().value(), a.inputs()[0].value());
}

#[test]
fn reduction_destination_is_zero_initialized_even_after_an_earlier_store() {
    for initialized in [false, true] {
        let mut b = builder();
        let x = value(&mut b, "X", Storage::External);
        let y = value(&mut b, "Y", Storage::External);
        b.bind_input("X", x);
        let mut statements = Vec::new();
        if initialized {
            let expression = Expression::Store {
                destination: TensorAccess::new(y, [AccessIndex::FullTile, AccessIndex::FullTile]),
                value: Box::new(Expression::Constant(Constant::Integer(7))),
            };
            let id = b.add_operation([], [y], expression);
            statements.push(Statement::Operation(id));
        }
        let reduction = add(&mut b, [y, x], y);
        statements.push(Statement::Loop(Loop {
            kind: LoopKind::Sequential,
            domain: LoopDomain {
                variable: "k".into(),
                start: IndexExpr::Constant(0),
                stop: IndexExpr::Constant(2),
                step: IndexExpr::Constant(1),
            },
            body: vec![Statement::Operation(reduction)],
        }));
        let plan = b.build(statements, "Y", y).unwrap();
        let reduction = plan
            .operation(plan.statements().last().unwrap().operations()[0])
            .unwrap();
        assert_eq!(reduction.inflows(), &[plan.inputs()[0].value()]);
        assert!(!reduction.inflows().contains(&plan.output().value()));
        assert!(crate::plan::accumulation_rhs(reduction.expression(), "lv0").is_some());
    }
}

#[test]
fn destination_load_outside_a_reduction_requires_a_producer() {
    let mut b = builder();
    let x = value(&mut b, "X", Storage::External);
    let y = value(&mut b, "Y", Storage::External);
    b.bind_input("X", x);
    let op = add(&mut b, [y, x], y);
    assert!(matches!(
        b.build(vec![Statement::Operation(op)], "Y", y),
        Err(PhysicalInvariantError::MissingProducer { .. })
    ));
}

#[test]
fn typed_expressions_validate_references_and_reject_nested_effects() {
    let mut b = builder();
    let x = value(&mut b, "X", Storage::External);
    let y = value(&mut b, "Y", Storage::External);
    b.bind_input("X", x);
    let access = |id| TensorAccess::new(id, [AccessIndex::FullTile, AccessIndex::FullTile]);
    for (rhs, diagnostic) in [
        (
            Expression::Load(TensorAccess::new(x, [AccessIndex::FullTile])),
            "access rank",
        ),
        (
            Expression::Load(TensorAccess::new(
                x,
                [AccessIndex::FullTile, AccessIndex::Elem("missing".into())],
            )),
            "unbound index",
        ),
        (
            Expression::Sqr(Box::new(Expression::Store {
                destination: access(y),
                value: Box::new(Expression::Load(access(x))),
            })),
            "pure computation",
        ),
        (
            Expression::AllGather {
                source: access(x),
                destination: access(y),
                axis: 1,
            },
            "pure computation",
        ),
    ] {
        let mut b = b.clone();
        let op = b.add_operation(
            [x],
            [y],
            Expression::Store {
                destination: access(y),
                value: Box::new(rhs),
            },
        );
        let error = b
            .build(vec![Statement::Operation(op)], "Y", y)
            .err()
            .unwrap();
        assert!(error.to_string().contains(diagnostic), "{error}");
    }

    let op = b.add_operation(
        [x],
        [y],
        Expression::Store {
            destination: access(y),
            value: Box::new(Expression::Load(access(ValueInstanceId::from_index(100)))),
        },
    );
    assert!(matches!(
        b.build(vec![Statement::Operation(op)], "Y", y),
        Err(PhysicalInvariantError::InvalidValueId {
            value: 100,
            context: "expression operand"
        })
    ));
}
