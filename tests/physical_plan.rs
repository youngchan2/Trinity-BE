mod support;
use support::explicit::*;
use trinity_lowering::*;

fn gemm_plan(gather_weight: bool) -> PhysicalPlan {
    let target = TargetCapability::Cuda(CudaTargetCapability::Hopper);
    let world_size = if gather_weight { 2 } else { 1 };
    let mut builder = PhysicalPlanBuilder::new(target, world_size);
    let x = builder.add_named_value("X", DType::Bf16, [128, 64], Storage::External);
    let weight_shape = if gather_weight { [64, 128] } else { [64, 256] };
    let w = builder.add_named_value("W", DType::Bf16, weight_shape, Storage::External);
    let y = builder.add_named_value("Y", DType::Bf16, [128, 256], Storage::External);
    builder.bind_input("X", x);
    builder.bind_input("W", w);

    let mut statements = Vec::new();
    let w = if gather_weight {
        let gathered = builder.add_named_value("G", DType::Bf16, [64, 256], Storage::Global);
        let access = [AccessIndex::FullTile, tile("chunk", 128)];
        let operation = builder.add_operation(
            [w],
            [gathered],
            Expression::AllGather {
                source: TensorAccess::new(w, access.clone()),
                destination: TensorAccess::new(gathered, access),
                axis: 1,
            },
        );
        statements.push(loop_node(
            LoopKind::Parallel,
            "chunk",
            0,
            128,
            128,
            vec![Statement::Operation(operation)],
        ));
        gathered
    } else {
        w
    };

    let ci = [tile("m", 128), tile("n", 128)];
    let rhs = Expression::Add(Box::new([
        load(y, ci.clone()),
        Expression::Matmul(Box::new([
            load(x, [tile("m", 128), tile("k", 64)]),
            load(w, [tile("k", 64), tile("n", 128)]),
        ])),
    ]));
    let operation = builder.add_operation([x, w], [y], store(y, rhs, ci));
    statements.push(loop_node(
        LoopKind::Parallel,
        "m",
        0,
        128,
        128,
        vec![loop_node(
            LoopKind::Parallel,
            "n",
            0,
            256,
            128,
            vec![loop_node(
                LoopKind::Sequential,
                "k",
                0,
                64,
                64,
                vec![Statement::Operation(operation)],
            )],
        )],
    ));
    builder.build(statements, "Y", y).unwrap()
}

#[test]
fn builds_a_single_gpu_gemm_through_the_public_api() {
    let plan = gemm_plan(false);
    assert_eq!(plan.world_size(), 1);
    assert_eq!(plan.operations().len(), 1);
    assert_eq!(plan.statements().len(), 1);
    assert_eq!(plan.inputs()[0].tensor(), "W");
    assert_eq!(plan.inputs()[1].tensor(), "X");
    assert_eq!(plan.output().tensor(), "Y");
    let statement = &plan.statements()[0];
    let statement = plan.operation(statement.operations()[0]).unwrap();
    assert_eq!(statement.inflows().len(), 2);
    assert_eq!(statement.outflows(), &[plan.output().value()]);
    assert!(plan.same_body(&gemm_plan(false)));
}

#[test]
fn builds_all_gather_and_gemm_through_the_same_public_api() {
    let plan = gemm_plan(true);
    assert_eq!(plan.world_size(), 2);
    assert_eq!(plan.operations().len(), 2);
    let [gather, gemm] = plan.statements() else {
        panic!("expected gather and GEMM statements in program order");
    };
    let gather = plan.operation(gather.operations()[0]).unwrap();
    let gemm = plan.operation(gemm.operations()[0]).unwrap();
    let gathered = gather.outflows()[0];
    let Expression::AllGather {
        source,
        destination,
        axis,
    } = gather.expression()
    else {
        panic!("expected all-gather");
    };
    assert_eq!(source.value, plan.inputs()[0].value());
    assert_eq!(destination.value, gathered);
    assert_eq!(*axis, 1);
    assert_eq!(source.indices[1], tile("lv0", 128));
    assert!(gemm.inflows().contains(&gathered));
    assert_eq!(gemm.outflows(), &[plan.output().value()]);
    let value = plan.value_instance(gathered).unwrap();
    assert_eq!(value.shape(), &[64, 256]);
    assert_eq!(value.storage(), Storage::Global);
    assert!(plan.same_body(&gemm_plan(true)));
}
