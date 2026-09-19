mod support;
use trinity_lowering::*;
const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Hopper);
use support::explicit::{load, store, tile};
fn gemm_expression(a: ValueInstanceId, b: ValueInstanceId, c: ValueInstanceId) -> Expression {
    let ci = [tile("m", 128), tile("n", 128)];
    store(
        c,
        Expression::Add(Box::new([
            load(c, ci.clone()),
            Expression::Matmul(Box::new([
                load(a, [tile("m", 128), tile("k", 64)]),
                load(b, [tile("k", 64), tile("n", 128)]),
            ])),
        ])),
        ci,
    )
}
fn gemm_text() -> String {
    let view = |name, role| format!("(view ({role} {name}) (layout (axis a0 128) (axis a1 128)))");
    let ci = "(keyed_index (slot a0 (tile m 128)) (slot a1 (tile n 128)))";
    format!(
        "(ploop 0 128 128 m (ploop 0 128 128 n (sloop 0 128 64 k
      (store {} (+ (load {} {ci}) (@
        (load {} (keyed_index (slot a0 (tile m 128)) (slot a1 (tile k 64))))
        (load {} (keyed_index (slot a0 (tile k 64)) (slot a1 (tile n 128)))))) {ci}))))",
        view("C", "output"),
        view("C", "output"),
        view("A", "input"),
        view("B", "input")
    )
}

#[test]
fn builder_and_text_produce_the_same_scheduled_gemm() {
    for world_size in [1, 2] {
        let mut b = PhysicalPlanBuilder::new(TARGET, world_size);
        let a = b.add_named_value("A", DType::Bf16, [128, 128], Storage::External);
        b.bind_input("A", a);
        let w = b.add_named_value("B", DType::Bf16, [128, 128], Storage::External);
        b.bind_input("B", w);
        let c = b.add_named_value("C", DType::Bf16, [128, 128], Storage::External);
        let op = b.add_operation([a, w], [c], gemm_expression(a, w, c));
        use support::explicit::loop_node;
        let statement = loop_node(
            LoopKind::Parallel,
            "m",
            0,
            128,
            128,
            vec![loop_node(
                LoopKind::Parallel,
                "n",
                0,
                128,
                128,
                vec![loop_node(
                    LoopKind::Sequential,
                    "k",
                    0,
                    128,
                    64,
                    vec![Statement::Operation(op)],
                )],
            )],
        );
        let builder = b.build(vec![statement], "C", c).unwrap();
        let config = IrConfig {
            world_size,
            dtypes: [
                ("A".into(), DType::Bf16),
                ("B".into(), DType::Bf16),
                ("C".into(), DType::Bf16),
            ]
            .into(),
            ..Default::default()
        };
        let text = lower_ir(&gemm_text(), &config).unwrap().remove(0);
        assert!(builder.same_body(&text));
        assert_eq!(builder.hash(), text.hash());
    }
}

#[test]
fn ir_preserves_gemm_without_selecting_a_supported_cuda_implementation() {
    // The former Hopper selector rejected FP32 and these tile widths in the Reader.
    let config = IrConfig {
        dtypes: ["A", "B", "C"]
            .map(|name| (name.into(), DType::Fp32))
            .into(),
        ..Default::default()
    };
    let text = gemm_text()
        .replace("128 128 m", "128 32 m")
        .replace("128 128 n", "128 32 n")
        .replace("128 64 k", "128 16 k")
        .replace("tile m 128", "tile m 32")
        .replace("tile n 128", "tile n 32")
        .replace("tile k 64", "tile k 16");
    let plan = lower_ir(&text, &config).unwrap().remove(0);
    let (_, op) = plan.operations().next().unwrap();
    let Expression::Store { destination, value } = op.expression() else {
        panic!("store")
    };
    assert_eq!(destination.indices[0], tile("lv0", 32));
    let Expression::Add(add) = value.as_ref() else {
        panic!("accumulation")
    };
    let Expression::Matmul(multiply) = &add[1] else {
        panic!("matmul")
    };
    let Expression::Load(input) = &multiply[0] else {
        panic!("load")
    };
    assert_eq!(input.indices[1], tile("lv2", 16));
    assert_eq!(op.inflows().len(), 2);
    assert!(!op.inflows().contains(&plan.output().value()));
    assert!(
        plan.value_instances()
            .all(|(_, v)| v.dtype() == DType::Fp32)
    );
}
