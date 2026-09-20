use trinity_lowering::*;
const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Hopper);
fn atom(s: impl ToString) -> Expression {
    Expression::Atom(s.to_string())
}
fn e(op: &str, args: impl IntoIterator<Item = Expression>) -> Expression {
    Expression::List(std::iter::once(atom(op)).chain(args).collect())
}
fn view(name: &str, role: &str, shape: &[usize]) -> Expression {
    e(
        "view",
        [
            e(role, [atom(name)]),
            e(
                "layout",
                shape
                    .iter()
                    .enumerate()
                    .map(|(i, n)| e("axis", [atom(format!("a{i}")), atom(n)])),
            ),
        ],
    )
}
fn idx(parts: Vec<Expression>) -> Expression {
    e(
        "keyed_index",
        parts
            .into_iter()
            .enumerate()
            .map(|(i, p)| e("slot", [atom(format!("a{i}")), p])),
    )
}
fn tile(var: &str, width: usize) -> Expression {
    e("tile", [atom(var), atom(width)])
}
fn load(v: Expression, i: Expression) -> Expression {
    e("load", [v, i])
}
fn gemm_instance() -> ImplementationInstance {
    gemm_implementations(TARGET)[0]
        .enumerate([DType::Bf16; 3], [&[128, 128]; 3])
        .pop()
        .unwrap()
}
fn gemm_text() -> String {
    let a = view("A", "input", &[128, 128]);
    let b = view("B", "input", &[128, 128]);
    let c = view("C", "output", &[128, 128]);
    let ci = idx(vec![tile("m", 128), tile("n", 128)]);
    let expr = e(
        "store",
        [
            c.clone(),
            e(
                "+",
                [
                    load(c, ci.clone()),
                    e(
                        "@",
                        [
                            load(a, idx(vec![tile("m", 128), tile("k", 64)])),
                            load(b, idx(vec![tile("k", 64), tile("n", 128)])),
                        ],
                    ),
                ],
            ),
            ci,
        ],
    );
    fn print(e: &Expression) -> String {
        match e {
            Expression::Atom(s) => s.clone(),
            Expression::List(xs) => {
                format!("({})", xs.iter().map(print).collect::<Vec<_>>().join(" "))
            }
        }
    }
    format!(
        "(ploop 0 128 128 m (ploop 0 128 128 n (sloop 0 128 64 k {})))",
        print(&expr)
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
        let op = b.add_operation(
            [a, w],
            [c],
            OperationPayload::Compute(ComputeOperation::new(gemm_instance())),
        );
        b.add_statement(Statement::Operation(op));
        let builder = b.finalize("C", c).unwrap();
        let config = LoopIrConfig {
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
