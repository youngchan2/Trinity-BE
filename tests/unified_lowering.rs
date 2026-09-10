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
fn parallel(var: &str, stop: usize, step: usize, body: Vec<Statement>) -> Statement {
    Statement::Loop(Loop {
        kind: LoopKind::Parallel,
        domain: LoopDomain {
            variable: var.into(),
            start: IndexExpr::Constant(0),
            stop: IndexExpr::Constant(stop as i64),
            step: IndexExpr::Constant(step as i64),
        },
        body,
    })
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
        let text = lower_loop_ir(&gemm_text(), &config).unwrap().remove(0);
        assert!(builder.same_body(&text));
        assert_eq!(builder.hash(), text.hash());
        let a = emit(&builder).unwrap();
        let b = emit(&text).unwrap();
        assert_eq!(a.code(), b.code());
    }
}

/// Explicit scheduled compute, normalized communication, and normalized compute
/// all share one program and one dependence resolver.
pub fn mixed(backend: usize) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TARGET, 2);
    let x = b.add_named_value("X", DType::Bf16, [128, 128], Storage::External);
    b.bind_input("X", x);
    let y = b.add_named_value("Y", DType::Bf16, [128, 128], Storage::Global);
    let v = view("Y", "tensor", &[128, 128]);
    let index = idx(vec![tile("r", 64), tile("c", 64)]);
    let expression = e(
        "store",
        [
            v,
            e(
                "relu",
                [load(view("X", "input", &[128, 128]), index.clone())],
            ),
            index,
        ],
    );
    let relu = pointwise_implementations(TARGET)
        .iter()
        .flat_map(|i| i.enumerate(&[DType::Bf16; 2], &[&[128, 128][..]; 2], None))
        .find(|i| i.id().as_str() == "cuda.relu")
        .unwrap();
    let id = b.add_expression([x], [y], expression, relu);
    b.add_statement(parallel(
        "r",
        128,
        64,
        vec![parallel("c", 128, 64, vec![Statement::Operation(id)])],
    ));
    let gathered = b.add_named_value("G", DType::Bf16, [128, 256], Storage::Global);
    let instance = all_gather_implementations(TARGET)[backend]
        .enumerate(DType::Bf16, [&[128, 128], &[128, 256]], 1, 2)
        .pop()
        .unwrap();
    let id = b.add_operation(
        [y],
        [gathered],
        OperationPayload::Communication(CommunicationOperation::new(
            CommunicationKind::AllGather,
            instance,
        )),
    );
    b.add_statement(Statement::Operation(id));
    let w = b.add_named_value("W", DType::Bf16, [256, 128], Storage::External);
    b.bind_input("W", w);
    let out = b.add_named_value("Out", DType::Bf16, [128, 128], Storage::External);
    let instance = gemm_implementations(TARGET)[0]
        .enumerate([DType::Bf16; 3], [&[128, 256], &[256, 128], &[128, 128]])
        .pop()
        .unwrap();
    let id = b.add_operation(
        [gathered, w],
        [out],
        OperationPayload::Compute(ComputeOperation::new(instance)),
    );
    b.add_statement(Statement::Operation(id));
    b.finalize("Out", out).unwrap()
}
#[test]
fn mixed_program_keeps_communication_resources_and_stage_dependencies() {
    for backend in 0..3 {
        let plan = mixed(backend);
        let source = emit(&plan).unwrap();
        assert!(source.requirements().nvshmem);
        assert_eq!(source.requirements().nvls, backend == 0);
        assert!(source.requirements().buffers.iter().any(|b| b.symmetric));
        let gemm: Vec<_> = source
            .execution()
            .tasks
            .iter()
            .filter(|t| !t.stages.is_empty())
            .collect();
        assert_eq!(gemm.len(), 2);
        assert!(gemm.iter().all(|t|t.stages.len()==4&&t.stages.iter().all(|s|!s.dependencies.is_empty())));
        if backend != 0 {
            assert_eq!(source.requirements().minimum_workers, 2);
        }
    }
}
#[test]
#[ignore = "requires NVCC and NVSHMEM; never executes GPU work"]
fn mixed_programs_compile_and_link() {
    for backend in 0..3 {
        let source = emit(&mixed(backend)).unwrap();
        let _artifact = compile(source).unwrap();
    }
}

fn copy(v: &str, role: &str, input: &str, input_role: &str) -> Expression {
    let index = idx(vec![atom("fulltile")]);
    e(
        "store",
        [
            view(v, role, &[128]),
            load(view(input, input_role, &[128]), index.clone()),
            index,
        ],
    )
}
#[test]
fn ordered_rewrites_preserve_read_and_write_hazards() {
    let mut b = PhysicalPlanBuilder::new(TARGET, 2);
    let x = b.add_named_value("X", DType::Fp32, [128], Storage::External);
    b.bind_input("X", x);
    let temp = b.add_named_value("T", DType::Fp32, [128], Storage::Global);
    let out = b.add_named_value("Y", DType::Fp32, [128], Storage::External);
    let imp = pointwise_implementations(TARGET)
        .iter()
        .flat_map(|i| i.enumerate(&[DType::Fp32; 2], &[&[128][..]; 2], None))
        .next()
        .unwrap();
    for (ins, dest, expr) in [
        ([x], temp, copy("T", "tensor", "X", "input")),
        ([temp], out, copy("Y", "output", "T", "tensor")),
        ([x], temp, copy("T", "tensor", "X", "input")),
    ] {
        let id = b.add_expression(ins, [dest], expr, imp.clone());
        b.add_statement(Statement::Operation(id));
    }
    let plan = b.finalize("Y", out).unwrap();
    let source = emit(&plan).unwrap();
    for rank in 0..2 {
        let t = &source.execution().tasks[rank * 3..rank * 3 + 3];
        assert!(
            t[2].dependencies
                .contains(&trinity_lowering::emit::Dependency { rank, slot: 1 })
        );
    }
}
#[test]
fn parallel_overlapping_writers_are_rejected() {
    let mut b = PhysicalPlanBuilder::new(TARGET, 1);
    let x = b.add_named_value("X", DType::Fp32, [128], Storage::External);
    b.bind_input("X", x);
    let y = b.add_named_value("Y", DType::Fp32, [128], Storage::External);
    let imp = pointwise_implementations(TARGET)
        .iter()
        .flat_map(|i| i.enumerate(&[DType::Fp32; 2], &[&[128][..]; 2], None))
        .next()
        .unwrap();
    let id = b.add_expression([x], [y], copy("Y", "output", "X", "input"), imp);
    b.add_statement(parallel("i", 2, 1, vec![Statement::Operation(id)]));
    assert!(matches!(
        emit(&b.finalize("Y", y).unwrap()),
        Err(EmitError::Contract(_))
    ));
}
