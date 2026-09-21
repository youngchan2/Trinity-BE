mod support;
use support::explicit::{load, loop_node, store, tile};
use trinity_lowering::*;

const TARGETS: [CudaTargetCapability; 3] = [
    CudaTargetCapability::Hopper,
    CudaTargetCapability::Sm89,
    CudaTargetCapability::Sm120,
];

fn copy_text(kind: &str, start: i64, stop: i64) -> String {
    format!(
        "({kind} {start} {stop} 64 i (store (view (output Y) (layout (axis a 128))) (load (view (input X) (layout (axis a 128))) (keyed_index (slot a (tile i 64)))) (keyed_index (slot a (tile i 64)))))"
    )
}
fn copy_ir(text: &str, target: CudaTargetCapability) -> PhysicalPlan {
    lower_ir(
        text,
        &IrConfig {
            target: TargetCapability::Cuda(target),
            dtypes: [("X".into(), DType::Fp32), ("Y".into(), DType::Fp32)].into(),
            ..Default::default()
        },
    )
    .unwrap()
    .remove(0)
}
fn identity(target: CudaTargetCapability) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(target), 1);
    let x = b.add_value(DType::Fp32, [128], Storage::External);
    b.bind_input("X with \"quotes\"", x);
    b.build(vec![], "Y", x).unwrap()
}
fn gemm(target: CudaTargetCapability, storage: Storage) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(target), 1);
    let x = b.add_value(DType::Bf16, [16, 128], Storage::External);
    let w = b.add_value(DType::Bf16, [128, 256], Storage::External);
    let c = b.add_value(DType::Bf16, [16, 256], storage);
    let y = b.add_value(DType::Bf16, [16, 256], Storage::External);
    b.bind_input("X", x);
    b.bind_input("W", w);
    let ci = [AccessIndex::FullTile, tile("n", 128)];
    let rhs = Expression::Add(Box::new([
        load(c, ci.clone()),
        Expression::Matmul(Box::new([
            load(x, [AccessIndex::FullTile, tile("k", 64)]),
            load(w, [tile("k", 64), tile("n", 128)]),
        ])),
    ]));
    let mm = b.add_operation([x, w], [c], store(c, rhs, ci.clone()));
    let relu = b.add_operation(
        [c],
        [y],
        store(y, Expression::Relu(Box::new(load(c, ci.clone()))), ci),
    );
    let serial = loop_node(
        LoopKind::Sequential,
        "k",
        0,
        128,
        64,
        vec![Statement::Operation(mm)],
    );
    b.build(
        vec![loop_node(
            LoopKind::Parallel,
            "n",
            0,
            256,
            128,
            vec![serial, Statement::Operation(relu)],
        )],
        "Y",
        y,
    )
    .unwrap()
}
fn reduction(target: CudaTargetCapability) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(target), 1);
    let x = b.add_value(DType::Fp32, [5, 131], Storage::External);
    let y = b.add_value(DType::Fp32, [5], Storage::External);
    b.bind_input("X", x);
    let rhs = Expression::Add(Box::new([
        load(y, [AccessIndex::FullTile]),
        Expression::ReduceSum {
            axis: 1,
            value: Box::new(load(
                x,
                [
                    AccessIndex::FullTile,
                    AccessIndex::ClippedTile {
                        variable: "k".into(),
                        width: 65,
                    },
                ],
            )),
        },
    ]));
    let op = b.add_operation([x], [y], store(y, rhs, [AccessIndex::FullTile]));
    b.build(
        vec![loop_node(
            LoopKind::Sequential,
            "k",
            0,
            131,
            65,
            vec![Statement::Operation(op)],
        )],
        "Y",
        y,
    )
    .unwrap()
}
#[test]
fn native_families_emit_on_each_target_with_per_kernel_resources() {
    for target in TARGETS {
        for plan in [
            identity(target),
            copy_ir(&copy_text("ploop", 0, 128), target),
            copy_ir(&copy_text("sloop", 0, 128), target),
            gemm(target, Storage::Global),
            gemm(target, Storage::Register),
            reduction(target),
        ] {
            let source = emit(&plan).unwrap();
            assert_eq!(source.requirements().target, target);
            assert_eq!(source.requirements().workspace_bytes, 0);
            assert!(!source.requirements().nvshmem);
            for b in &source.requirements().buffers {
                assert!(b.alignment >= b.dtype.size_bytes());
            }
            assert_eq!(source.code(), emit(&plan).unwrap().code());
        }
    }
}

#[test]
fn builder_and_ir_emit_identical_copy_and_ordered_nonzero_ranges() {
    let target = CudaTargetCapability::Hopper;
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(target), 1);
    let x = b.add_named_value("X", DType::Fp32, [128], Storage::External);
    let y = b.add_named_value("Y", DType::Fp32, [128], Storage::External);
    b.bind_input("X", x);
    let index = [tile("i", 64)];
    let op = b.add_operation([x], [y], store(y, load(x, index.clone()), index));
    let plan = b
        .build(
            vec![loop_node(
                LoopKind::Parallel,
                "i",
                0,
                128,
                64,
                vec![Statement::Operation(op)],
            )],
            "Y",
            y,
        )
        .unwrap();
    let ir = copy_ir(&copy_text("ploop", 0, 128), target);
    assert!(plan.same_body(&ir));
    assert_eq!(emit(&plan).unwrap().code(), emit(&ir).unwrap().code());
    let text = format!(
        "(seq {} {})",
        copy_text("ploop", 0, 64),
        copy_text("ploop", 64, 128)
    );
    let source = emit(&copy_ir(&text, target)).unwrap();
    assert_eq!(source.code().matches("<<<").count(), 2);
    assert!(source.code().contains("64LL +"));
}

#[test]
fn rejects_incomplete_outputs_unordered_ctas_and_overflow() {
    let target = CudaTargetCapability::Hopper;
    assert!(matches!(
        emit(&copy_ir(&copy_text("ploop", 64, 128), target)),
        Err(EmitError::InvalidExecution { .. })
    ));
    let text = copy_text("ploop", 0, 128).replace("0 128 64 i", "0 128 32 i");
    assert!(matches!(
        emit(&copy_ir(&text, target)),
        Err(EmitError::InvalidExecution { .. })
    ));
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(target), 1);
    let x = b.add_value(DType::Fp32, [usize::MAX, 2], Storage::External);
    b.bind_input("X", x);
    let plan = b.build(vec![], "Y", x).unwrap();
    assert!(matches!(
        emit(&plan),
        Err(EmitError::InvalidExecution { .. })
    ));
}

#[test]
fn rejects_unsupported_execution_without_falling_back() {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Sm89), 2);
    let x = b.add_value(DType::Fp32, [4], Storage::External);
    b.bind_input("X", x);
    assert!(matches!(
        emit(&b.build(vec![], "Y", x).unwrap()),
        Err(EmitError::UnsupportedExecution { .. })
    ));
    let text = format!(
        "(ploop 0 2 1 outer {})",
        copy_text("ploop", 0, 128).replace("0 128 64 i", "outer 128 64 i")
    );
    assert!(matches!(
        emit(&copy_ir(&text, CudaTargetCapability::Hopper)),
        Err(EmitError::UnsupportedExecution { .. })
    ));
}

fn tail_pipeline(target: CudaTargetCapability, inner: LoopKind) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(target), 1);
    let x = b.add_value(DType::Bf16, [3, 131], Storage::External);
    let t = b.add_value(DType::Bf16, [3, 131], Storage::Global);
    let y = b.add_value(DType::Fp32, [3, 131], Storage::External);
    b.bind_input("X", x);
    let ix = [
        tile("row", 1),
        AccessIndex::ClippedTile {
            variable: "col".into(),
            width: 64,
        },
    ];
    let a = b.add_operation(
        [x],
        [t],
        store(
            t,
            Expression::Sqr(Box::new(load(x, ix.clone()))),
            ix.clone(),
        ),
    );
    let z = b.add_operation(
        [t],
        [y],
        store(y, Expression::Sigmoid(Box::new(load(t, ix.clone()))), ix),
    );
    let columns = loop_node(
        inner,
        "col",
        0,
        131,
        64,
        vec![Statement::Operation(a), Statement::Operation(z)],
    );
    b.build(
        vec![loop_node(LoopKind::Parallel, "row", 0, 3, 1, vec![columns])],
        "Y",
        y,
    )
    .unwrap()
}

#[test]
fn clipped_tails_and_ordinary_serial_bodies_keep_their_launch_scope() {
    for target in TARGETS {
        for inner in [LoopKind::Parallel, LoopKind::Sequential] {
            let source = emit(&tail_pipeline(target, inner)).unwrap();
            assert_eq!(source.code().matches("<<<").count(), 1);
            assert_eq!(
                source.code().contains("for (int64_t serial_"),
                inner == LoopKind::Sequential
            );
            assert_eq!(source.requirements().buffers.len(), 3);
        }
    }
}

#[test]
fn intermediate_coverage_and_cross_cta_dependencies_require_launch_order() {
    let config = IrConfig {
        dtypes: ["X", "T", "Y"].map(|n| (n.into(), DType::Fp32)).into(),
        ..Default::default()
    };
    let producer = copy_text("ploop", 0, 64).replace("(output Y)", "(tensor T)");
    let consumer = copy_text("ploop", 0, 128).replace("(input X)", "(tensor T)");
    let text = format!("(seq {producer} {consumer})");
    let plan = lower_ir(&text, &config).unwrap().remove(0);
    assert!(
        matches!(emit(&plan), Err(EmitError::InvalidExecution { reason }) if reason.contains("unproduced"))
    );
    let text = text.replace("ploop 0 64 64", "ploop 0 128 64");
    assert_eq!(
        emit(&lower_ir(&text, &config).unwrap()[0])
            .unwrap()
            .code()
            .matches("<<<")
            .count(),
        2
    );

    let mut b = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
    let x = b.add_value(DType::Fp32, [128], Storage::External);
    let t = b.add_value(DType::Fp32, [128], Storage::Global);
    let y = b.add_value(DType::Fp32, [128], Storage::External);
    b.bind_input("X", x);
    let a = b.add_operation(
        [x],
        [t],
        store(t, load(x, [tile("i", 64)]), [tile("i", 64)]),
    );
    let z = b.add_operation(
        [t],
        [y],
        store(y, load(t, [AccessIndex::FullTile]), [AccessIndex::FullTile]),
    );
    let plan = b
        .build(
            vec![loop_node(
                LoopKind::Parallel,
                "i",
                0,
                128,
                64,
                vec![Statement::Operation(a), Statement::Operation(z)],
            )],
            "Y",
            y,
        )
        .unwrap();
    assert!(matches!(emit(&plan),Err(EmitError::Combination { reason }) if reason.contains("CTA")));
}

#[test]
#[ignore = "requires CUDA 13 NVCC and vendored CUTLASS; no GPU required"]
fn emitted_tail_and_serial_programs_compile() {
    for target in TARGETS {
        for inner in [LoopKind::Parallel, LoopKind::Sequential] {
            compile(emit(&tail_pipeline(target, inner)).unwrap())
                .unwrap_or_else(|e| panic!("{target} {inner:?}: {e}"));
            eprintln!("compiled {target} {inner:?} tail");
        }
    }
}

#[test]
#[ignore = "requires CUDA 13 NVCC and vendored CUTLASS; no GPU required"]
fn emitted_programs_compile_and_link_on_all_targets() {
    for target in TARGETS {
        let plans = [
            identity(target),
            copy_ir(&copy_text("ploop", 0, 128), target),
            reduction(target),
            gemm(target, Storage::Global),
            gemm(target, Storage::Register),
        ];
        for (case, plan) in plans.iter().enumerate() {
            let source = emit(plan).unwrap();
            let artifact = compile(source).unwrap_or_else(|e| panic!("{target} case {case}: {e}"));
            let symbols = std::process::Command::new("nm")
                .args(["-D", "--defined-only"])
                .arg(artifact.artifact_path())
                .output()
                .unwrap();
            let symbols = String::from_utf8_lossy(&symbols.stdout);
            for name in [
                "trinity_abi",
                "trinity_prepare",
                "trinity_launch",
                "trinity_status",
                "trinity_release",
            ] {
                assert!(symbols.contains(name));
            }
            eprintln!("compiled {target} case {case}");
        }
    }
}
