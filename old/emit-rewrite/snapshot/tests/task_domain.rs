//! Execution domains and Body ABI: sequential bounds must not create scheduler work.
#[allow(dead_code)]
mod support;
use trinity_lowering::*;

fn varying_k(world: usize) -> PhysicalPlan {
    let text = r#"(ploop 0 128 128 m (ploop 0 256 128 n (sloop 0 (+ n 128) 64 k
      (store (view (output C) (layout (axis a 128) (axis b 256)))
        (+ (load (view (output C) (layout (axis a 128) (axis b 256)))
                 (keyed_index (slot a (tile m 128)) (slot b (tile n 128))))
           (@ (load (view (input A) (layout (axis a 128) (axis b 256)))
                    (keyed_index (slot a (tile m 128)) (slot b (tile k 64))))
              (load (view (input B) (layout (axis a 256) (axis b 256)))
                    (keyed_index (slot a (tile k 64)) (slot b (tile n 128))))))
        (keyed_index (slot a (tile m 128)) (slot b (tile n 128)))))))"#;
    let config = LoopIrConfig {
        world_size: world,
        dtypes: ["A", "B", "C"].map(|s| (s.into(), DType::Bf16)).into(),
        ..Default::default()
    };
    lower_ir(text, &config).unwrap().remove(0)
}

fn indexed_copy(world: usize, dependent: bool) -> PhysicalPlan {
    let x = "(view (input X) (layout (axis a 4) (axis b 4)))";
    let y = "(view (output Y) (layout (axis a 4) (axis b 4)))";
    let all = "(keyed_index (slot a fulltile) (slot b fulltile))";
    let at = "(keyed_index (slot a (elem i)) (slot b (elem j)))";
    let stop = if dependent { "(+ i 2)" } else { "8" };
    let text = format!(
        "(seq (store {y} (load {x} {all}) {all}) (ploop 2 8 2 i (ploop 0 {stop} 2 j (store {y} (load {x} {at}) {at}))))"
    );
    let config = LoopIrConfig {
        world_size: world,
        dtypes: [("X".into(), DType::Fp32), ("Y".into(), DType::Fp32)].into(),
        ..Default::default()
    };
    lower_ir(&text, &config).unwrap().remove(0)
}

#[test]
fn coordinate_dependent_serial_bounds_share_one_body_and_keep_physical_stages() {
    for world in [1, 2] {
        let source = emit(&varying_k(world)).unwrap();
        assert_eq!(source.bodies().len(), 1);
        assert!(!source.code().contains("body_0_args"));
        assert!(source.code().contains("coordinates[1]"));
        match source.execution() {
            emit::Execution::Streamed(e) => {
                assert_eq!(e.tasks.len(), 1);
                assert_eq!(e.launches.len(), 1);
                assert_eq!(e.grids[0].axes, [1, 2]);
                assert_eq!(
                    e.grids[0].coordinates(&e.tasks[0], 1).unwrap(),
                    Some(vec![0, 128])
                );
                for forbidden in ["kTiles", "kArguments", "kStages", "kDependencies"] {
                    assert!(!source.code().contains(forbidden));
                }
            }
            emit::Execution::Persistent(e) => {
                let arguments = source
                    .code()
                    .split("kArguments[] = {")
                    .nth(1)
                    .unwrap()
                    .split("};")
                    .next()
                    .unwrap();
                let coordinates = arguments
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim_end_matches("LL").parse::<i64>().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(coordinates, [0, 0, 0, 128]); // shared across both ranks
                for rank in 0..world {
                    assert_eq!(
                        e.schedule
                            .iter()
                            .filter(|s| s.rank == rank)
                            .map(|s| s.stages.len())
                            .collect::<Vec<_>>(),
                        [2, 4]
                    );
                }
                assert!(e.tasks.iter().all(|t| t.body == 0 && t.bindings().is_ok()));
            }
        }
    }
}

#[test]
fn parallel_grid_matches_lexical_enumeration_with_start_step_and_padding() {
    for dependent in [false, true] {
        let source = emit(&indexed_copy(1, dependent)).unwrap();
        let e = source.execution().as_streamed().unwrap();
        assert_eq!(e.tasks.len(), 2);
        assert_eq!(e.launches.len(), 2);
        assert_eq!(e.grids[1].axes, [3, 4]);
        let actual = (0..e.grids[1].blocks)
            .filter_map(|i| e.grids[1].coordinates(&e.tasks[1], i).unwrap())
            .collect::<Vec<_>>();
        let expected = (2..8)
            .step_by(2)
            .flat_map(|i| {
                (0..if dependent { i + 2 } else { 8 })
                    .step_by(2)
                    .map(move |j| vec![i, j])
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        let persistent = emit(&indexed_copy(2, dependent)).unwrap();
        let pe = persistent.execution().as_persistent().unwrap();
        let singleton = pe
            .tasks
            .iter()
            .zip(&pe.schedule)
            .filter(|(t, s)| t.statement == 1 && s.rank == 0)
            .map(|(t, _)| {
                t.domain
                    .iter()
                    .map(|d| d.start.evaluate(&Default::default()).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(singleton, expected);
        // elem retains coordinate / original step, even for singleton domains.
        for t in pe.tasks.iter().filter(|t| t.statement == 1) {
            assert!(t.domain.iter().all(|d| d.step == IndexExpr::Constant(2)));
        }
        let origins = pe
            .tasks
            .iter()
            .zip(&pe.schedule)
            .zip(&pe.work)
            .filter(|((t, s), _)| t.statement == 1 && s.rank == 0)
            .map(|(_, w)| w.writes[0].origin[..2].to_vec())
            .collect::<Vec<_>>();
        assert_eq!(
            origins,
            expected
                .iter()
                .map(|c| c.iter().map(|v| (*v / 2) as usize).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn clipped_tails_share_body_and_validate_full_output_coverage() {
    for shape in [&[129][..], &[3, 129], &[2, 257]] {
        for world in [1, 2] {
            let source = emit(&support::tensor::pointwise(
                "relu",
                shape,
                &[DType::Fp32; 2],
                None,
                world,
            ))
            .unwrap();
            assert_eq!(source.bodies().len(), 1);
            let code = source.bodies()[0].render().unwrap();
            assert!(code.contains("std::min<std::int64_t>"));
            if let Some(e) = source.execution().as_streamed() {
                assert_eq!(e.tasks.len(), 1);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA 13.0+, CUTLASS and NVSHMEM; no GPU required"]
fn coordinate_bounds_and_shared_tails_compile_and_link() {
    for world in [1, 2] {
        for plan in [
            varying_k(world),
            indexed_copy(world, true),
            support::tensor::pointwise("relu", &[3, 129], &[DType::Fp32; 2], None, world),
        ] {
            compile(emit(&plan).unwrap()).unwrap_or_else(|e| panic!("{e}"));
        }
    }
}

fn reduction_tail(world: usize) -> PhysicalPlan {
    reduction_tail_storage(world, false)
}

fn reduction_tail_storage(world: usize, shared: bool) -> PhysicalPlan {
    fn atom(x: impl ToString) -> Expression {
        Expression::Atom(x.to_string())
    }
    fn expr(op: &str, args: impl IntoIterator<Item = Expression>) -> Expression {
        Expression::List(std::iter::once(atom(op)).chain(args).collect())
    }
    fn view(name: &str, role: &str, shape: &[usize]) -> Expression {
        expr(
            "view",
            [
                expr(role, [atom(name)]),
                expr(
                    "layout",
                    shape
                        .iter()
                        .enumerate()
                        .map(|(i, n)| expr("axis", [atom(format!("a{i}")), atom(n)])),
                ),
            ],
        )
    }
    fn index(input: bool) -> Expression {
        let mut slots = vec![expr(
            "slot",
            [atom("a0"), expr("clipped_tile", [atom("m"), atom(128)])],
        )];
        if input {
            slots.push(expr("slot", [atom("a1"), atom("fulltile")]));
        }
        expr("keyed_index", slots)
    }
    let mut b = PhysicalPlanBuilder::new(support::tensor::TARGET, world);
    let x = b.add_named_value("X", DType::Fp32, [129, 256], Storage::External);
    b.bind_input("X", x);
    let y = b.add_named_value(
        "Y",
        DType::Fp32,
        [129],
        if shared {
            Storage::Shared
        } else {
            Storage::External
        },
    );
    let instance = reduce_sum_implementations(support::tensor::TARGET)[0]
        .enumerate([DType::Fp32; 2], [&[129, 256], &[129]], 1)
        .pop()
        .unwrap();
    let op = b.add_expression(
        [x],
        [y],
        expr(
            "store",
            [
                view("Y", if shared { "tensor" } else { "output" }, &[129]),
                expr(
                    "rsum",
                    [
                        expr("load", [view("X", "input", &[129, 256]), index(true)]),
                        atom(1),
                    ],
                ),
                index(false),
            ],
        ),
        instance,
    );
    let mut body = vec![Statement::Operation(op)];
    let output = if shared {
        let z = b.add_named_value("Z", DType::Fp32, [129], Storage::External);
        let imp = pointwise_implementations(support::tensor::TARGET)
            .iter()
            .find(|d| d.id().as_str() == "cuda.relu")
            .unwrap()
            .enumerate(&[DType::Fp32; 2], &[&[129][..]; 2], None)
            .pop()
            .unwrap();
        body.push(Statement::Operation(b.add_expression(
            [y],
            [z],
            expr(
                "store",
                [
                    view("Z", "output", &[129]),
                    expr(
                        "relu",
                        [expr("load", [view("Y", "tensor", &[129]), index(false)])],
                    ),
                    index(false),
                ],
            ),
            imp,
        )));
        z
    } else {
        y
    };
    b.add_statement(Statement::Loop(Loop {
        kind: LoopKind::Parallel,
        domain: LoopDomain {
            variable: "m".into(),
            start: IndexExpr::Constant(0),
            stop: IndexExpr::Constant(256),
            step: IndexExpr::Constant(128),
        },
        body,
    }));
    b.finalize("Out", output).unwrap()
}

#[test]
fn singleton_reduction_tail_preserves_the_parallel_sum_order_in_a_shared_body() {
    for world in [1, 2] {
        let source = emit(&reduction_tail(world)).unwrap();
        assert_eq!(source.bodies().len(), 1);
        assert_eq!(source.requirements().shared_memory_bytes, 512);
        let code = source.bodies()[0].render().unwrap();
        assert!(code.contains("==1)"));
        assert!(code.contains("<256;"));
        assert!(code.contains("stride_") && code.contains(">>=1"));
        assert!(code.contains("else"));
    }
}

#[test]
fn incomplete_regions_are_rejected_before_either_execution_path() {
    let x = "(view (input X) (layout (axis a 128)))";
    let y = "(view (output Y) (layout (axis a 128)))";
    let t = "(view (tensor T) (layout (axis a 128)))";
    let tile = "(keyed_index (slot a (tile i 64)))";
    let full = "(keyed_index (slot a fulltile))";
    for world_size in [1, 2] {
        let config = LoopIrConfig {
            world_size,
            dtypes: ["X", "Y", "T"].map(|s| (s.into(), DType::Fp32)).into(),
            ..Default::default()
        };
        for text in [
            format!("(ploop 0 64 64 i (store {y} (load {x} {tile}) {tile}))"),
            format!(
                "(seq (ploop 0 64 64 i (store {t} (load {x} {tile}) {tile})) (store {y} (load {t} {full}) {full}))"
            ),
            format!("(ploop 0 2 1 i (store {y} (load {x} {full}) {full}))"),
        ] {
            let plan = lower_ir(&text, &config).unwrap().remove(0);
            assert!(matches!(emit(&plan), Err(EmitError::Contract(_))));
        }
    }
}

#[test]
#[ignore = "requires CUDA 13.0+, CUTLASS and NVSHMEM; no GPU required"]
fn singleton_reduction_tail_compiles_and_links() {
    for world in [1, 2] {
        compile(emit(&reduction_tail(world)).unwrap()).unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn reduction_tail_forwards_shared_values_using_cta_local_offsets() {
    for world in [1, 2] {
        let source = emit(&reduction_tail_storage(world, true)).unwrap();
        assert_eq!(source.bodies().len(), 1);
        assert_eq!(source.requirements().buffers.len(), 2);
        assert_eq!(source.requirements().shared_memory_bytes, 1024);
        let code = source.bodies()[0].render().unwrap();
        let reduction_store = code
            .split("if(threadIdx.x==0)")
            .nth(1)
            .unwrap()
            .split(';')
            .next()
            .unwrap();
        assert!(reduction_store.contains("(lv0)-(lv0)"));
    }
}

#[test]
#[ignore = "requires CUDA 13.0+, CUTLASS and NVSHMEM; no GPU required"]
fn shared_reduction_tail_compiles_and_links() {
    for world in [1, 2] {
        compile(emit(&reduction_tail_storage(world, true)).unwrap())
            .unwrap_or_else(|e| panic!("{e}"));
    }
}
