use trinity_lowering::*;

fn config(world_size: usize, split: i64) -> LoopIrConfig {
    let meta: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/loop_ir/IR.v3.meta.json")).unwrap();
    let mut c = LoopIrConfig {
        world_size,
        ..Default::default()
    };
    for name in meta["tensor_shapes"].as_object().unwrap().keys() {
        c.dtypes.insert(
            name.clone(),
            if name == "attn_O3" {
                DType::Fp32
            } else {
                DType::Bf16
            },
        );
    }
    c.dtypes
        .insert("__split_fdba4cffaeb51ebe".into(), DType::Fp32);
    for (name, value) in meta["example_bindings"].as_object().unwrap() {
        c.symbols.insert(name.clone(), value.as_i64().unwrap());
    }
    c.symbols.insert("__nsplit_fdba4cffaeb51ebe".into(), split);
    c
}
const FFN: &str = include_str!("fixtures/loop_ir/IR.v3.txt");
const SPLIT: &str = include_str!("fixtures/loop_ir/IR.v3.split_k.txt");

#[test]
fn ffn_bodies_preserve_two_independent_gemms_and_phases() {
    let p = lower_ir(FFN, &config(1, 4)).unwrap().remove(0);
    assert_eq!(p.statements().len(), 8);
    let s = emit(&p).unwrap();
    assert_eq!(s.execution().as_streamed().unwrap().launches.len(), 8);
    assert_eq!(s.code().matches("clear(accumulator_").count(), 4);
    assert_eq!(s.code().matches("warpgroup_wait<0>").count(), 4);
    assert!(s.code().contains("<16?16:0"));
    assert!(!s.code().contains('$'));
    let paired = s
        .execution()
        .as_streamed()
        .unwrap()
        .tasks
        .iter()
        .filter(|t| p.statements()[t.statement].operations().len() == 2)
        .collect::<Vec<_>>();
    assert!(!paired.is_empty());
    assert!(paired.iter().all(|t| s.bodies()[t.body].mainloop.is_some()));
    let persistent = emit(&lower_ir(FFN, &config(2, 4)).unwrap().remove(0)).unwrap();
    let e = persistent.execution().as_persistent().unwrap();
    assert!(
        e.tasks
            .iter()
            .zip(&e.schedule)
            .filter(|(t, _)| p.statements()[t.statement].operations().len() == 2)
            .all(|(_, m)| m.stages.len() == 128)
    );
    assert!(fuse(&p, fusion_rules(p.target())).unwrap()[0].same_body(&p));
}
#[test]
fn split_reduce_waits_for_its_own_partials_on_each_rank() {
    for split in [1, 2, 4] {
        for world in [1, 2] {
            let p = lower_ir(SPLIT, &config(world, split))
                .unwrap()
                .remove(0);
            let s = emit(&p).unwrap();
            match s.execution() {
                trinity_lowering::emit::Execution::Streamed(e) => {
                    assert_eq!(e.tasks.len(), 9);
                    assert_eq!(e.launches.len(), 9);
                    assert_eq!(e.grids[0].blocks, 32 * split as u64);
                    assert_eq!(e.grids[1].blocks, 32);
                    assert_eq!(e.launches[0].task, 0);
                    assert_eq!(e.launches[1].task, 1);
                }
                trinity_lowering::emit::Execution::Persistent(e) => {
                    for (task, metadata) in
                        e.tasks.iter().zip(&e.schedule).filter(|(t, _)| t.body == 1)
                    {
                        assert_eq!(metadata.dependencies.len(), split as usize);
                        for dep in &metadata.dependencies {
                            let partial = &e.tasks[metadata.rank * e.tasks_per_rank + dep.slot];
                            assert_eq!(partial.body, 0);
                            assert_eq!(
                                partial.bindings().unwrap().values().next(),
                                task.bindings().unwrap().values().next()
                            );
                        }
                    }
                }
            }
            let scratch = s
                .requirements()
                .buffers
                .iter()
                .find(|b| b.shape.len() == 3)
                .unwrap();
            assert_eq!(scratch.strides, vec![16 * 4096, 4096, 1]);
            assert_eq!(scratch.dtype, DType::Fp32);
        }
    }
}
#[test]
fn alpha_renaming_is_canonical_and_errors_are_located() {
    let a = lower_ir(FFN, &config(1, 4)).unwrap().remove(0);
    let renamed = FFN
        .replace(" k\n", " kk\n")
        .replace("(tile k ", "(tile kk ");
    let b = lower_ir(&renamed, &config(1, 4)).unwrap().remove(0);
    assert!(a.same_body(&b));
    assert_eq!(a.hash(), b.hash());
    let a = lower_ir(SPLIT, &config(1, 4)).unwrap().remove(0);
    let renamed = SPLIT.replace("__msplit_fdba4cffaeb51ebe", "another_split_index");
    let b = lower_ir(&renamed, &config(1, 4)).unwrap().remove(0);
    assert!(a.same_body(&b));
    let mut c = config(1, 4);
    c.symbols.remove("tile_k");
    let e = lower_ir(FFN, &c).err().unwrap();
    assert!(e.offset > 0);
    assert!(e.message.contains("tile_k"));
    let e = lower_ir(&FFN.replace("sigmoid", "unknown_op"), &config(1, 4))
        .err()
        .unwrap();
    assert!(e.offset > 0);
    assert!(e.message.contains("unknown_op"));
    assert!(
        lower_ir(SPLIT, &config(1, 3))
            .err()
            .unwrap()
            .message
            .contains("exactly divide")
    );
}

fn copy_program(start: usize, stop: usize, kind: &str) -> String {
    format!(
        "({kind} {start} {stop} 64 i (store (view (output Y) (layout (axis a 128))) (load (view (input X) (layout (axis a 128))) (keyed_index (slot a (tile i 64)))) (keyed_index (slot a (tile i 64)))))"
    )
}
fn copy_config() -> LoopIrConfig {
    LoopIrConfig {
        dtypes: [("X".into(), DType::Fp32), ("Y".into(), DType::Fp32)].into(),
        ..Default::default()
    }
}
#[test]
fn ordinary_serial_loops_keep_iterations_and_program_order_is_hashed() {
    let p = lower_ir(&copy_program(0, 128, "sloop"), &copy_config())
        .unwrap()
        .remove(0);
    let s = emit(&p).unwrap();
    assert_eq!(s.execution().as_streamed().unwrap().tasks.len(), 1);
    assert!(s.code().contains("for(std::int64_t lv0=0;lv0<128;lv0+=64)"));
    assert!(!s.code().contains("float acc=0.0f"));
    let a = copy_program(0, 64, "ploop");
    let b = copy_program(64, 128, "ploop");
    let ab = lower_ir(&format!("(seq {a} {b})"), &copy_config())
        .unwrap()
        .remove(0);
    let ba = lower_ir(&format!("(seq {b} {a})"), &copy_config())
        .unwrap()
        .remove(0);
    assert!(!ab.same_body(&ba));
    assert_ne!(ab.hash(), ba.hash());
}
#[test]
fn rejects_invalid_tile_bounds_and_parallel_work_in_serial_scope() {
    let error = lower_ir(&copy_program(64, 192, "ploop"), &copy_config())
        .err()
        .unwrap();
    assert!(error.offset > 0 && error.message.contains("out-of-bounds"));
    let text = format!("(sloop 0 2 1 outer {})", copy_program(0, 128, "ploop"));
    assert!(
        lower_ir(&text, &copy_config())
            .err()
            .unwrap()
            .message
            .contains("parallel work inside")
    );
}

#[test]
fn accumulation_preserves_an_explicit_prior_initial_value() {
    let view_y = "(view (output Y) (layout (axis a 128)))";
    let view_x = "(view (input X) (layout (axis a 128)))";
    let index = "(keyed_index (slot a fulltile))";
    let text = format!(
        "(seq (store {view_y} (load {view_x} {index}) {index}) (sloop 0 2 1 k (store {view_y} (+ (load {view_y} {index}) (load {view_x} {index})) {index})))"
    );
    let plan = lower_ir(&text, &copy_config()).unwrap().remove(0);
    let source = emit(&plan).unwrap();
    assert!(source.code().contains("=float(static_cast<float*>"));
    assert!(!source.code().contains("=0.0f; for(std::int64_t lv0"));
    assert_eq!(
        source
            .execution()
            .as_streamed()
            .unwrap()
            .launches
            .iter()
            .map(|l| l.task)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    let mut config = copy_config();
    config.world_size = 2;
    let persistent = emit(&lower_ir(&text, &config).unwrap().remove(0)).unwrap();
    assert_eq!(
        persistent.execution().as_persistent().unwrap().schedule[1].dependencies[0].slot,
        0
    );
}
#[test]
#[ignore = "requires NVCC and NVSHMEM, never executes GPU work"]
fn ffn_sources_compile_and_link() {
    for world in [1, 2] {
        for text in [FFN, SPLIT] {
            let plan = lower_ir(text, &config(world, 4)).unwrap().remove(0);
            let source = emit(&plan).unwrap();
            std::fs::write(
                format!("/tmp/trinity-loop-{world}-{}.cu", text.len()),
                source.code(),
            )
            .unwrap();
            let artifact = compile(source).unwrap_or_else(|e| panic!("{e}"));
            assert!(artifact.artifact_path().is_file());
        }
    }
}
