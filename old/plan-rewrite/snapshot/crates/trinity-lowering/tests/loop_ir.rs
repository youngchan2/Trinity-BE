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
    let [Statement::Loop(l)] = p.statements() else {
        panic!("expected sequential loop")
    };
    assert_eq!(l.kind, LoopKind::Sequential);
    assert_eq!(l.domain.start, IndexExpr::Constant(0));
    assert_eq!(l.domain.stop, IndexExpr::Constant(128));
    assert_eq!(l.domain.step, IndexExpr::Constant(64));
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
