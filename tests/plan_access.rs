use std::collections::BTreeMap;
use trinity_lowering::*;

const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Hopper);

fn tiled_copy(width: TileWidth) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TARGET, 1);
    let x = b.add_named_value("X", DType::Fp32, [128], Storage::External);
    let y = b.add_named_value("Y", DType::Fp32, [128], Storage::External);
    b.bind_input("X", x);
    let access = |value| {
        TensorAccess::new(
            value,
            [
                AccessIndex::Tile {
                    variable: "row".into(),
                    width: width.clone(),
                },
                AccessIndex::FullTile,
            ],
        )
        .with_view_shape([4, 32])
    };
    let op = b.add_operation(
        [x],
        [y],
        Expression::Store {
            destination: access(y),
            value: Box::new(Expression::Load(access(x))),
        },
    );
    let step = match width {
        TileWidth::Constant(n) => IndexExpr::Constant(n as i64),
        TileWidth::Symbol(s) => IndexExpr::Variable(s),
    };
    b.build(
        vec![Statement::Loop(Loop {
            kind: LoopKind::Parallel,
            domain: LoopDomain {
                variable: "row".into(),
                start: IndexExpr::Constant(0),
                stop: IndexExpr::Constant(4),
                step,
            },
            body: vec![Statement::Operation(op)],
        })],
        "Y",
        y,
    )
    .unwrap()
}

#[test]
fn symbols_bind_tile_and_step_together_without_mutating_the_source_plan() {
    let symbolic = tiled_copy("BLOCK".into());
    assert_eq!(symbolic.symbols(), ["BLOCK".into()].into());
    assert!(
        emit(&symbolic)
            .unwrap_err()
            .to_string()
            .contains("bind_symbols")
    );
    let mut hashes = vec![];
    for width in [1usize, 2, 4] {
        let bound = symbolic
            .bind_symbols(&[("BLOCK".into(), width as i64)].into())
            .unwrap();
        let literal = tiled_copy(width.into());
        assert!(bound.symbols().is_empty());
        assert!(bound.same_body(&literal));
        assert_eq!(bound.hash(), literal.hash());
        let source = emit(&bound).unwrap();
        assert_eq!(source.code(), emit(&literal).unwrap().code());
        // Runtime buffers retain the allocation shape; CuTe sees the access view.
        assert!(
            source
                .requirements()
                .buffers
                .iter()
                .all(|b| b.shape == [128])
        );
        assert!(
            source
                .code()
                .contains("cute::make_stride(int64_t(32), int64_t(1))")
        );
        hashes.push(bound.hash());
    }
    assert_ne!(hashes[0], hashes[1]);
    assert_ne!(hashes[1], hashes[2]);
    assert_eq!(symbolic.symbols(), ["BLOCK".into()].into());
    for bindings in [
        BTreeMap::new(),
        [("BLOCK".into(), 0)].into(),
        [("BLOCK".into(), -1)].into(),
    ] {
        assert!(symbolic.bind_symbols(&bindings).is_err());
    }
}

#[test]
fn canonical_loop_names_do_not_capture_configuration_symbols() {
    let symbolic = tiled_copy("lv0".into());
    let [Statement::Loop(l)] = symbolic.statements() else {
        panic!("loop")
    };
    assert_ne!(l.domain.variable, "lv0");
    let bound = symbolic.bind_symbols(&[("lv0".into(), 2)].into()).unwrap();
    assert!(bound.same_body(&tiled_copy(2usize.into())));
}

fn config() -> IrConfig {
    IrConfig {
        dtypes: ["X", "T", "Y"].map(|s| (s.into(), DType::Fp32)).into(),
        ..Default::default()
    }
}

#[test]
fn text_reader_preserves_different_views_of_one_intermediate() {
    let text = "(seq
      (store (view (tensor T) (layout (axis row 2) (axis col 64)))
        (load (view (input X) (layout (axis row 2) (axis col 64))) (keyed_index)) (keyed_index))
      (store (view (output Y) (layout (axis row 4) (axis col 32)))
        (load (view (tensor T) (layout (axis row 4) (axis col 32))) (keyed_index)) (keyed_index)))";
    let p = lower_ir(text, &config()).unwrap().remove(0);
    let ops: Vec<_> = p.operations().map(|(_, op)| op).collect();
    assert_eq!(ops[0].outflows(), ops[1].inflows());
    let Expression::Store { value, .. } = ops[1].expression() else {
        panic!("store")
    };
    let Expression::Load(access) = value.as_ref() else {
        panic!("load")
    };
    assert_eq!(p.value_instance(access.value).unwrap().shape(), [2, 64]);
    assert_eq!(access.view_shape.as_deref(), Some([4, 32].as_slice()));
    // Coverage written as [2,64] must be readable as [4,32].
    let source = emit(&p).unwrap();
    assert!(
        source
            .code()
            .contains("cute::make_stride(int64_t(64), int64_t(1))")
    );
    assert!(
        source
            .code()
            .contains("cute::make_stride(int64_t(32), int64_t(1))")
    );
    let python = emit::emit_python(&p).unwrap();
    assert_eq!(
        python.manifest()["operations"][1]["expression"]["view_shape"],
        serde_json::json!([4, 32])
    );
    assert!(python.sources().keys().any(|name| name.contains("triton")));
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tests/plan_access");
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(path.join("views.py"), python.emit()).unwrap();
    assert!(lower_ir(&text.replace("axis row 4", "axis row 3"), &config()).is_err());
}

#[test]
fn element_indices_use_loop_ordinal_in_coverage_checks() {
    let text = "(ploop 0 4 2 i (store
      (view (output Y) (layout (axis row 2) (axis col 4))) 1
      (keyed_index (slot row (elem i)))))";
    let p = lower_ir(text, &config()).unwrap().remove(0);
    emit(&p).unwrap();
}

#[test]
fn binding_parameters_preserves_outer_loop_coordinates() {
    let text = "(ploop 0 128 BLOCK i (sloop i (+ i BLOCK) 1 j (store
      (view (output Y) (layout (axis a 128))) 1
      (keyed_index (slot a (tile j 1))))))";
    let symbolic = lower_ir(text, &config()).unwrap().remove(0);
    let bound = symbolic
        .bind_symbols(&[("BLOCK".into(), 32), ("lv0".into(), 999)].into())
        .unwrap();
    let [Statement::Loop(outer)] = bound.statements() else {
        panic!("outer loop")
    };
    let [Statement::Loop(inner)] = outer.body.as_slice() else {
        panic!("inner loop")
    };
    assert_eq!(
        inner.domain.start,
        IndexExpr::Variable(outer.domain.variable.clone())
    );
    assert_eq!(
        inner.domain.stop,
        IndexExpr::Add(
            Box::new(IndexExpr::Variable(outer.domain.variable.clone())),
            Box::new(IndexExpr::Constant(32))
        )
    );
}

#[test]
fn symbolic_ir_and_eager_bindings_produce_the_same_copy() {
    let text = "(ploop 0 128 BLOCK i (store
        (view (output Y) (layout (axis a 128)))
        (load (view (input X) (layout (axis a 128))) (keyed_index (slot a (tile i BLOCK))))
        (keyed_index (slot a (tile i BLOCK)))))";
    let symbolic = lower_ir(text, &config()).unwrap().remove(0);
    let mut concrete = config();
    concrete.symbols.insert("BLOCK".into(), 32);
    let bound = symbolic.bind_symbols(&concrete.symbols).unwrap();
    let eager = lower_ir(text, &concrete).unwrap().remove(0);
    assert!(bound.same_body(&eager));
    assert_eq!(emit(&bound).unwrap().code(), emit(&eager).unwrap().code());
}

#[test]
fn view_changes_do_not_hide_cross_cta_write_conflicts() {
    let mut b = PhysicalPlanBuilder::new(TARGET, 1);
    let y = b.add_value(DType::Fp32, [2, 4], Storage::External);
    let mut body = vec![];
    for shape in [[2, 4], [4, 2]] {
        let access = TensorAccess::new(y, [AccessIndex::Elem("i".into()), AccessIndex::FullTile])
            .with_view_shape(shape);
        let op = b.add_operation(
            [],
            [y],
            Expression::Store {
                destination: access,
                value: Box::new(Expression::Constant(Constant::Integer(1))),
            },
        );
        body.push(Statement::Operation(op));
    }
    let p = b
        .build(
            vec![Statement::Loop(Loop {
                kind: LoopKind::Parallel,
                domain: LoopDomain {
                    variable: "i".into(),
                    start: IndexExpr::Constant(0),
                    stop: IndexExpr::Constant(2),
                    step: IndexExpr::Constant(1),
                },
                body,
            })],
            "Y",
            y,
        )
        .unwrap();
    let error = emit(&p).unwrap_err().to_string();
    assert!(error.contains("overlap between CTAs"), "{error}");
}

#[test]
fn builder_checks_view_capacity_and_view_rank() {
    for (shape, indices) in [
        (vec![4, 31], vec![AccessIndex::FullTile; 2]),
        (vec![4, 32], vec![AccessIndex::FullTile]),
        (vec![0, 32], vec![AccessIndex::FullTile; 2]),
        (vec![usize::MAX, 2], vec![AccessIndex::FullTile; 2]),
    ] {
        let mut b = PhysicalPlanBuilder::new(TARGET, 1);
        let y = b.add_value(DType::Fp32, [128], Storage::External);
        let op = b.add_operation(
            [],
            [y],
            Expression::Store {
                destination: TensorAccess::new(y, indices).with_view_shape(shape),
                value: Box::new(Expression::Constant(Constant::Integer(1))),
            },
        );
        assert!(b.build(vec![Statement::Operation(op)], "Y", y).is_err());
    }
}

#[test]
fn gemm_uses_matrix_views_over_flat_allocations() {
    let mut b = PhysicalPlanBuilder::new(TARGET, 1);
    let a = b.add_value(DType::Bf16, [16 * 64], Storage::External);
    let w = b.add_value(DType::Bf16, [64 * 128], Storage::External);
    let y = b.add_value(DType::Fp32, [16 * 128], Storage::External);
    b.bind_input("A", a);
    b.bind_input("W", w);
    let tile = AccessIndex::Tile {
        variable: "k".into(),
        width: 64usize.into(),
    };
    let output = TensorAccess::new(y, [AccessIndex::FullTile, AccessIndex::FullTile])
        .with_view_shape([16, 128]);
    let mm = Expression::Matmul(Box::new([
        Expression::Load(
            TensorAccess::new(a, [AccessIndex::FullTile, tile.clone()]).with_view_shape([16, 64]),
        ),
        Expression::Load(
            TensorAccess::new(w, [tile, AccessIndex::FullTile]).with_view_shape([64, 128]),
        ),
    ]));
    let op = b.add_operation(
        [a, w],
        [y],
        Expression::Store {
            destination: output.clone(),
            value: Box::new(Expression::Add(Box::new([Expression::Load(output), mm]))),
        },
    );
    let p = b
        .build(
            vec![Statement::Loop(Loop {
                kind: LoopKind::Sequential,
                domain: LoopDomain {
                    variable: "k".into(),
                    start: IndexExpr::Constant(0),
                    stop: IndexExpr::Constant(64),
                    step: IndexExpr::Constant(64),
                },
                body: vec![Statement::Operation(op)],
            })],
            "Y",
            y,
        )
        .unwrap();
    let source = emit(&p).unwrap();
    assert!(source.code().contains("warpgroup"));
    assert!(
        source
            .requirements()
            .buffers
            .iter()
            .all(|b| b.shape.len() == 1)
    );
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tests/plan_access");
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(path.join("gemm.cu"), source.code()).unwrap();
    std::fs::write(
        path.join("copy.cu"),
        emit(&tiled_copy(2usize.into())).unwrap().code(),
    )
    .unwrap();
}
