//! Shared-plan/provider boundary tests. No source AST is needed for re-emission.
use std::{fs, path::PathBuf, process::Command};
use trinity_lowering::{
    AccessIndex as A, Constant, CudaTargetCapability, DType, Expression as E, IndexExpr as I, Loop,
    LoopDomain, LoopKind, PhysicalPlanBuilder, Statement, Storage, TargetCapability, TensorAccess,
    ValueOp,
    analysis::{Bindings, ProgramFacts, TensorMetadata, analyze_text},
    emit::{TritonKernelProvider, emit_python, emit_triton},
    triton::{AutotuneOptions, Options},
};

const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Sm120);

fn options() -> Options {
    Options {
        autotune: AutotuneOptions {
            max_configs: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn syntax(name: &str, source: &str) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tests/triton_provider");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.py"));
    fs::write(&path, source).unwrap();
    let result = Command::new("python3")
        .args(["-c", "import ast, pathlib, sys; p=pathlib.Path(sys.argv[1]); compile(ast.parse(p.read_text()), str(p), 'exec')"])
        .arg(path).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn builder_views_and_symbolic_loops_reach_the_program_provider() {
    let mut b = PhysicalPlanBuilder::new(TARGET, 1);
    b.add_value(DType::Fp16, [8], Storage::Global); // unused scratch remains named
    let x = b.add_named_value("X", DType::Fp16, [2, 4], Storage::External);
    let y = b.add_named_value("Y", DType::Fp16, [8], Storage::External);
    b.bind_input("features", x);
    let axis = A::Tile {
        variable: "i".into(),
        width: "BLOCK".into(),
    };
    let op = b.add_operation(
        [x],
        [y],
        E::Store {
            destination: TensorAccess::new(y, [axis.clone()]),
            value: Box::new(E::Load(TensorAccess::new(x, [axis]).with_view_shape([8]))),
        },
    );
    let physical = b
        .build(
            vec![Statement::Loop(Loop {
                kind: LoopKind::Parallel,
                domain: LoopDomain {
                    variable: "i".into(),
                    start: I::Constant(0),
                    stop: I::Constant(8),
                    step: I::Variable("BLOCK".into()),
                },
                body: vec![Statement::Operation(op)],
            })],
            "result",
            y,
        )
        .unwrap();
    let mut opts = options();
    opts.symbols.insert("BLOCK".into(), 4);
    let program = TritonKernelProvider
        .lower_program(&physical, opts.clone())
        .unwrap();
    assert_eq!(program.physical_plan().target(), TARGET);
    assert!(program.plan().analysis().ir().is_none());
    assert_eq!(program.plan().kernels().len(), 1);
    assert!(program.emit().contains("META_BLOCK"));
    assert_eq!(program.emit(), emit_triton(&physical, opts).unwrap());
    syntax("builder_view", &program.emit());
    let executable = emit_python(&physical).unwrap();
    assert_eq!(executable.manifest()["mode"], "triton_program");
    assert_eq!(executable.manifest()["inputs"][0]["name"], "features");
    assert_eq!(executable.manifest()["inputs"][0]["argument"], "X");
    syntax("builder_executable", &executable.emit());
}

#[test]
fn multiple_outputs_and_cache_mutation_keep_one_region_and_one_input_argument() {
    let text = "(ploop 0 8 4 i (seq
        (store (view (output Cache) (layout (axis m 8)))
            (+ (load (view (input Cache) (layout (axis m 8)))
                     (keyed_index (slot m (tile i 4)))) 1)
            (keyed_index (slot m (tile i 4))))
        (store (view (output Y) (layout (axis m 8)))
            (* (load (view (input Cache) (layout (axis m 8)))
                     (keyed_index (slot m (tile i 4)))) 2)
            (keyed_index (slot m (tile i 4))))))";
    let program = TritonKernelProvider
        .lower_source(analyze_text(text).unwrap(), options())
        .unwrap();
    let physical = program.physical_plan();
    assert_eq!(physical.outputs().len(), 2);
    assert_eq!(physical.mutable_inputs().len(), 1);
    assert!(matches!(&physical.statements()[0], Statement::Region(_)));
    assert_eq!(program.plan().kernels().len(), 1);
    let source = program.emit();
    assert!(source.contains("def forward(Cache, Y=None):"));
    assert!(source.contains("restore_value=['Cache_ptr']"));
    syntax("multi_output", &source);
    syntax(
        "multi_output_executable",
        &emit_python(physical).unwrap().emit(),
    );
}

#[test]
fn imported_accumulator_records_only_its_first_initialization() {
    let text = "(sloop 0 32 16 k (seq
        (store (view (output S) (layout (axis m 16)))
            (+ (load (view (output S) (layout (axis m 16))) (keyed_index)) 1)
            (keyed_index))
        (store (view (output S) (layout (axis m 16)))
            (+ (load (view (output S) (layout (axis m 16))) (keyed_index)) 2)
            (keyed_index))))";
    let program = TritonKernelProvider
        .lower_source(analyze_text(text).unwrap(), options())
        .unwrap();
    let physical = program.physical_plan();
    let s = physical.output().value();
    let ops: Vec<_> = physical.operations().map(|(_, op)| op).collect();
    assert_eq!(ops[0].zero_init(), &[s]);
    assert!(ops[0].inflows().is_empty());
    assert!(ops[1].zero_init().is_empty());
    assert_eq!(ops[1].inflows(), &[s]);
    syntax("accumulator", &program.emit());
}

#[test]
fn an_existing_producer_is_not_replaced_by_implicit_accumulator_zero() {
    let text = "(ploop 0 16 16 m (seq
        (store (view (output S) (layout (axis m 16)))
            (load (view (input X) (layout (axis m 16))) (keyed_index)) (keyed_index))
        (sloop 0 32 16 k
            (store (view (output S) (layout (axis m 16)))
                (+ (load (view (output S) (layout (axis m 16))) (keyed_index)) 1)
                (keyed_index)))))";
    let program = TritonKernelProvider
        .lower_source(analyze_text(text).unwrap(), options())
        .unwrap();
    let p = program.physical_plan();
    let ops: Vec<_> = p.operations().map(|(_, op)| op).collect();
    assert!(ops.iter().all(|op| op.zero_init().is_empty()));
    assert_eq!(ops[1].inflows(), &[p.output().value()]);
    syntax("producer_then_accumulate", &program.emit());
}

#[test]
fn symbolic_split_scratch_and_access_views_specialize_together() {
    let text = "(seq (mloop 0 128 tile_k k s num_splits
          (store (view (tensor P) (layout (axis s num_splits) (axis m 16)))
            (+ (load (view (tensor P) (layout (axis s num_splits) (axis m 16)))
                     (keyed_index (slot s (elem s)) (slot m fulltile)))
               (unsqueeze (rsum (load (view (input X) (layout (axis k 128) (axis m 16)))
                                      (keyed_index (slot k (tile k tile_k)) (slot m fulltile))) 0) 0))
            (keyed_index (slot s (elem s)) (slot m fulltile))))
      (ploop 0 16 16 m (store (view (output Y) (layout (axis m 16)))
        (rsum (load (view (tensor P) (layout (axis s num_splits) (axis m 16))) (keyed_index)) 0)
        (keyed_index))))";
    let program = TritonKernelProvider
        .lower_source(analyze_text(text).unwrap(), options())
        .unwrap();
    let p = program
        .physical_plan()
        .bind_symbols(&[("num_splits".into(), 2), ("tile_k".into(), 16)].into())
        .unwrap();
    let (id, value) = p
        .value_instances()
        .find(|(_, v)| v.name() == Some("P"))
        .unwrap();
    assert_eq!(value.shape(), &[2, 16]);
    assert!(p.symbols().is_empty());
    for (_, op) in p.operations() {
        for a in op.expression().accesses().iter().filter(|a| a.value == id) {
            assert_eq!(a.shape(value.shape()), &[2, 16]);
        }
    }
    let rebound = TritonKernelProvider.lower_program(&p, options()).unwrap();
    assert_eq!(rebound.plan().kernels().len(), 2);
    syntax("split_bound", &rebound.emit());
}

#[test]
fn extended_expression_uses_fallback_without_an_unimplemented_reference() {
    let mut b = PhysicalPlanBuilder::new(TARGET, 1);
    let y = b.add_named_value("Y", DType::Fp32, [16], Storage::External);
    let op = b.add_operation(
        [],
        [y],
        E::Store {
            destination: TensorAccess::new(y, [A::FullTile]),
            value: Box::new(E::Apply {
                op: ValueOp::Exp,
                args: vec![E::Constant(Constant::Integer(1))].into(),
            }),
        },
    );
    let p = b.build(vec![Statement::Operation(op)], "Y", y).unwrap();
    let program = emit_python(&p).unwrap();
    assert_eq!(program.manifest()["mode"], "triton_program");
    syntax("extended_expression", &program.emit());
}

#[test]
fn provider_does_not_override_explicit_shape_or_dtype() {
    let text = "(store (view (output Y) (layout (axis m 16)))
        (load (view (input X) (layout (axis m 16))) (keyed_index)) (keyed_index))";
    let program = TritonKernelProvider
        .lower_source(analyze_text(text).unwrap(), options())
        .unwrap();
    let p = program.physical_plan();
    let mut shape = options();
    shape.shapes.insert("X".into(), vec![32]);
    assert!(
        TritonKernelProvider
            .lower_program(p, shape)
            .err()
            .unwrap()
            .to_string()
            .contains("shape override")
    );
    let mut dtype = options();
    dtype
        .dtypes
        .insert("X".into(), trinity_lowering::triton::TensorDType::Bf16);
    assert!(
        TritonKernelProvider
            .lower_program(p, dtype)
            .err()
            .unwrap()
            .to_string()
            .contains("dtype override")
    );
}

#[test]
fn mla_access_regions_and_loop_bounds_survive_the_common_plan() {
    for stage in [14, 16, 20] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/batched_mla");
        let text =
            fs::read_to_string(root.join(format!("batched_mla_postprocessed_stage{stage}.txt")))
                .unwrap();
        let original = analyze_text(&text).unwrap();
        let mut opts = options();
        for line in include_str!("fixtures/batched_mla/shapes.txt").lines() {
            let mut words = line.split_whitespace();
            opts.shapes.insert(
                words.next().unwrap().into(),
                words.map(|w| w.parse().unwrap()).collect(),
            );
        }
        let program = TritonKernelProvider
            .lower_source(original.clone(), opts.clone())
            .unwrap();
        let projected = program.plan().analysis();
        assert!(projected.ir().is_none());
        assert_eq!(original.kernels().len(), projected.kernels().len());
        assert_eq!(original.statements().len(), projected.statements().len());
        assert_eq!(original.accesses().len(), projected.accesses().len());
        for (old, new) in original.accesses().iter().zip(projected.accesses()) {
            assert_eq!(old.kind, new.kind);
            assert_eq!(
                original.tensor(old.tensor).name,
                projected.tensor(new.tensor).name
            );
            assert_eq!(old.scope, new.scope);
            assert_eq!(old.statement, new.statement);
        }
        for (old, new) in original.scopes().iter().zip(projected.scopes()) {
            assert_eq!(old.kind, new.kind);
            assert_eq!(old.children, new.children);
            if let (Some(a), Some(b)) = (&old.loop_info, &new.loop_info) {
                assert_eq!((&a.start, &a.end, &a.step), (&b.start, &b.end, &b.step));
            }
        }
        let mut bindings = Bindings {
            shapes: opts.shapes.clone(),
            symbols: program.physical_plan().bindings().clone(),
        };
        let metadata = TensorMetadata::collect(&original, &mut bindings).unwrap();
        let facts = ProgramFacts::resolve(&original, &mut bindings, metadata).unwrap();
        assert_eq!(
            facts.accesses,
            program.plan().common().accesses,
            "stage {stage}: address ranges changed"
        );
        syntax(&format!("mla_stage{stage}"), &program.emit());
        let rebuilt = TritonKernelProvider
            .lower_program(program.physical_plan(), opts)
            .unwrap();
        assert_eq!(
            program.emit(),
            rebuilt.emit(),
            "re-emission must not need source syntax"
        );
    }
}

#[test]
fn output_order_does_not_follow_canonical_input_ids() {
    let text = "(ploop 0 8 4 i (seq
        (store (view (output Y) (layout (axis m 8)))
            2
            (keyed_index (slot m (tile i 4))))
        (store (view (output Cache) (layout (axis m 8)))
            (+ (load (view (input Cache) (layout (axis m 8))) (keyed_index (slot m (tile i 4)))) 1)
            (keyed_index (slot m (tile i 4))))))";
    let program = TritonKernelProvider
        .lower_source(analyze_text(text).unwrap(), options())
        .unwrap();
    assert_eq!(program.physical_plan().outputs()[0].tensor(), "Y");
    assert!(program.emit().contains("return (Y, Cache)"));
    syntax("output_order", &program.emit());
}
