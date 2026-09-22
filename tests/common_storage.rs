//! Storage inference is shared by source import, Triton and native CuTe emission.
use std::collections::BTreeMap;
use trinity_lowering::{
    AccessIndex, DType, Expression as E, PhysicalPlan, PhysicalPlanBuilder, ScheduledConfig,
    Statement, Storage, TensorAccess,
    analysis::{Bindings, ProgramFacts, analyze_text, storage},
    emit::TritonKernelProvider,
    triton::{AutotuneOptions, Options},
    *,
};

const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Hopper);

fn access(role: &str, name: &str, size: usize, index: &str) -> String {
    format!("(load (view ({role} {name}) (layout (axis m {size}))) (keyed_index (slot m {index})))")
}
fn store(role: &str, name: &str, size: usize, index: &str, rhs: &str) -> String {
    format!(
        "(store (view ({role} {name}) (layout (axis m {size}))) {rhs} (keyed_index (slot m {index})))"
    )
}
fn local_ir() -> String {
    let x = access("input", "X", 16, "fulltile");
    let t = access("tensor", "T", 16, "fulltile");
    format!(
        "(ploop 0 1 1 p (seq {} {}))",
        store("tensor", "T", 16, "fulltile", &format!("(sqr {x})")),
        store("output", "Y", 16, "fulltile", &format!("(sigmoid {t})"))
    )
}
fn materialized_ir() -> String {
    format!(
        "(ploop 0 1 1 p (seq (sloop 0 16 4 i {}) {}))",
        store(
            "tensor",
            "T",
            16,
            "(tile i 4)",
            &access("input", "X", 16, "(tile i 4)")
        ),
        store(
            "output",
            "Y",
            16,
            "fulltile",
            &access("tensor", "T", 16, "fulltile")
        )
    )
}
fn physical(text: &str) -> PhysicalPlan {
    PhysicalPlanBuilder::from_scheduled(
        &analyze_text(text).unwrap(),
        ScheduledConfig {
            target: TARGET,
            bindings: Bindings::default(),
            default_dtype: DType::Fp32,
            dtypes: [("T".into(), DType::Fp32)].into(),
        },
    )
    .unwrap()
}
fn options() -> Options {
    Options {
        autotune: AutotuneOptions {
            max_configs: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn classes(p: &PhysicalPlan) -> BTreeMap<String, Storage> {
    p.value_instances()
        .map(|(_, v)| (v.name().unwrap().into(), v.storage()))
        .collect()
}

fn text_config(text: &str) -> IrConfig {
    IrConfig {
        target: TARGET,
        dtypes: analyze_text(text)
            .unwrap()
            .tensors()
            .iter()
            .map(|t| (t.name.clone(), DType::Fp32))
            .collect(),
        ..Default::default()
    }
}

#[test]
fn both_source_entries_share_register_and_materialization_decisions() {
    let cross = format!(
        "(seq (ploop 0 1 1 p {}) (ploop 0 1 1 q {}))",
        store(
            "tensor",
            "T",
            16,
            "fulltile",
            &access("input", "X", 16, "fulltile")
        ),
        store(
            "output",
            "Y",
            16,
            "fulltile",
            &access("tensor", "T", 16, "fulltile")
        )
    );
    for (text, expected) in [
        (local_ir(), Storage::Register),
        (materialized_ir(), Storage::Global),
        (cross, Storage::Global),
    ] {
        let p = lower_ir(&text, &text_config(&text)).unwrap().remove(0);
        assert_eq!(classes(&p), classes(&physical(&text)));
        assert_eq!(classes(&p)["T"], expected);
        let program = TritonKernelProvider.lower_program(&p, options()).unwrap();
        assert_eq!(
            program.emit().contains("T_ptr"),
            expected == Storage::Global
        );
        if expected == Storage::Register {
            let native = emit(&p).unwrap();
            assert_eq!(native.requirements().buffers.len(), 2);
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target/tests/common_storage");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("text_register.cu"), native.code()).unwrap();
        }
    }
}

#[test]
fn text_storage_inference_preserves_unbound_tiles_and_matches_eager_binding() {
    let first = store(
        "tensor",
        "T",
        16,
        "(tile i BLOCK)",
        &access("input", "X", 16, "(tile i BLOCK)"),
    );
    let second = store(
        "output",
        "Y",
        16,
        "(tile i BLOCK)",
        &access("tensor", "T", 16, "(tile i BLOCK)"),
    );
    for (text, expected) in [
        (
            format!("(ploop 0 16 BLOCK i (seq {first} {second}))"),
            Storage::Register,
        ),
        (
            format!("(seq (ploop 0 16 BLOCK i {first}) (ploop 0 16 BLOCK i {second}))"),
            Storage::Global,
        ),
        (
            materialized_ir()
                .replace("16 4 i", "16 BLOCK i")
                .replace("tile i 4", "tile i BLOCK"),
            Storage::Global,
        ),
    ] {
        let mut config = text_config(&text);
        let symbolic = lower_ir(&text, &config).unwrap().remove(0);
        assert_eq!(classes(&symbolic)["T"], expected);
        assert_eq!(symbolic.symbols(), ["BLOCK".into()].into());
        for width in [1, 4, 8] {
            config.symbols.insert("BLOCK".into(), width);
            let bound = symbolic.bind_symbols(&config.symbols).unwrap();
            let eager = lower_ir(&text, &config).unwrap().remove(0);
            assert!(bound.same_body(&eager));
            assert_eq!(classes(&bound)["T"], expected);
            TritonKernelProvider
                .lower_program(&bound, options())
                .unwrap();
        }
    }
}

#[test]
fn unproven_symbolic_coverage_is_rejected_instead_of_guessing_storage() {
    let text = format!(
        "(seq (ploop 0 16 STRIDE i {}) (ploop 0 16 BLOCK j {}))",
        store(
            "tensor",
            "T",
            16,
            "(tile i BLOCK)",
            &access("input", "X", 16, "(tile i BLOCK)")
        ),
        store(
            "output",
            "Y",
            16,
            "(tile j BLOCK)",
            &access("tensor", "T", 16, "(tile j BLOCK)")
        )
    );
    let error = lower_ir(&text, &text_config(&text)).err().unwrap();
    assert!(error.message.contains("not proven disjoint") || error.message.contains("not covered"));
}

#[test]
fn same_region_intermediate_is_register_and_both_providers_use_it() {
    let p = physical(&local_ir());
    assert_eq!(
        classes(&p),
        [
            ("X".into(), Storage::External),
            ("T".into(), Storage::Register),
            ("Y".into(), Storage::External)
        ]
        .into()
    );
    let source = emit(&p).unwrap();
    // CuTe must not allocate a scratch buffer for the inferred register value.
    assert_eq!(source.requirements().buffers.len(), 2);
    let program = TritonKernelProvider.lower_program(&p, options()).unwrap();
    let t = program.plan().analysis().tensor_id("T").unwrap();
    assert!(!program.plan().kernels()[0].tensors[&t].publish);
    assert!(!program.emit().contains("T_ptr"));
    assert!(!program.emit().contains("T = torch.empty"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tests/common_storage");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("inferred_register.cu"), source.code()).unwrap();
}

#[test]
fn intermediate_crossing_regions_is_global() {
    let text = format!(
        "(seq (ploop 0 1 1 p {}) (ploop 0 1 1 q {}))",
        store(
            "tensor",
            "T",
            16,
            "fulltile",
            &access("input", "X", 16, "fulltile")
        ),
        store(
            "output",
            "Y",
            16,
            "fulltile",
            &access("tensor", "T", 16, "fulltile")
        )
    );
    let p = physical(&text);
    assert_eq!(classes(&p)["T"], Storage::Global);
    let native = emit(&p).unwrap();
    assert_eq!(native.requirements().buffers.len(), 3);
    let program = TritonKernelProvider.lower_program(&p, options()).unwrap();
    assert_eq!(program.plan().kernels().len(), 2);
    assert!(program.emit().contains("T = torch.empty"));
    assert!(program.emit().contains("tl.store(T_ptr"));
    assert!(program.emit().contains("tl.load(T_ptr"));
}

#[test]
fn different_tile_writes_followed_by_a_full_read_require_materialization() {
    let p = physical(&materialized_ir());
    assert_eq!(classes(&p)["T"], Storage::Global);
    let program = TritonKernelProvider.lower_program(&p, options()).unwrap();
    let t = program.plan().analysis().tensor_id("T").unwrap();
    assert_eq!(
        program.plan().kernels()[0].tensors[&t].storage,
        storage::AccessMode::Materialized
    );
    assert!(program.emit().contains("tl.store(T_ptr"));
    assert!(program.emit().contains("tl.load(T_ptr"));
}

#[test]
fn a_contained_subtile_read_keeps_the_producer_in_registers() {
    let text = format!(
        "(ploop 0 1 1 p (seq {} (sloop 0 16 4 j {})))",
        store(
            "tensor",
            "T",
            16,
            "fulltile",
            &access("input", "X", 16, "fulltile")
        ),
        store(
            "output",
            "Y",
            16,
            "(tile j 4)",
            &access("tensor", "T", 16, "(tile j 4)")
        )
    );
    let p = physical(&text);
    assert_eq!(classes(&p)["T"], Storage::Register);
    let program = TritonKernelProvider.lower_program(&p, options()).unwrap();
    assert!(!program.plan().kernels()[0].local_reads.is_empty());
    assert!(!program.emit().contains("T_ptr"));
    assert!(program.emit().contains("tl.gather"));
}

#[test]
fn loop_carried_accumulator_stays_local_until_the_output_store() {
    let update = format!(
        "(+ {} (rsum {} 0))",
        access("tensor", "S", 1, "fulltile"),
        access("input", "X", 16, "(tile i 4)")
    );
    let text = format!(
        "(ploop 0 1 1 p (seq (sloop 0 16 4 i {}) {}))",
        store("tensor", "S", 1, "fulltile", &update),
        store(
            "output",
            "Y",
            1,
            "fulltile",
            &access("tensor", "S", 1, "fulltile")
        )
    );
    let p = physical(&text);
    assert_eq!(classes(&p)["S"], Storage::Register);
    let program = TritonKernelProvider.lower_program(&p, options()).unwrap();
    let s = program.plan().analysis().tensor_id("S").unwrap();
    assert_eq!(
        program.plan().kernels()[0].tensors[&s]
            .initialization
            .as_ref()
            .unwrap()
            .value,
        storage::InitialValue::Zero
    );
    assert!(!program.emit().contains("S_ptr"));
}

fn explicit_pipeline(storage: Storage) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TARGET, 1);
    let x = b.add_named_value("X", DType::Fp32, [16], Storage::External);
    let t = b.add_named_value("T", DType::Fp32, [16], storage);
    let y = b.add_named_value("Y", DType::Fp32, [16], Storage::External);
    b.bind_input("X", x);
    let access = |id| TensorAccess::new(id, [AccessIndex::FullTile]);
    let first = b.add_operation(
        [x],
        [t],
        E::Store {
            destination: access(t),
            value: Box::new(E::Sqr(Box::new(E::Load(access(x))))),
        },
    );
    let second = b.add_operation(
        [t],
        [y],
        E::Store {
            destination: access(y),
            value: Box::new(E::Sigmoid(Box::new(E::Load(access(t))))),
        },
    );
    b.build(
        vec![Statement::Region(vec![
            Statement::Operation(first),
            Statement::Operation(second),
        ])],
        "Y",
        y,
    )
    .unwrap()
}

#[test]
fn explicit_global_storage_is_not_silently_eliminated_by_triton() {
    for storage in [Storage::Global, Storage::Register] {
        let p = explicit_pipeline(storage);
        let program = TritonKernelProvider.lower_program(&p, options()).unwrap();
        let source = program.emit();
        assert_eq!(
            source.contains("T = torch.empty"),
            storage == Storage::Global
        );
        assert_eq!(
            source.contains("tl.store(T_ptr"),
            storage == Storage::Global
        );
        assert_eq!(
            emit(&p).unwrap().requirements().buffers.len(),
            if storage == Storage::Global { 3 } else { 2 }
        );
    }
}

#[test]
fn shared_analysis_rejects_a_register_contract_that_needs_global_memory() {
    let ir = analyze_text(&materialized_ir()).unwrap();
    let mut bindings = Bindings::default();
    let facts = ProgramFacts::analyze(&ir, &mut bindings).unwrap();
    let inferred = storage::infer(&ir, &bindings, &facts.kernels).unwrap();
    let t = ir.tensor_id("T").unwrap();
    let mut contracts = inferred.values;
    contracts.insert(t, Storage::Register);
    let error = storage::for_values(&ir, &bindings, &facts.kernels, &contracts).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Register storage cannot satisfy")
    );
}

#[test]
fn mla_storage_classifications_survive_specialization_and_provider_lowering() {
    for stage in [14, 16, 20] {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/batched_mla");
        let text = std::fs::read_to_string(
            root.join(format!("batched_mla_postprocessed_stage{stage}.txt")),
        )
        .unwrap();
        let mut opts = options();
        for line in include_str!("fixtures/batched_mla/shapes.txt").lines() {
            let mut words = line.split_whitespace();
            opts.shapes.insert(
                words.next().unwrap().into(),
                words.map(|w| w.parse().unwrap()).collect(),
            );
        }
        let program = TritonKernelProvider
            .lower_source(analyze_text(&text).unwrap(), opts)
            .unwrap();
        let p = program.physical_plan();
        assert!(
            p.value_instances()
                .any(|(_, v)| v.storage() == Storage::Register)
        );
        assert!(
            p.value_instances()
                .any(|(_, v)| v.storage() == Storage::Global)
        );
        for (_, v) in p.value_instances() {
            if v.storage() == Storage::Register {
                let id = program
                    .plan()
                    .analysis()
                    .tensor_id(v.name().unwrap())
                    .unwrap();
                let uses: Vec<_> = program
                    .plan()
                    .kernels()
                    .iter()
                    .filter_map(|k| k.tensors.get(&id))
                    .collect();
                assert_eq!(uses.len(), 1);
                assert!(!uses[0].publish);
                assert_eq!(uses[0].storage, storage::AccessMode::Register);
            }
        }
        let bound = p.bind_symbols(&BTreeMap::new()).unwrap();
        assert_eq!(classes(p), classes(&bound));
        TritonKernelProvider
            .lower_program(&bound, options())
            .unwrap();
    }
}
