//! Common storage inference shared by source imports and native emission.
use std::collections::BTreeMap;
use trinity_lowering::{
    analysis::{Bindings, ProgramFacts, analyze_text, storage},
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
