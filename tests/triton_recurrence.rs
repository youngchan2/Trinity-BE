use trinity_lowering::{
    analysis::{Bindings, EntryValue, ProgramFacts, analyze_text},
    triton::{InitialValue, Options, Storage, lower},
};

const ACC: &str = "(load (view (output O) (layout (axis m 16) (axis n 16)))
                        (keyed_index (slot m fulltile) (slot n fulltile)))";
const X: &str = "(load (view (input X) (layout (axis m 16) (axis n 16)))
                      (keyed_index (slot m fulltile) (slot n fulltile)))";
const DOT: &str = "(@
    (load (view (input A) (layout (axis m 16) (axis k 32)))
          (keyed_index (slot m fulltile) (slot k (tile k 16))))
    (load (view (input B) (layout (axis k 32) (axis n 16)))
          (keyed_index (slot k (tile k 16)) (slot n fulltile))))";

fn program(rhs: &str) -> String {
    format!(
        "(sloop 0 32 16 k
           (store (view (output O) (layout (axis m 16) (axis n 16)))
                  {rhs}
                  (keyed_index (slot m fulltile) (slot n fulltile))))"
    )
}

#[test]
fn nested_additive_recurrences_initialize_before_the_reduction_loop() {
    for rhs in [
        format!("(+ {ACC} {DOT})"),
        format!("(+ {DOT} (+ {X} {ACC}))"),
        format!("(+ (+ {ACC} {X}) {DOT})"),
        format!("(+ {X} (+ {DOT} (+ {X} (* 2 {ACC}))))"),
        format!("(+ {DOT} (* (+ {X} {ACC}) 0.5))"),
    ] {
        let ir = analyze_text(&program(&rhs)).unwrap();
        let plan = lower(
            ir,
            Options {
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("{rhs}: {error}"));
        let output = plan.analysis().tensor_id("O").unwrap();
        assert_eq!(plan.kernels().len(), 1);
        let flow = &plan.common().kernels[0].tensors[&output];
        assert_eq!(flow.entry_value, EntryValue::ZeroRecurrence, "{rhs}");
        assert_eq!(flow.additive_updates.len(), 1, "{rhs}");
        let kernel = &plan.kernels()[0];
        let tensor = &kernel.tensors[&output];
        assert_eq!(tensor.storage, Storage::Register);
        let init = tensor.initialization.as_ref().unwrap();
        assert_eq!(init.value, InitialValue::Zero);
        assert_eq!(init.scope, kernel.root_scope, "{rhs}");
        assert_eq!(tensor.accumulators, flow.additive_updates);
        // Emit the complete kernel too: classification must reach codegen.
        assert!(plan.emit().contains("tl.zeros"));
    }
}

#[test]
fn nested_self_loads_under_other_operators_are_not_zero_recurrences() {
    for rhs in [
        format!("(+ {DOT} (+ {X} (exp {ACC})))"),
        format!("(+ {DOT} (+ {X} (* {ACC} {ACC})))"),
        format!("(+ {DOT} (+ {X} (* {X} {ACC})))"),
        format!("(/ {ACC} 2)"),
    ] {
        let ir = analyze_text(&program(&rhs)).unwrap();
        let output = ir.tensor_id("O").unwrap();
        let facts = ProgramFacts::analyze(&ir, &mut Bindings::default()).unwrap();
        let flow = &facts.kernels[0].tensors[&output];
        assert_eq!(flow.entry_value, EntryValue::Unavailable, "{rhs}");
        assert!(flow.additive_updates.is_empty(), "{rhs}");
        let error = lower(ir, Options::default()).unwrap_err().to_string();
        assert!(error.contains("first read is not a defined value or additive accumulator"));
    }
}

#[test]
fn nested_self_load_must_match_the_written_region() {
    let other_region = ACC.replace("(slot n fulltile)", "(slot n (const_tile 0 8))");
    let rhs = format!("(+ {DOT} (+ {X} {other_region}))");
    let ir = analyze_text(&program(&rhs)).unwrap();
    let output = ir.tensor_id("O").unwrap();
    let facts = ProgramFacts::analyze(&ir, &mut Bindings::default()).unwrap();
    let flow = &facts.kernels[0].tensors[&output];
    assert_eq!(flow.entry_value, EntryValue::Unavailable);
    assert!(flow.additive_updates.is_empty());
}
