//! Common dtype contracts must be complete before selecting any kernel provider.
use std::collections::BTreeMap;
use trinity_lowering::{
    DType as D, PhysicalPlan, PhysicalPlanBuilder, ScheduledConfig,
    analysis::{Bindings, analyze_text},
    emit::TritonKernelProvider,
    triton::Options,
};

fn view(role: &str, name: &str) -> String {
    format!("(view ({role} {name}) (layout (axis m 16) (axis n 16)))")
}
fn load(role: &str, name: &str) -> String {
    format!("(load {} (keyed_index))", view(role, name))
}
fn store(role: &str, name: &str, value: &str) -> String {
    format!("(store {} {value} (keyed_index))", view(role, name))
}
fn sequence(statements: &[String]) -> String {
    match statements {
        [one] => one.clone(),
        [first, rest @ ..] => format!("(seq {first} {})", sequence(rest)),
        _ => unreachable!(),
    }
}
fn plan(text: &str, default: D, dtypes: BTreeMap<String, D>) -> PhysicalPlan {
    PhysicalPlanBuilder::from_scheduled(
        &analyze_text(text).unwrap(),
        ScheduledConfig {
            target: trinity_lowering::IrConfig::default().target,
            bindings: Bindings::default(),
            default_dtype: default,
            dtypes,
        },
    )
    .unwrap()
}
fn value<'a>(plan: &'a PhysicalPlan, name: &str) -> &'a trinity_lowering::ValueInstance {
    plan.value_instances()
        .find(|(_, v)| v.name() == Some(name))
        .unwrap()
        .1
}

#[test]
fn common_plan_finalizes_exp_reduction_and_dot_types_before_provider_entry() {
    let text = sequence(&[
        store("tensor", "E", &format!("(exp {})", load("input", "X"))),
        store(
            "tensor",
            "S",
            &format!("(bcast (rsum {} 1) 1)", load("tensor", "E")),
        ),
        store(
            "tensor",
            "G",
            &format!("(@ {} {})", load("tensor", "S"), load("input", "V")),
        ),
        store("output", "Y", &load("tensor", "G")),
    ]);
    for dtype in [D::Fp16, D::Bf16] {
        let plan = plan(&text, dtype, BTreeMap::new());
        for name in ["E", "S", "G"] {
            assert_eq!(value(&plan, name).dtype(), dtype);
            assert!(!value(&plan, name).dtype_is_explicit());
        }
        // Provider defaults cannot reinterpret a finalized plan's inferred dtype.
        let program = TritonKernelProvider
            .lower_program(&plan, Options::default())
            .unwrap();
        for (_, common) in plan.value_instances() {
            assert_eq!(
                D::from(
                    program.plan().tensor_dtype(
                        program
                            .plan()
                            .analysis()
                            .tensor_id(common.name().unwrap())
                            .unwrap()
                    )
                ),
                common.dtype()
            );
        }
        let options = Options {
            dtypes: [("E".into(), trinity_lowering::triton::TensorDType::Fp32)].into(),
            ..Default::default()
        };
        let error = TritonKernelProvider
            .lower_program(&plan, options)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("dtype override differs from PhysicalPlan value E")
        );
    }
}

#[test]
fn recurrence_inference_does_not_seed_the_default_into_typed_accumulators() {
    let recurrence = format!(
        "(sloop 0 2 1 k {})",
        store(
            "tensor",
            "Acc",
            &format!("(+ {} {})", load("tensor", "Acc"), load("input", "X"))
        )
    );
    let text = sequence(&[recurrence, store("output", "Y", &load("tensor", "Acc"))]);
    let plan = plan(&text, D::Fp16, [("X".into(), D::Bf16)].into());
    assert_eq!(value(&plan, "Acc").dtype(), D::Bf16);
    assert_eq!(value(&plan, "Y").dtype(), D::Fp16);
}

#[test]
fn explicit_cast_storage_override_and_scalar_defaults_propagate_in_common_analysis() {
    let text = sequence(&[
        store("tensor", "Seed", "1"),
        store(
            "tensor",
            "Mixed",
            &format!("(+ {} {})", load("tensor", "Seed"), load("input", "X")),
        ),
        store(
            "tensor",
            "Narrow",
            &format!("(cast bf16 {})", load("tensor", "Mixed")),
        ),
        store("tensor", "Fixed", &load("tensor", "Narrow")),
        store("tensor", "After", &load("tensor", "Fixed")),
        store("output", "Y", &load("tensor", "After")),
    ]);
    let plan = plan(
        &text,
        D::Fp16,
        [("X".into(), D::Bf16), ("Fixed".into(), D::Fp32)].into(),
    );
    for (name, dtype) in [
        ("Seed", D::Fp16),
        ("Mixed", D::Fp32),
        ("Narrow", D::Bf16),
        ("Fixed", D::Fp32),
        ("After", D::Fp32),
        ("Y", D::Fp16),
    ] {
        assert_eq!(value(&plan, name).dtype(), dtype, "{name}");
    }
    assert!(value(&plan, "Fixed").dtype_is_explicit());
    assert!(!value(&plan, "After").dtype_is_explicit());
}

#[test]
fn native_and_triton_providers_receive_the_same_inferred_value_contract() {
    let text = format!(
        "(ploop 0 16 16 m {})",
        sequence(&[
            store("tensor", "T", &format!("(+ {} 1)", load("input", "X"))),
            store("output", "Y", &format!("(* {} 2)", load("tensor", "T"))),
        ])
    );
    let plan = plan(&text, D::Bf16, BTreeMap::new());
    assert_eq!(value(&plan, "T").dtype(), D::Bf16);
    assert!(!value(&plan, "T").dtype_is_explicit());
    let candidates = trinity_lowering::emit::kernel_candidates(&plan).unwrap();
    assert!(
        candidates
            .values()
            .all(|op| op.candidates.iter().any(|c| c.provider() == "cute"))
    );
    let native = trinity_lowering::emit::emit(&plan).unwrap();
    assert!(
        native
            .requirements()
            .buffers
            .iter()
            .all(|b| b.dtype == D::Bf16)
    );
    assert!(native.code().contains("trinity_launch"));
    let triton = TritonKernelProvider
        .lower_program(&plan, Options::default())
        .unwrap();
    assert!(triton.emit().contains("tl.bfloat16"));
    // This checks shared contracts and source generation, not cross-provider rounding equivalence.
}
