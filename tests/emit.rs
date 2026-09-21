//! Retirement boundary: no execution path may fall back to the archived emitter.

use trinity_lowering::*;

fn identity() -> PhysicalPlan {
    let mut builder = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
    let input = builder.add_value(DType::Fp32, [4], Storage::External);
    builder.bind_input("X", input);
    builder.build(vec![], "Y", input).unwrap()
}

#[test]
fn identity_emits_an_abi_without_kernel_launches() {
    let source = emit(&identity()).unwrap();
    assert!(!source.code().contains("<<<"));
    assert!(source.code().contains("trinity_launch"));
    assert_eq!(source.requirements().buffers[0].alignment, 4);
}

struct UncheckedRule;

impl FusionRule for UncheckedRule {
    fn apply(
        &self,
        _plan: &PhysicalPlan,
        _producer: &Statement,
        _consumer: &Statement,
    ) -> Result<Vec<FusionRewrite>, FusionError> {
        panic!("rules must not run without the replacement fusion validator")
    }
}

#[test]
fn fusion_does_not_return_unvalidated_candidates() {
    let plan = identity();
    assert!(fuse(&plan, &[]).unwrap()[0].same_body(&plan));
    assert!(matches!(
        fuse(&plan, &[&UncheckedRule]),
        Err(FusionError::Unavailable)
    ));
    assert!(fusion_rules(plan.target()).is_empty());
}
