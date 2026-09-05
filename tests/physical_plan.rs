use trinity_lowering::{
    CommunicationKind, CommunicationOperation, ComputeOperation, DType, OperationPayload,
    PhysicalPlan, PhysicalPlanBuilder, Storage, TargetCapability, all_gather_implementations,
    gemm_implementations,
};

fn gemm_plan(gather_weight: bool) -> PhysicalPlan {
    let target = TargetCapability::Hopper;
    let world_size = if gather_weight { 2 } else { 1 };
    let mut builder = PhysicalPlanBuilder::new(target, world_size);
    let x = builder.add_value(DType::Bf16, [128, 64], Storage::External);
    let weight_shape = if gather_weight { [64, 128] } else { [64, 256] };
    let w = builder.add_value(DType::Bf16, weight_shape, Storage::External);
    let y = builder.add_value(DType::Bf16, [128, 256], Storage::External);
    builder.bind_input("X", x);
    builder.bind_input("W", w);

    let w = if gather_weight {
        let gathered = builder.add_value(DType::Bf16, [64, 256], Storage::Global);
        let instance = all_gather_implementations(target)[0]
            .enumerate(DType::Bf16, [&weight_shape, &[64, 256]], 1, world_size)
            .pop()
            .unwrap();
        let operation = builder.add_operation(
            [w],
            [gathered],
            OperationPayload::Communication(CommunicationOperation::new(
                CommunicationKind::AllGather,
                instance,
            )),
        );
        builder.add_action([operation]);
        gathered
    } else {
        w
    };

    let instance = gemm_implementations(target)[0]
        .enumerate([DType::Bf16; 3], [&[128, 64], &[64, 256], &[128, 256]])
        .pop()
        .unwrap();
    let operation = builder.add_operation(
        [x, w],
        [y],
        OperationPayload::Compute(ComputeOperation::new(instance)),
    );
    builder.add_action([operation]);
    builder.finalize("Y", y).unwrap()
}

#[test]
fn builds_a_single_gpu_gemm_through_the_public_api() {
    let plan = gemm_plan(false);
    assert_eq!(plan.world_size(), 1);
    assert_eq!(plan.operations().len(), 1);
    assert_eq!(plan.actions().len(), 1);
    assert_eq!(plan.inputs()[0].tensor(), "W");
    assert_eq!(plan.inputs()[1].tensor(), "X");
    assert_eq!(plan.output().tensor(), "Y");
    let (_, action) = plan.actions().next().unwrap();
    assert_eq!(action.inputs().len(), 2);
    assert_eq!(action.outputs(), &[plan.output().value()]);
    assert!(plan.same_body(&gemm_plan(false)));
}

#[test]
fn builds_all_gather_and_gemm_through_the_same_public_api() {
    let plan = gemm_plan(true);
    assert_eq!(plan.world_size(), 2);
    assert_eq!(plan.operations().len(), 2);
    let mut actions = plan.actions();
    let (_, gather) = actions.next().unwrap();
    let (_, gemm) = actions.next().unwrap();
    assert!(actions.next().is_none());
    let gathered = gather.outputs()[0];
    assert!(gemm.inputs().contains(&gathered));
    assert_eq!(gemm.outputs(), &[plan.output().value()]);
    let value = plan.value_instance(gathered).unwrap();
    assert_eq!(value.shape(), &[64, 256]);
    assert_eq!(value.storage(), Storage::Global);
    assert!(plan.same_body(&gemm_plan(true)));
}
