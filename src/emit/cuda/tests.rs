use super::*;
use crate::DType;
use crate::{
    AttributeSet, ComputeOperation, ImplementationDefinition, ImplementationId,
    ImplementationInstance, OperationId, OperationPayload, PhysicalPlanBuilder, TargetCapability,
};

struct Definition {
    bad_region: bool,
}
impl ImplementationDefinition for Definition {
    fn id(&self) -> ImplementationId {
        ImplementationId::new("test.cuda_extension")
    }
    fn cuda(&self) -> Option<&dyn CudaImplementation> {
        Some(self)
    }
}
impl CudaImplementation for Definition {
    fn schedule(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationSchedule, EmitError> {
        use crate::physical::normalize::*;
        let op = plan.operation(id).unwrap();
        let index = vec![atom("fulltile"), atom("fulltile")];
        Ok(OperationSchedule {
            expression: Some(store(
                plan,
                op.outputs()[0],
                load(plan, op.inputs()[0], index.clone()),
                index,
            )),
            ..Default::default()
        })
    }
    fn phases(
        &self,
        _plan: &PhysicalPlan,
        _id: OperationId,
    ) -> Result<CudaPhaseTemplate, EmitError> {
        Body::template("/* extension body */", None, "", &[], &[])
    }
    fn work(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
        rank: usize,
        _coordinate: [usize; 3],
    ) -> Result<Option<Work>, EmitError> {
        use super::graph::full_region;
        let op = plan.operation(id).unwrap();
        let mut write = full_region(plan, op.outputs()[0], rank);
        if self.bad_region {
            write.extent[0] += 1;
        }
        Ok(Some(Work {
            reads: vec![full_region(plan, op.inputs()[0], rank)],
            writes: vec![write],
            ..Default::default()
        }))
    }
}

fn plan(definition: &'static Definition) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
    let input = b.add_value(DType::Bf16, [128, 128], Storage::External);
    b.bind_input("X", input);
    let output = b.add_value(DType::Bf16, [128, 128], Storage::External);
    let instance = ImplementationInstance::new(definition, AttributeSet::default());
    let op = b.add_operation(
        [input],
        [output],
        OperationPayload::Compute(ComputeOperation::new(instance)),
    );
    b.add_statement(crate::Statement::Operation(op));
    b.finalize("Y", output).unwrap()
}
#[test]
fn dispatches_through_the_definition_interface() {
    static ENABLED: Definition = Definition { bad_region: false };
    static BAD: Definition = Definition { bad_region: true };
    assert!(
        emit(&plan(&ENABLED))
            .unwrap()
            .code()
            .contains("extension body")
    );
    assert!(matches!(emit(&plan(&BAD)), Err(EmitError::Contract(_))));
}
#[test]
fn identity_plan_shares_its_input_output_allocation_and_has_no_work() {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 2);
    let input = b.add_value(DType::Bf16, [128, 128], Storage::External);
    b.bind_input("A", input);
    let source = emit(&b.finalize("Y", input).unwrap()).unwrap();
    assert_eq!(source.requirements().buffers.len(), 1);
    assert_eq!(source.requirements().buffers[0].input_names, ["A"]);
    assert_eq!(source.execution().tasks_per_rank, 0);
    assert_eq!(
        source.execution().output_dependencies[0],
        [Dependency { rank: 0, slot: 0 }]
    );
}

#[test]
fn body_reuse_does_not_reorder_a_dependency_or_duplicate_kernel_names() {
    use super::access::{Effects, Event};
    use super::graph::full_region;
    static DEFINITION: Definition = Definition { bad_region: false };
    let plan = plan(&DEFINITION);
    let input = full_region(&plan, plan.inputs()[0].value(), 0);
    let output = full_region(&plan, plan.output().value(), 0);
    let mut tasks = Vec::new();
    let mut effects = Vec::new();
    for slot in 0..3 {
        let read = if slot == 0 { input } else { output };
        tasks.push(Task {
            rank: 0,
            slot,
            statement: slot,
            task_set: slot,
            body: slot % 2,
            arguments: Default::default(),
            argument: 0,
            coordinate: [0; 3],
            shared_memory_bytes: 0,
            dependencies: Vec::new(),
            stages: Vec::new(),
            ordered_collective: false,
        });
        effects.push(Effects {
            work: Work {
                reads: vec![read],
                writes: vec![output],
                ..Default::default()
            },
            events: vec![
                Event {
                    region: read,
                    write: false,
                    stage: None,
                },
                Event {
                    region: output,
                    write: true,
                    stage: None,
                },
            ],
        });
    }
    let execution = graph::resolve(&plan, tasks, effects, vec![vec![0], vec![1], vec![2]]).unwrap();
    assert_eq!(
        execution
            .launches
            .iter()
            .map(|l| l.body)
            .collect::<Vec<_>>(),
        [0, 1, 0]
    );
    assert_eq!(
        execution.tasks[2].dependencies,
        vec![Dependency { rank: 0, slot: 1 }]
    );
    let original = emit(&plan).unwrap();
    let code = render::program(
        original.requirements(),
        &execution,
        &["body zero".into(), "body one".into()],
    )
    .unwrap();
    for id in 0..3 {
        assert_eq!(
            code.matches(&format!("__global__ void kernel_{id}("))
                .count(),
            1
        );
    }
}

#[test]
fn accumulation_dispatch_and_shared_reservation_follow_the_selected_backend() {
    // Give the built-in GEMM a larger scratch contract. The common emitter must
    // dispatch its accumulation hook and place a fused tile after that scratch.
    struct LargerScratch;
    impl LargerScratch {
        fn backend(&self) -> &'static dyn CudaImplementation {
            crate::gemm_implementations(TargetCapability::Cuda(CudaTargetCapability::Hopper))[0]
                .cuda()
                .unwrap()
        }
    }
    impl ImplementationDefinition for LargerScratch {
        fn id(&self) -> ImplementationId {
            crate::gemm_implementations(TargetCapability::Cuda(CudaTargetCapability::Hopper))[0]
                .id()
        }
        fn cuda(&self) -> Option<&dyn CudaImplementation> {
            Some(self)
        }
    }
    impl CudaImplementation for LargerScratch {
        fn schedule(
            &self,
            plan: &PhysicalPlan,
            id: OperationId,
        ) -> Result<OperationSchedule, EmitError> {
            self.backend().schedule(plan, id)
        }
        fn phases(&self, plan: &PhysicalPlan, id: OperationId) -> Result<Body, EmitError> {
            let mut body = self.backend().phases(plan, id)?;
            body.prologue.resources.shared_memory_bytes = 81920;
            Ok(body)
        }
        fn accumulation(
            &self,
            plan: &PhysicalPlan,
            id: OperationId,
            scope: &AccumulationScope,
        ) -> Result<AccumulationBody, EmitError> {
            assert_eq!(scope.inputs[0].width, [128, 64]);
            assert_eq!(scope.output.width, [128, 128]);
            let mut result = self.backend().accumulation(plan, id, scope)?;
            result.body.prologue.resources.shared_memory_bytes = 81920;
            result
                .body
                .prologue
                .code
                .append(&Code::text("/* selected accumulation hook */\n"));
            Ok(result)
        }
    }
    static DEFINITION: LargerScratch = LargerScratch;
    let target = TargetCapability::Cuda(CudaTargetCapability::Hopper);
    for world in [1, 2] {
        let mut builder = PhysicalPlanBuilder::new(target, world);
        let a = builder.add_value(DType::Bf16, [128, 128], Storage::External);
        let b = builder.add_value(DType::Bf16, [128, 128], Storage::External);
        let c = builder.add_value(DType::Bf16, [128, 128], Storage::External);
        builder.bind_input("A", a);
        builder.bind_input("B", b);
        builder.bind_input("C", c);
        let intermediate = builder.add_value(DType::Bf16, [128, 128], Storage::Global);
        let output = builder.add_value(DType::Bf16, [128, 128], Storage::External);
        for (inputs, out) in [([a, b], intermediate), ([intermediate, c], output)] {
            let instance = ImplementationInstance::new(
                &DEFINITION,
                AttributeSet::new([("tile_m", 128), ("tile_n", 128), ("tile_k", 64)]),
            );
            let op = builder.add_operation(
                inputs,
                [out],
                OperationPayload::Compute(ComputeOperation::new(instance)),
            );
            builder.add_statement(crate::Statement::Operation(op));
        }
        let plan = builder.finalize("Y", output).unwrap();
        let candidates = crate::fuse(&plan, crate::fusion_rules(target)).unwrap();
        assert!(candidates.len() > 1);
        let source = emit(candidates.last().unwrap()).unwrap();
        assert_eq!(
            source.code().matches("selected accumulation hook").count(),
            2
        );
        assert!(source.code().contains("memory)+81920"));
        assert_eq!(
            source.requirements().shared_memory_bytes,
            81920 + 128 * 128 * 2
        );
    }
}
