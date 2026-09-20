use super::*;
use crate::PhysicalPlan;
use crate::Storage;
use crate::{
    AttributeSet, ComputeOperation, ImplementationDefinition, ImplementationId,
    ImplementationInstance, OperationId, OperationPayload, PhysicalPlanBuilder, TargetCapability,
};
use crate::{CudaTargetCapability, DType};

struct Definition {
    bad_region: u8,
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
        use crate::plan::normalize::*;
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
        _plan: &PhysicalPlan,
        _id: OperationId,
        _rank: usize,
        _coordinate: [usize; 3],
    ) -> Result<Option<Work>, EmitError> {
        panic!("Streamed must not request physical task work");
    }
    fn accesses(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
        rank: usize,
        _coordinate: [usize; 3],
    ) -> Result<Option<Accesses>, EmitError> {
        use super::regions::full_region;
        let op = plan.operation(id).unwrap();
        let mut write = full_region(plan, op.outputs()[0], rank);
        match self.bad_region {
            1 => write.extent[0] += 1,
            2 => write.value = crate::ValueInstanceId::from_index(usize::MAX),
            3 => write.rank = plan.world_size(),
            _ => {}
        }
        Ok(Some(Accesses {
            reads: vec![full_region(plan, op.inputs()[0], rank)],
            writes: vec![write],
            ..Default::default()
        }))
    }
}

fn plan(definition: &'static dyn ImplementationDefinition) -> PhysicalPlan {
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
    static ENABLED: Definition = Definition { bad_region: 0 };
    static BAD: Definition = Definition { bad_region: 1 };
    assert!(
        emit(&plan(&ENABLED))
            .unwrap()
            .code()
            .contains("extension body")
    );
    static BAD_VALUE: Definition = Definition { bad_region: 2 };
    static BAD_RANK: Definition = Definition { bad_region: 3 };
    for invalid in [&BAD, &BAD_VALUE, &BAD_RANK] {
        assert!(matches!(emit(&plan(invalid)), Err(EmitError::Contract(_))));
        assert!(matches!(
            crate::emit::validate(&plan(invalid)),
            Err(EmitError::Contract(_))
        ));
    }
}

#[test]
fn prepared_program_keeps_final_buffers_and_renders_without_rebuilding() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingDefinition(AtomicUsize);
    impl ImplementationDefinition for CountingDefinition {
        fn id(&self) -> ImplementationId {
            ImplementationId::new("test.prepared_program")
        }
        fn cuda(&self) -> Option<&dyn CudaImplementation> {
            Some(self)
        }
    }
    impl CudaImplementation for CountingDefinition {
        fn schedule(
            &self,
            plan: &PhysicalPlan,
            id: OperationId,
        ) -> Result<OperationSchedule, EmitError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Definition { bad_region: 0 }.schedule(plan, id)
        }
        fn phases(&self, plan: &PhysicalPlan, id: OperationId) -> Result<Body, EmitError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            let mut body = Definition { bad_region: 0 }.phases(plan, id)?;
            body.prologue.resources.symmetric_values =
                plan.operation(id).unwrap().inputs().to_vec();
            Ok(body)
        }
        fn accesses(
            &self,
            plan: &PhysicalPlan,
            id: OperationId,
            rank: usize,
            coordinate: [usize; 3],
        ) -> Result<Option<Accesses>, EmitError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Definition { bad_region: 0 }.accesses(plan, id, rank, coordinate)
        }
    }
    static DEFINITION: CountingDefinition = CountingDefinition(AtomicUsize::new(0));
    let plan = plan(&DEFINITION);
    let before_prepare = DEFINITION.0.load(Ordering::Relaxed);
    let prepared = crate::emit::prepare(&plan).unwrap();
    let after_prepare = DEFINITION.0.load(Ordering::Relaxed);
    assert!(after_prepare > before_prepare);
    drop(plan);
    let crate::emit::EmittedSource::Cuda(source) = prepared.render().unwrap();
    assert_eq!(DEFINITION.0.load(Ordering::Relaxed), after_prepare);
    let buffers = &source.requirements().buffers;
    assert!(buffers.iter().all(|b| b.alignment == 16));
    assert!(buffers[0].symmetric);
    assert!(!buffers[1].symmetric);
    assert!(source.code().contains("extension body"));
}
#[test]
fn identity_plan_shares_its_input_output_allocation_and_has_no_work() {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 2);
    let input = b.add_value(DType::Bf16, [128, 128], Storage::External);
    b.bind_input("A", input);
    let source = emit(&b.finalize("Y", input).unwrap()).unwrap();
    assert_eq!(source.requirements().buffers.len(), 1);
    assert_eq!(source.requirements().buffers[0].input_names, ["A"]);
    assert_eq!(
        source.execution().as_persistent().unwrap().tasks_per_rank,
        0
    );
    assert_eq!(
        source
            .execution()
            .as_persistent()
            .unwrap()
            .output_dependencies[0],
        [Dependency { rank: 0, slot: 0 }]
    );
}

#[test]
fn body_reuse_does_not_reorder_a_dependency_or_duplicate_kernel_names() {
    use super::persistent::access::{Effects, Event};
    use super::regions::full_region;
    static DEFINITION: Definition = Definition { bad_region: 0 };
    let plan = plan(&DEFINITION);
    let input = full_region(&plan, plan.inputs()[0].value(), 0);
    let output = full_region(&plan, plan.output().value(), 0);
    let mut tasks = Vec::new();
    let mut effects = Vec::new();
    for slot in 0..3 {
        let read = if slot == 0 { input } else { output };
        tasks.push(Task {
            statement: slot,
            path: vec![slot],
            body: slot % 2,
            domain: Vec::new(),
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
    let schedule = (0..3)
        .map(|slot| TaskScheduling {
            rank: 0,
            slot,
            dependencies: Vec::new(),
            stages: Vec::new(),
            ordered_collective: false,
        })
        .collect();
    let execution = persistent::graph::resolve(&plan, tasks.clone(), schedule, effects).unwrap();
    assert_eq!(
        execution.tasks.iter().map(|l| l.body).collect::<Vec<_>>(),
        [0, 1, 0]
    );
    assert_eq!(
        execution.schedule[2].dependencies,
        vec![Dependency { rank: 0, slot: 1 }]
    );
    let original = emit(&plan).unwrap();
    let code = render::program(
        original.requirements(),
        &Execution::Streamed(
            StreamedExecution::new(
                tasks,
                vec![
                    Grid {
                        axes: vec![],
                        blocks: 1
                    };
                    3
                ],
            )
            .unwrap(),
        ),
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
        fn stage_accesses(
            &self,
            plan: &PhysicalPlan,
            id: OperationId,
            domain: &crate::LoopDomain,
            expression: &crate::Expression,
        ) -> Result<Option<StageAccessPattern>, EmitError> {
            self.backend().stage_accesses(plan, id, domain, expression)
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
