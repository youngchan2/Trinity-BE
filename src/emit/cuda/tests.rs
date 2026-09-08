use super::*;
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
    fn specialize(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationEmission, EmitError> {
        let op = plan.operation(id).unwrap();
        let work = (0..plan.world_size())
            .map(|rank| {
                let mut write = full_region(plan, op.outputs()[0], rank);
                if self.bad_region {
                    write.extent[0] += 1;
                }
                vec![Work {
                    reads: vec![full_region(plan, op.inputs()[0], rank)],
                    writes: vec![write],
                    ..Work::default()
                }]
            })
            .collect();
        Ok(OperationEmission {
            body: format!(
                "template<class Runtime> __device__ bool operation_{}(Bindings const&, Tile const&, void*, Runtime const&) {{ /* extension body */ return true; }}",
                id.index()
            ),
            work,
            shared_memory_bytes: 0,
            symmetric_values: vec![],
            nvls: false,
        })
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
    b.add_action([op]);
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
