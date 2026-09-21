//! Broadcast SIMT implementation.
use super::expression::{THREADS, scalar_expression, supported_shape};
use crate::LoopKind;
use crate::emit::cuda::{CudaImplementation, CudaPhaseTemplate, EmitError, OperationSchedule};
use crate::plan::normalize::*;
use crate::{
    AttributeSet, DType, ImplementationDefinition, ImplementationId, ImplementationInstance,
    OperationId, OperationPayload, PhysicalPlan,
};

use crate::BroadcastImplementation;
struct Broadcast;
static INSTANCE: Broadcast = Broadcast;
pub(super) static IMPLEMENTATIONS: &[&dyn BroadcastImplementation] = &[&INSTANCE];

fn attributes() -> AttributeSet {
    AttributeSet::new([("block_threads", THREADS), ("axis", 1)])
}
fn supports(dtypes: [DType; 2], shapes: [&[usize]; 2], axis: usize) -> bool {
    shapes.iter().all(|s| supported_shape(s))
        && axis == 1
        && shapes[1].len() == 2
        && shapes[0] == &shapes[1][..1]
        && dtypes[0] == dtypes[1]
}
impl ImplementationDefinition for Broadcast {
    fn id(&self) -> ImplementationId {
        ImplementationId::new("cuda.broadcast")
    }
    fn cuda(&self) -> Option<&dyn CudaImplementation> {
        Some(self)
    }
}
impl BroadcastImplementation for Broadcast {
    fn enumerate(
        &'static self,
        dtype: DType,
        shapes: [&[usize]; 2],
        axis: usize,
    ) -> Vec<ImplementationInstance> {
        if !supports([dtype; 2], shapes, axis) {
            return vec![];
        }
        vec![ImplementationInstance::new(self, attributes())]
    }
}
impl CudaImplementation for Broadcast {
    fn scalar_expression(&self, operator: &str) -> Option<&'static str> {
        scalar_expression(operator)
    }
    fn schedule(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationSchedule, EmitError> {
        self.phases(plan, id)?;
        let op = plan.operation(id).unwrap();
        let out = op.outputs()[0];
        let shape = plan.value_instance(out).unwrap().shape();
        let parts = vec![tile("row", 1), clipped_tile("col", 128)];
        let input = load(plan, op.inputs()[0], vec![tile("row", 1)]);
        let dimensions = vec![
            dimension("row", shape[0], 1, LoopKind::Parallel),
            dimension("col", shape[1].div_ceil(128) * 128, 128, LoopKind::Parallel),
        ];
        let coordinates = vec![variable("row"), variable("col")];
        Ok(OperationSchedule {
            expression: Some(store(plan, out, expr("bcast", [input, atom(1)]), parts)),
            dimensions,
            coordinates,
        })
    }
    fn phases(&self, plan: &PhysicalPlan, id: OperationId) -> Result<CudaPhaseTemplate, EmitError> {
        let invalid = || unsupported("tensor geometry, dtype or attributes");
        let op = plan.operation(id).unwrap();
        let OperationPayload::Compute(c) = op.payload() else {
            return Err(invalid());
        };
        let ([input], [output]) = (op.inputs(), op.outputs()) else {
            return Err(invalid());
        };
        let a = plan.value_instance(*input).ok_or_else(invalid)?;
        let b = plan.value_instance(*output).ok_or_else(invalid)?;
        if *c.implementation().attributes() != attributes()
            || !supports([a.dtype(), b.dtype()], [a.shape(), b.shape()], 1)
        {
            return Err(invalid());
        }
        Ok(CudaPhaseTemplate::default())
    }
}
