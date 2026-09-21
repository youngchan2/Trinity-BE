//! ReduceSum SIMT implementation.
use super::expression::{THREADS, scalar_expression, supported_shape};
use crate::LoopKind;
use crate::emit::cuda::{CudaImplementation, CudaPhaseTemplate, EmitError, OperationSchedule};
use crate::plan::normalize::*;
use crate::{
    AttributeSet, DType, ImplementationDefinition, ImplementationId, ImplementationInstance,
    OperationId, OperationPayload, PhysicalPlan,
};

use crate::ReduceSumImplementation;
struct ReduceSum;
static INSTANCE: ReduceSum = ReduceSum;
pub(super) static IMPLEMENTATIONS: &[&dyn ReduceSumImplementation] = &[&INSTANCE];

fn attributes() -> AttributeSet {
    AttributeSet::new([("block_threads", THREADS), ("axis", 1)])
}
fn supports(dtypes: [DType; 2], shapes: [&[usize]; 2], axis: usize) -> bool {
    shapes.iter().all(|s| supported_shape(s))
        && axis == 1
        && shapes[0].len() == 2
        && shapes[1] == &shapes[0][..1]
        && dtypes[1] == DType::Fp32
}
impl ImplementationDefinition for ReduceSum {
    fn id(&self) -> ImplementationId {
        ImplementationId::new("cuda.reduce_sum")
    }
    fn cuda(&self) -> Option<&dyn CudaImplementation> {
        Some(self)
    }
}
impl ReduceSumImplementation for ReduceSum {
    fn enumerate(
        &'static self,
        dtypes: [DType; 2],
        shapes: [&[usize]; 2],
        axis: usize,
    ) -> Vec<ImplementationInstance> {
        if !supports(dtypes, shapes, axis) {
            return vec![];
        }
        vec![ImplementationInstance::new(self, attributes())]
    }
}
impl CudaImplementation for ReduceSum {
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
        let parts = vec![tile("row", 1)];
        let input = load(plan, op.inputs()[0], vec![tile("row", 1), atom("fulltile")]);
        let dimensions = vec![dimension("row", shape[0], 1, LoopKind::Parallel)];
        let coordinates = vec![variable("row")];
        Ok(OperationSchedule {
            expression: Some(store(plan, out, expr("rsum", [input, atom(1)]), parts)),
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
