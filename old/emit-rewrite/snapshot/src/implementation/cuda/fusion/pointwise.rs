use super::super::gemm::hopper_wgmma::fusion_shape;
use crate::{FusionError, FusionRewrite, FusionRule, PhysicalPlan, Statement, Storage};
pub(super) struct PointwiseFusion;
impl FusionRule for PointwiseFusion {
    fn apply(
        &self,
        plan: &PhysicalPlan,
        producer: &Statement,
        consumer: &Statement,
    ) -> Result<Vec<FusionRewrite>, FusionError> {
        let left = producer.operations();
        let right = consumer.operations();
        let [consumer_id] = right.as_slice() else {
            return Ok(Vec::new());
        };
        fn parallel_only(statement: &Statement) -> bool {
            match statement {
                Statement::Operation(_) => true,
                Statement::Loop(l) => {
                    l.kind == crate::LoopKind::Parallel
                        && l.body.len() == 1
                        && parallel_only(&l.body[0])
                }
            }
        }
        if !parallel_only(consumer) || !crate::fusion::pointwise(plan, *consumer_id) {
            return Ok(Vec::new());
        }
        let Some(&last) = left.last() else {
            return Ok(Vec::new());
        };
        if left.iter().any(|id| {
            fusion_shape(plan, plan.operation(*id).unwrap()).is_none()
                && !crate::fusion::pointwise(plan, *id)
        }) {
            return Ok(Vec::new());
        }
        let a = plan.operation(last).unwrap();
        let b = plan.operation(*consumer_id).unwrap();
        let [bridge] = a.outputs() else {
            return Ok(Vec::new());
        };
        let value = plan.value_instance(*bridge).unwrap();
        if value.storage() != Storage::Global
            || !b.inputs().contains(bridge)
            || value.shape() != plan.value_instance(b.outputs()[0]).unwrap().shape()
            || plan
                .operations()
                .any(|(id, op)| id != *consumer_id && op.inputs().contains(bridge))
            || plan
                .operations()
                .filter(|(_, op)| op.outputs().contains(bridge))
                .count()
                != 1
        {
            return Ok(Vec::new());
        }
        Ok(vec![FusionRewrite::new(
            left.into_iter().chain(right),
            [(*bridge, Storage::Register)],
        )])
    }
}
