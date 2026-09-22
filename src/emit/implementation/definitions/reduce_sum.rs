//! ReduceSum SIMT implementation.
use super::expression::{THREADS, supported_shape};
use crate::{
    AttributeSet, DType, ImplementationDefinition, ImplementationId, ImplementationInstance,
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
