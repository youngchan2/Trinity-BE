//! Broadcast SIMT implementation.
use super::expression::{THREADS, supported_shape};
use crate::{
    AttributeSet, DType, ImplementationDefinition, ImplementationId, ImplementationInstance,
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
