//! Identity and geometry for already scheduled expressions; no code generation.

use crate::{AttributeSet, ImplementationDefinition, ImplementationId, ImplementationInstance};

pub(super) const THREADS: usize = 128;

struct ExpressionImplementation;
static EXPRESSION: ExpressionImplementation = ExpressionImplementation;

impl ImplementationDefinition for ExpressionImplementation {
    fn id(&self) -> ImplementationId {
        ImplementationId::new("cuda.simt.expression")
    }
}

pub(super) fn expression_instance() -> ImplementationInstance {
    ImplementationInstance::new(&EXPRESSION, AttributeSet::new([("block_threads", THREADS)]))
}

pub(super) fn supported_shape(shape: &[usize]) -> bool {
    matches!(shape.len(), 1 | 2) && !shape.contains(&0)
}
