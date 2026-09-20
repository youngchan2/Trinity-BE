//! Shared scalar expressions and SIMT launch constraints.
use crate::emit::cuda::{CudaImplementation, CudaPhaseTemplate, EmitError};
use crate::{
    AttributeSet, ImplementationDefinition, ImplementationId, ImplementationInstance, OperationId,
    PhysicalPlan,
};
pub(super) const THREADS: usize = 128;
pub(crate) fn scalar_expression(operator: &str) -> Option<&'static str> {
    Some(match operator {
        "+" => "x + y",
        "-" => "x - y",
        "*" => "x * y",
        "/" => "x / y",
        "sqr" => "x * x",
        "sqrt" => "sqrtf(x)",
        "sigmoid" => "1.0f / (1.0f + expf(-x))",
        "relu" => "x < 0.0f ? 0.0f : x",
        _ => return None,
    })
}

struct ExpressionImplementation;
static EXPRESSION: ExpressionImplementation = ExpressionImplementation;
impl ImplementationDefinition for ExpressionImplementation {
    fn id(&self) -> ImplementationId {
        ImplementationId::new("cuda.simt.expression")
    }
    fn cuda(&self) -> Option<&dyn CudaImplementation> {
        Some(self)
    }
}
impl CudaImplementation for ExpressionImplementation {
    fn scalar_expression(&self, operator: &str) -> Option<&'static str> {
        scalar_expression(operator)
    }
    fn phases(
        &self,
        _plan: &PhysicalPlan,
        _id: OperationId,
    ) -> Result<CudaPhaseTemplate, EmitError> {
        Ok(CudaPhaseTemplate::default())
    }
}
pub(crate) fn expression_instance() -> ImplementationInstance {
    ImplementationInstance::new(&EXPRESSION, AttributeSet::new([("block_threads", THREADS)]))
}

pub(super) fn supported_shape(shape: &[usize]) -> bool {
    matches!(shape.len(), 1 | 2) && !shape.contains(&0)
}
