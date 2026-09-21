//! Common bindings, platform dispatch, and the prepared program contract.

use crate::{PhysicalPlan, TargetCapability};

use super::{EmitError, EmittedSource, bindings, cuda};

pub(crate) trait PreparedProgram {
    fn render(self: Box<Self>) -> Result<EmittedSource, EmitError>;
}

/// Build bindings and prepare a validated program for its target.
pub(crate) fn prepare(plan: &PhysicalPlan) -> Result<Box<dyn PreparedProgram>, EmitError> {
    // Assign binding slots to External/Global tensors.
    let bindings = bindings::build_bindings(plan);

    match plan.target() {
        TargetCapability::Cuda(_) => Ok(Box::new(cuda::prepare(plan, bindings)?)),
    }
}
