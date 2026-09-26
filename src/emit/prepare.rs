//! Backend-neutral preparation for operation candidate discovery.
use super::EmitError;
use crate::analysis::regions::{OperationScope, operation_scopes};
use crate::{OperationId, PhysicalPlan};
use std::collections::BTreeMap;

pub(super) struct PreparedPlan<'a> {
    pub plan: &'a PhysicalPlan,
    pub scopes: BTreeMap<OperationId, OperationScope>,
}

pub(super) fn prepare(plan: &PhysicalPlan) -> Result<PreparedPlan<'_>, EmitError> {
    let symbols = plan.symbols();
    if !symbols.is_empty() {
        return Err(EmitError::InvalidExecution {
            reason: format!(
                "unbound configuration symbols {symbols:?}; call PhysicalPlan::bind_symbols before emission"
            ),
        });
    }
    Ok(PreparedPlan {
        plan,
        scopes: operation_scopes(plan),
    })
}
