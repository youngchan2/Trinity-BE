//! Triton grid restrictions, after common lexical loop analysis.
use super::super::{Error, invalid};
use crate::analysis::*;

pub(super) fn validate(ir: &ScheduledIr, id: ScopeId) -> Result<(), Error> {
    let scope = ir.scope(id);
    let info = scope.loop_info.as_ref().unwrap();
    if scope.kind.is_parallel()
        && (!info.start.loop_dependencies().is_empty() || !info.end.loop_dependencies().is_empty())
    {
        return Err(invalid(
            "parallel grid bounds must be program-wide scalar parameters",
        ));
    }
    Ok(())
}
