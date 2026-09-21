//! Triton ownership and producer coverage checks using resolved shapes and storage.
use super::super::plan::Initialization;
use super::super::{Error, invalid};
use crate::analysis::access::{cell_of_tile, covers};
use crate::analysis::scalar::{constant, loop_range};
use crate::analysis::*;
use std::collections::BTreeSet;

pub(super) fn unowned_axes(
    ir: &ScheduledIr,
    write: &AccessInfo,
    parallel: &[ScopeId],
    options: &Bindings,
) -> Result<BTreeSet<ScopeId>, Error> {
    let mut result = BTreeSet::new();
    for id in parallel {
        let (_, _, step) = loop_range(ir, *id, &options.symbols)?;
        let owns = write.index.iter().any(|d| match d {
            IndexDim::Tile {
                start: IndexExpr::LoopVar(s),
                width,
            } if s == id => {
                constant(width, &options.symbols).is_ok_and(|w| w > 0 && w as usize <= step)
            }
            IndexDim::Elem(IndexExpr::LoopVar(s)) => s == id,
            _ => false,
        });
        let owns = owns
            || write.index.iter().any(|dim| {
                if let IndexDim::Elem(IndexExpr::LoopVar(cell)) = dim {
                    cell_of_tile(
                        *cell,
                        &IndexExpr::LoopVar(*id),
                        &ir.scope(*id).loop_info.as_ref().unwrap().step,
                        ir,
                    )
                } else {
                    false
                }
            });
        if !owns {
            result.insert(*id);
        }
    }
    Ok(result)
}

// Prove coverage for dense producer/consumer loops without equating loop variable names.
// Unknown region relations are rejected rather than zero-filled.
pub(super) fn validate_materialized_reads(
    ir: &ScheduledIr,
    uses: &[AccessId],
    representative: AccessId,
    init: Option<&Initialization>,
    options: &Bindings,
) -> Result<(), Error> {
    let mut definitions = Vec::new();
    if init.is_some() {
        definitions.push(representative);
    }
    for id in uses {
        let a = ir.access(*id);
        if a.kind == AccessKind::Write {
            definitions.push(*id);
            continue;
        }
        if !definitions
            .iter()
            .any(|w| covers(ir, ir.access(*w), a, options))
        {
            return Err(invalid(format!(
                "{}: materialized read {:?} has no covering earlier definition",
                ir.tensor(a.tensor).name,
                a.source_span
            )));
        }
    }
    Ok(())
}
