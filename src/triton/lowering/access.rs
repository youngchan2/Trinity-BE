//! Bind a logically contained read to an available Triton local definition.
use super::super::plan::LocalRead;
use crate::analysis::access::{cell_of_tile, correspondence};
use crate::analysis::*;

pub(super) fn local_read(
    ir: &ScheduledIr,
    write: AccessId,
    read: AccessId,
    bindings: &Bindings,
) -> Option<LocalRead> {
    let w = ir.access(write);
    let r = ir.access(read);
    if w.statement.index() >= r.statement.index() || !ir.is_within(r.scope, w.scope) {
        return None;
    }
    let pairs = correspondence(w, r, ir, bindings)?;
    for (a, b) in &pairs {
        let wd = &w.index[*a];
        let rd = &r.index[*b];
        if wd == rd {
            continue;
        }
        match (wd, rd) {
            (IndexDim::FullTile, _) => {}
            (IndexDim::Tile { start, width }, IndexDim::Elem(IndexExpr::LoopVar(cell)))
                if cell_of_tile(*cell, start, width, ir) => {}
            _ => return None,
        }
    }
    Some(LocalRead {
        definition: write,
        axes: pairs,
    })
}
