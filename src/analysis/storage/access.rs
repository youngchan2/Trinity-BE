//! Bind a contained read to an earlier local definition, independent of codegen.
use super::LocalRead;
use crate::analysis::access::{cell_of_tile, correspondence, resolve};
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
    let pairs = if let Some(pairs) = correspondence(w, r, ir, bindings) {
        pairs
    } else {
        // A register tile [.., N] can be viewed as [.., P, C] when the
        // prefix coordinates agree and N is fully owned by this program.
        // Do not mistake a partial N tile for the complete factored axis.
        if w.tensor != r.tensor
            || r.index.len() != w.index.len() + 1
            || w.index.last() != Some(&IndexDim::FullTile)
        {
            return None;
        }
        let wa = resolve(w, ir, bindings).ok()?;
        let ra = resolve(r, ir, bindings).ok()?;
        let n = w.index.len() - 1;
        if (0..n).any(|i| w.index[i] != r.index[i] || wa.axes[i].extent != ra.axes[i].extent) {
            return None;
        }
        let extents = [ra.axes[n].extent, ra.axes[n + 1].extent];
        if extents[0].checked_mul(extents[1])? != wa.axes[n].extent {
            return None;
        }
        for axis in &ra.axes[n..] {
            let start = super::super::scalar::constant(&axis.start, &bindings.symbols).ok()?;
            if start < 0 || usize::try_from(start).ok()?.checked_add(axis.width)? > axis.extent {
                return None;
            }
        }
        return Some(LocalRead {
            definition: write,
            axes: (0..n).map(|i| (i, i)).collect(),
            split_last: Some(extents),
        });
    };
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
        split_last: None,
    })
}
