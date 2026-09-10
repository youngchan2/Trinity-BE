//! View correspondence and local tile containment; no tensor-name heuristics.
use super::super::Options;
use super::super::plan::LocalRead;
use super::super::shape::{constant, loop_range};
use crate::analysis::*;

pub(super) fn axes(a: &AccessInfo, ir: &ProgramAnalysis, options: &Options) -> Vec<usize> {
    (0..a.index.len())
        .filter(|i| {
            if let Some(shape) = &a.view_shape {
                shape[*i] != IndexExpr::Integer(1)
            } else {
                options.shapes[&ir.tensor(a.tensor).name][*i] != 1
            }
        })
        .collect()
}

fn extent(a: &AccessInfo, axis: usize, ir: &ProgramAnalysis, options: &Options) -> Option<i64> {
    a.view_shape
        .as_ref()
        .map(|shape| constant(&shape[axis], options).ok())
        .unwrap_or_else(|| Some(options.shapes[&ir.tensor(a.tensor).name][axis] as i64))
}

pub(super) fn correspondence(
    w: &AccessInfo,
    r: &AccessInfo,
    ir: &ProgramAnalysis,
    options: &Options,
) -> Option<Vec<(usize, usize)>> {
    if w.tensor != r.tensor {
        return None;
    }
    let wa = axes(w, ir, options);
    let ra = axes(r, ir, options);
    if wa.len() != ra.len() {
        return None;
    }
    let pairs: Vec<_> = wa.into_iter().zip(ra).collect();
    pairs
        .iter()
        .all(|(a, b)| extent(w, *a, ir, options) == extent(r, *b, ir, options))
        .then_some(pairs)
}

/// A serial cell binding enumerates exactly a parent tile. This relation is
/// structural, so it remains true when the tile parameter changes during tuning.
pub(super) fn cell_of_tile(
    cell: ScopeId,
    start: &IndexExpr,
    width: &IndexExpr,
    ir: &ProgramAnalysis,
) -> bool {
    let info = ir.scope(cell).loop_info.as_ref().unwrap();
    ir.scope(cell).kind == ScopeKind::SequentialLoop
        && info.step == IndexExpr::Integer(1)
        && info.start == *start
        && (info.end == IndexExpr::Apply("+".into(), vec![start.clone(), width.clone()])
            || info.end == IndexExpr::Apply("+".into(), vec![width.clone(), start.clone()]))
}

pub(super) fn local_read(
    ir: &ProgramAnalysis,
    write: AccessId,
    read: AccessId,
    options: &Options,
) -> Option<LocalRead> {
    let w = ir.access(write);
    let r = ir.access(read);
    if w.statement.index() >= r.statement.index() || !ir.is_within(r.scope, w.scope) {
        return None;
    }
    let pairs = correspondence(w, r, ir, options)?;
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

/// Prove a complete dense producer across its loops. Each changing axis must
/// have its own independent binding; diagonal writes do not cover a rectangle.
pub(super) fn full_producer(
    ir: &ProgramAnalysis,
    write: &AccessInfo,
    read: &AccessInfo,
    options: &Options,
) -> bool {
    if write.tensor != read.tensor {
        return false;
    }
    let mut used = std::collections::BTreeSet::new();
    (0..write.index.len()).all(|axis| {
        let Some(size) = extent(write, axis, ir, options) else {
            return false;
        };
        match &write.index[axis] {
            IndexDim::FullTile => true,
            IndexDim::ConstTile { start, width } => {
                constant(start, options).ok() == Some(0)
                    && constant(width, options).is_ok_and(|w| w >= size)
            }
            IndexDim::Tile {
                start: IndexExpr::LoopVar(id),
                width,
            } => {
                used.insert(*id)
                    && loop_range(ir, *id, options).is_ok_and(|(s, e, step)| {
                        s == 0 && e >= size && constant(width, options).ok() == Some(step as i64)
                    })
            }
            IndexDim::Elem(IndexExpr::LoopVar(id)) => {
                used.insert(*id)
                    && loop_range(ir, *id, options)
                        .is_ok_and(|(s, e, step)| s == 0 && e.div_euclid(step as i64) >= size)
            }
            _ => false,
        }
    })
}
