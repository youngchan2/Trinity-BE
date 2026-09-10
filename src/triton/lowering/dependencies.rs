//! Triton ownership and producer coverage checks using resolved shapes and storage.
use super::super::plan::Initialization;
use super::super::shape::{constant, loop_range};
use super::super::{Error, Options, invalid};
use crate::analysis::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn unowned_axes(
    ir: &ProgramAnalysis,
    write: &AccessInfo,
    parallel: &[ScopeId],
    options: &Options,
) -> Result<BTreeSet<ScopeId>, Error> {
    let mut result = BTreeSet::new();
    for id in parallel {
        let (_, _, step) = loop_range(ir, *id, options)?;
        let owns = write.index.iter().any(|d| match d {
            IndexDim::Tile {
                start: IndexExpr::LoopVar(s),
                width,
            } if s == id => constant(width, options).is_ok_and(|w| w > 0 && w as usize <= step),
            IndexDim::Elem(IndexExpr::LoopVar(s)) => s == id,
            _ => false,
        });
        let owns = owns
            || write.index.iter().any(|dim| {
                if let IndexDim::Elem(IndexExpr::LoopVar(cell)) = dim {
                    super::access::cell_of_tile(
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
    ir: &ProgramAnalysis,
    uses: &[AccessId],
    representative: AccessId,
    init: Option<&Initialization>,
    options: &Options,
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

pub(super) fn covers(
    ir: &ProgramAnalysis,
    write: &AccessInfo,
    read: &AccessInfo,
    options: &Options,
) -> bool {
    if ir.scope(write.scope).kernel != ir.scope(read.scope).kernel
        && super::access::full_producer(ir, write, read, options)
    {
        return true;
    }
    if write.index.len() != read.index.len() || write.view_shape != read.view_shape {
        return false;
    }
    let mut mapping = BTreeMap::new();
    let mut inverse = BTreeMap::new();
    let cross_kernel = ir.scope(write.scope).kernel != ir.scope(read.scope).kernel;
    write
        .index
        .iter()
        .zip(&read.index)
        .enumerate()
        .all(|(axis, (w, r))| {
            if w == r {
                return true;
            }
            match (w, r) {
                (
                    IndexDim::Tile {
                        start: IndexExpr::LoopVar(ws),
                        width: ww,
                    },
                    IndexDim::Tile {
                        start: IndexExpr::LoopVar(rs),
                        width: rw,
                    },
                ) => {
                    let Ok((w0, w1, step)) = loop_range(ir, *ws, options) else {
                        return false;
                    };
                    let Ok((r0, r1, rstep)) = loop_range(ir, *rs, options) else {
                        return false;
                    };
                    if mapping.insert(*ws, *rs).is_some_and(|old| old != *rs)
                        || inverse.insert(*rs, *ws).is_some_and(|old| old != *ws)
                    {
                        return false;
                    }
                    (cross_kernel
                        || (ir.scope(*ws).kind == ScopeKind::SequentialLoop
                            && ir.scope(*rs).kind == ScopeKind::SequentialLoop))
                        && constant(ww, options).ok() == Some(step as i64)
                        && constant(rw, options).ok() == Some(rstep as i64)
                        && w0 <= r0
                        && w1 >= r1
                        && !ir.is_within(read.scope, *ws)
                }
                (
                    IndexDim::Tile {
                        start: IndexExpr::LoopVar(ws),
                        width,
                    },
                    IndexDim::FullTile,
                ) => {
                    let Ok((start, end, step)) = loop_range(ir, *ws, options) else {
                        return false;
                    };
                    let extent = if let Some(shape) = &write.view_shape {
                        constant(&shape[axis], options).unwrap_or(-1) as usize
                    } else {
                        options.shapes[&ir.tensor(write.tensor).name][axis]
                    };
                    (cross_kernel || ir.scope(*ws).kind == ScopeKind::SequentialLoop)
                        && !ir.is_within(read.scope, *ws)
                        && start == 0
                        && end >= extent as i64
                        && constant(width, options).ok() == Some(step as i64)
                        && write
                            .index
                            .iter()
                            .filter(|d| d.loop_dependencies().contains(ws))
                            .count()
                            == 1
                }
                (IndexDim::FullTile, _) => true,
                (
                    IndexDim::Elem(IndexExpr::LoopVar(ws)),
                    IndexDim::Elem(IndexExpr::LoopVar(rs)),
                ) if cross_kernel => {
                    loop_range(ir, *ws, options).ok() == loop_range(ir, *rs, options).ok()
                }
                _ => false,
            }
        })
}
