//! Logical views, access regions and producer coverage; no implementation layout.
use super::scalar::{constant, loop_range, positive, product, validate_index};
use super::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisAccess {
    pub start: IndexExpr,
    pub width: usize,
    pub extent: usize,
    /// Contiguous logical-view stride, not an arbitrary runtime buffer stride.
    pub stride: usize,
    /// Logical upper bound for a loop-backed tile, in addition to the view extent.
    pub loop_end: Option<IndexExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAccess {
    pub axes: Vec<AxisAccess>,
    /// Logical widths before any implementation padding.
    pub shape: Vec<usize>,
}

pub fn resolve(
    a: &AccessInfo,
    ir: &ScheduledIr,
    bindings: &Bindings,
) -> Result<ResolvedAccess, ResolveError> {
    let name = &ir.tensor(a.tensor).name;
    let base_shape = bindings
        .shapes
        .get(name)
        .ok_or_else(|| invalid(format!("missing shape for {name}")))?;
    let view_shape = a
        .view_shape
        .as_ref()
        .map(|shape| {
            shape
                .iter()
                .map(|s| positive(s, &bindings.symbols))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let shape = view_shape.as_ref().unwrap_or(base_shape);
    if product(shape)? != product(base_shape)? {
        return Err(invalid(format!(
            "{name}: contiguous view changes the element count"
        )));
    }
    if shape.len() != a.index.len() || shape.is_empty() || shape.contains(&0) {
        return Err(invalid(format!(
            "{name}: shape {shape:?} does not match access rank {}",
            a.index.len()
        )));
    }
    let mut axes = Vec::new();
    for (i, dim) in a.index.iter().enumerate() {
        let (start, width, loop_end) = match dim {
            IndexDim::FullTile => (IndexExpr::Integer(0), shape[i], None),
            IndexDim::Elem(expr) => {
                // Legacy elem denotes the tile ordinal, not the raw tile start.
                let start = if let IndexExpr::LoopVar(id) = expr {
                    IndexExpr::Apply(
                        "//".into(),
                        vec![
                            expr.clone(),
                            ir.scope(*id).loop_info.as_ref().unwrap().step.clone(),
                        ],
                    )
                } else {
                    expr.clone()
                };
                (start, 1, None)
            }
            IndexDim::Tile { start, width } => {
                let end = if let IndexExpr::LoopVar(id) = start {
                    Some(ir.scope(*id).loop_info.as_ref().unwrap().end.clone())
                } else {
                    None
                };
                (start.clone(), positive(width, &bindings.symbols)?, end)
            }
            IndexDim::ConstTile { start, width } => {
                (start.clone(), positive(width, &bindings.symbols)?, None)
            }
        };
        validate_index(&start, &bindings.symbols)?;
        axes.push(AxisAccess {
            start,
            width,
            extent: shape[i],
            stride: product(&shape[i + 1..])?,
            loop_end,
        });
    }
    let shape = axes.iter().map(|axis| axis.width).collect();
    Ok(ResolvedAccess { axes, shape })
}

pub fn axes(a: &AccessInfo, ir: &ScheduledIr, bindings: &Bindings) -> Vec<usize> {
    (0..a.index.len())
        .filter(|i| {
            if let Some(shape) = &a.view_shape {
                shape[*i] != IndexExpr::Integer(1)
            } else {
                bindings.shapes[&ir.tensor(a.tensor).name][*i] != 1
            }
        })
        .collect()
}

fn extent(a: &AccessInfo, axis: usize, ir: &ScheduledIr, bindings: &Bindings) -> Option<i64> {
    a.view_shape
        .as_ref()
        .map(|shape| constant(&shape[axis], &bindings.symbols).ok())
        .unwrap_or_else(|| Some(bindings.shapes[&ir.tensor(a.tensor).name][axis] as i64))
}

pub fn correspondence(
    w: &AccessInfo,
    r: &AccessInfo,
    ir: &ScheduledIr,
    bindings: &Bindings,
) -> Option<Vec<(usize, usize)>> {
    if w.tensor != r.tensor {
        return None;
    }
    let wa = axes(w, ir, bindings);
    let ra = axes(r, ir, bindings);
    if wa.len() != ra.len() {
        return None;
    }
    let pairs: Vec<_> = wa.into_iter().zip(ra).collect();
    pairs
        .iter()
        .all(|(a, b)| extent(w, *a, ir, bindings) == extent(r, *b, ir, bindings))
        .then_some(pairs)
}

/// A serial cell binding enumerates exactly a parent tile. This relation is
/// structural, so it remains true when the tile parameter changes during tuning.
pub fn cell_of_tile(cell: ScopeId, start: &IndexExpr, width: &IndexExpr, ir: &ScheduledIr) -> bool {
    let info = ir.scope(cell).loop_info.as_ref().unwrap();
    ir.scope(cell).kind == ScopeKind::SequentialLoop
        && info.step == IndexExpr::Integer(1)
        && info.start == *start
        && (info.end == IndexExpr::Apply("+".into(), vec![start.clone(), width.clone()])
            || info.end == IndexExpr::Apply("+".into(), vec![width.clone(), start.clone()]))
}

/// Prove a complete dense producer across its loops. Each changing axis must
/// have its own independent binding; diagonal writes do not cover a rectangle.
pub fn full_producer(
    ir: &ScheduledIr,
    write: &AccessInfo,
    read: &AccessInfo,
    bindings: &Bindings,
) -> bool {
    if write.tensor != read.tensor {
        return false;
    }
    let mut used = std::collections::BTreeSet::new();
    (0..write.index.len()).all(|axis| {
        let Some(size) = extent(write, axis, ir, bindings) else {
            return false;
        };
        match &write.index[axis] {
            IndexDim::FullTile => true,
            IndexDim::ConstTile { start, width } => {
                constant(start, &bindings.symbols).ok() == Some(0)
                    && constant(width, &bindings.symbols).is_ok_and(|w| w >= size)
            }
            IndexDim::Tile {
                start: IndexExpr::LoopVar(id),
                width,
            } => {
                used.insert(*id)
                    && loop_range(ir, *id, &bindings.symbols).is_ok_and(|(s, e, step)| {
                        s == 0
                            && e >= size
                            && constant(width, &bindings.symbols).ok() == Some(step as i64)
                    })
            }
            IndexDim::Elem(IndexExpr::LoopVar(id)) => {
                used.insert(*id)
                    && loop_range(ir, *id, &bindings.symbols)
                        .is_ok_and(|(s, e, step)| s == 0 && e.div_euclid(step as i64) >= size)
            }
            _ => false,
        }
    })
}
pub fn covers(
    ir: &ScheduledIr,
    write: &AccessInfo,
    read: &AccessInfo,
    bindings: &Bindings,
) -> bool {
    if write.tensor != read.tensor {
        return false;
    }
    if ir.scope(write.scope).kernel != ir.scope(read.scope).kernel
        && full_producer(ir, write, read, bindings)
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
                    let Ok((w0, w1, step)) = loop_range(ir, *ws, &bindings.symbols) else {
                        return false;
                    };
                    let Ok((r0, r1, rstep)) = loop_range(ir, *rs, &bindings.symbols) else {
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
                        && constant(ww, &bindings.symbols).ok() == Some(step as i64)
                        && constant(rw, &bindings.symbols).ok() == Some(rstep as i64)
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
                    let Ok((start, end, step)) = loop_range(ir, *ws, &bindings.symbols) else {
                        return false;
                    };
                    let extent = if let Some(shape) = &write.view_shape {
                        constant(&shape[axis], &bindings.symbols).unwrap_or(-1) as usize
                    } else {
                        bindings.shapes[&ir.tensor(write.tensor).name][axis]
                    };
                    (cross_kernel || ir.scope(*ws).kind == ScopeKind::SequentialLoop)
                        && !ir.is_within(read.scope, *ws)
                        && start == 0
                        && end >= extent as i64
                        && constant(width, &bindings.symbols).ok() == Some(step as i64)
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
                    loop_range(ir, *ws, &bindings.symbols).ok()
                        == loop_range(ir, *rs, &bindings.symbols).ok()
                }
                _ => false,
            }
        })
}
