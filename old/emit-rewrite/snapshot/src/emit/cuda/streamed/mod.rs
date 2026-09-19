//! Streamed grids, launch metadata, and execution construction.
pub(super) mod render;

use serde::Serialize;
use std::collections::BTreeMap;

use super::{EmitError, Task, access::fail, domain::TaskDomain};
use crate::LoopDomain;

pub(super) fn build_grids(domains: &[TaskDomain]) -> Result<Vec<Grid>, EmitError> {
    domains.iter().map(|d| Grid::new(&d.task.domain)).collect()
}

pub(super) fn build(
    domains: Vec<TaskDomain>,
    grids: Vec<Grid>,
) -> Result<StreamedExecution, EmitError> {
    StreamedExecution::new(domains.into_iter().map(|d| d.task).collect(), grids)
}

/// Axis iteration counts of the rectangular launch envelope. These describe
/// blocks, not scheduler tasks. Bounds inside a body never contribute axes.
#[derive(Debug, Clone, Serialize)]
pub struct Grid {
    pub axes: Vec<u64>,
    pub blocks: u64,
}
impl Grid {
    pub(super) fn new(domain: &[LoopDomain]) -> Result<Self, EmitError> {
        fn visit(
            ds: &[LoopDomain],
            b: &mut BTreeMap<String, i64>,
            counts: &mut [u64],
        ) -> Result<(), EmitError> {
            if let Some((d, rest)) = ds.split_first() {
                let (start, stop, step) = d.bounds(b).map_err(fail)?;
                counts[0] = counts[0].max(((stop - start) / step) as u64);
                if !rest.is_empty() {
                    // Constant suffixes need no prefix coordinate enumeration.
                    if rest.iter().all(|d| d.bounds(&BTreeMap::new()).is_ok()) {
                        for (d, count) in rest.iter().zip(&mut counts[1..]) {
                            let (a, z, s) = d.bounds(&BTreeMap::new()).map_err(fail)?;
                            *count = (*count).max(((z - a) / s) as u64);
                        }
                    } else {
                        for value in (start..stop).step_by(step as usize) {
                            b.insert(d.variable.clone(), value);
                            visit(rest, b, &mut counts[1..])?;
                        }
                        b.remove(&d.variable);
                    }
                }
            }
            Ok(())
        }
        let mut axes = vec![0; domain.len()];
        visit(domain, &mut BTreeMap::new(), &mut axes)?;
        let blocks = axes
            .iter()
            .try_fold(1u64, |a, b| a.checked_mul(*b))
            .filter(|n| *n <= i64::MAX as u64)
            .ok_or_else(|| fail("streamed grid size overflows int64"))?;
        Ok(Self { axes, blocks })
    }

    /// Decode a block exactly as the kernel wrapper does. None denotes padding
    /// in an enclosing-coordinate-dependent parallel domain.
    pub fn coordinates(&self, task: &Task, mut block: u64) -> Result<Option<Vec<i64>>, EmitError> {
        if self.axes.len() != task.domain.len() || self.axes.contains(&0) || block >= self.blocks {
            return Err(fail("invalid grid coordinate"));
        }
        let mut iterations = vec![0; self.axes.len()];
        for i in (0..iterations.len()).rev() {
            iterations[i] = block % self.axes[i];
            block /= self.axes[i];
        }
        let mut bindings = BTreeMap::new();
        let mut coordinates = Vec::new();
        for (d, i) in task.domain.iter().zip(iterations) {
            let (start, stop, step) = d.bounds(&bindings).map_err(fail)?;
            if i >= ((stop - start) / step) as u64 {
                return Ok(None);
            }
            let value = start
                .checked_add(
                    (i as i64)
                        .checked_mul(step)
                        .ok_or_else(|| fail("coordinate overflow"))?,
                )
                .ok_or_else(|| fail("coordinate overflow"))?;
            bindings.insert(d.variable.clone(), value);
            coordinates.push(value);
        }
        Ok(Some(coordinates))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamedLaunch {
    pub task: usize,
    pub block_base: u64,
    pub blocks: u32,
}

#[derive(Debug, Clone)]
pub struct StreamedExecution {
    pub tasks: Vec<Task>,
    /// One grid envelope per task.
    pub grids: Vec<Grid>,
    /// Actual kernel submissions, in stream order.
    pub launches: Vec<StreamedLaunch>,
}
impl StreamedExecution {
    pub(super) const MAX_GRID_X: u64 = i32::MAX as u64;
    pub(super) fn new(tasks: Vec<Task>, grids: Vec<Grid>) -> Result<Self, EmitError> {
        if tasks.len() != grids.len() {
            return Err(fail("task/grid count mismatch"));
        }
        let mut launches = Vec::new();
        for (task, grid) in grids.iter().enumerate() {
            let mut base = 0;
            while base < grid.blocks {
                let count = (grid.blocks - base).min(Self::MAX_GRID_X);
                launches.push(StreamedLaunch {
                    task,
                    block_base: base,
                    blocks: count as u32,
                });
                base += count;
            }
        }
        Ok(Self {
            tasks,
            grids,
            launches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IndexExpr as E;
    fn axis(name: &str, start: i64, stop: i64, step: i64) -> LoopDomain {
        LoopDomain {
            variable: name.into(),
            start: E::Constant(start),
            stop: E::Constant(stop),
            step: E::Constant(step),
        }
    }
    #[test]
    fn grid_limit_splits_submissions_without_enumerating_or_duplicating_coordinates() {
        let limit = StreamedExecution::MAX_GRID_X;
        let task = Task {
            body: 0,
            statement: 0,
            path: vec![0],
            domain: vec![axis("i", 2, 2 + (limit as i64 + 3) * 2, 2)],
        };
        let grid = Grid::new(&task.domain).unwrap();
        let e = StreamedExecution::new(vec![task], vec![grid]).unwrap();
        assert_eq!(e.launches.len(), 2);
        assert_eq!(
            (e.launches[0].block_base, e.launches[0].blocks),
            (0, limit as u32)
        );
        assert_eq!((e.launches[1].block_base, e.launches[1].blocks), (limit, 3));
        for block in [0, limit - 1, limit, limit + 2] {
            assert_eq!(
                e.grids[0].coordinates(&e.tasks[0], block).unwrap(),
                Some(vec![2 + block as i64 * 2])
            );
        }
    }
    #[test]
    fn overflowing_grid_and_bound_arithmetic_are_rejected() {
        assert!(Grid::new(&[axis("i", 0, i64::MAX, 1), axis("j", 0, 2, 1)]).is_err());
        let mut d = axis("i", 0, 1, 1);
        d.stop = E::Add(Box::new(E::Constant(i64::MAX)), Box::new(E::Constant(1)));
        assert!(Grid::new(&[d]).is_err());
    }
}
