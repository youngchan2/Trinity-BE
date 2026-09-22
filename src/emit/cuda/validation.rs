//! Execution-region coverage and hazards. Coordinates are visited on the host;
//! no per-coordinate schedule or tensor-element table is emitted into CUDA.
use super::{
    execution::{DeviceStatement, Domain, Execution, coordinates},
    invalid,
};
use crate::emit::{
    EmitError,
    combine::CombinedPlan,
    provider::access::{Access, Axis},
};
use crate::{PhysicalPlan, Storage, ValueInstanceId};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Region {
    value: ValueInstanceId,
    axes: Vec<(i64, i64)>,
}

impl Region {
    fn overlaps(&self, other: &Self) -> bool {
        self.value == other.value
            && self
                .axes
                .iter()
                .zip(&other.axes)
                .all(|(&(a, z), &(b, y))| a < y && b < z)
    }

    fn subtract(&self, cut: &Self) -> Vec<Self> {
        if !self.overlaps(cut) {
            return vec![self.clone()];
        }

        let mut core = self.clone();
        let mut out = Vec::new();

        for axis in 0..self.axes.len() {
            let (a, z) = core.axes[axis];
            let (b, y) = cut.axes[axis];
            let (lo, hi) = (a.max(b), z.min(y));

            if a < lo {
                let mut r = core.clone();
                r.axes[axis] = (a, lo);
                out.push(r);
            }

            if hi < z {
                let mut r = core.clone();
                r.axes[axis] = (hi, z);
                out.push(r);
            }
            core.axes[axis] = (lo, hi);
        }

        out
    }
}

#[derive(Default, Clone)]
struct Regions(Vec<Region>);

impl Regions {
    fn covers(&self, region: &Region) -> bool {
        let mut missing = vec![region.clone()];
        for r in &self.0 {
            missing = missing.iter().flat_map(|p| p.subtract(r)).collect();
            if missing.is_empty() {
                return true;
            }
        }

        false
    }
    fn overlaps(&self, region: &Region) -> bool {
        self.0.iter().any(|r| r.overlaps(region))
    }

    fn insert(&mut self, region: Region) {
        let mut pieces = vec![region];
        for r in &self.0 {
            pieces = pieces.iter().flat_map(|p| p.subtract(r)).collect();
            if pieces.is_empty() {
                return;
            }
        }

        for mut piece in pieces {
            // Coalesce adjacent boxes to keep complete tensor coverage compact.
            loop {
                let merged = self.0.iter().enumerate().find_map(|(i, r)| {
                    if r.value != piece.value {
                        return None;
                    }

                    let differing: Vec<_> = r
                        .axes
                        .iter()
                        .zip(&piece.axes)
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .map(|(i, _)| i)
                        .collect();

                    if let [axis] = differing.as_slice() {
                        let (a, z) = r.axes[*axis];
                        let (b, y) = piece.axes[*axis];
                        if z == b || y == a {
                            return Some((i, *axis, (a.min(b), z.max(y))));
                        }
                    }

                    None
                });

                let Some((i, axis, bounds)) = merged else {
                    break;
                };

                self.0.swap_remove(i);
                piece.axes[axis] = bounds;
            }

            self.0.push(piece);
        }
    }
}

fn region(access: &Access, coordinates: &BTreeMap<String, i64>) -> Result<Vec<Region>, EmitError> {
    if access.storage == Storage::Register {
        return Ok(vec![]);
    }

    let mut axes = Vec::new();
    for (axis, &size) in access.axes.iter().zip(&access.shape) {
        let size = i64::try_from(size).map_err(|_| invalid("region extent exceeds int64"))?;
        let (start, width, clipped) = match axis {
            Axis::Full => (0, size, false),
            Axis::Tile {
                variable,
                width,
                clipped,
            } => (
                *coordinates
                    .get(variable)
                    .ok_or_else(|| invalid("missing tile coordinate"))?,
                *width as i64,
                *clipped,
            ),
            Axis::Element { variable, step } => (
                coordinates
                    .get(variable)
                    .ok_or_else(|| invalid("missing element coordinate"))?
                    .checked_div(step.evaluate(coordinates).map_err(invalid)?)
                    .ok_or_else(|| invalid("invalid element step"))?,
                1,
                false,
            ),
        };

        let end = start
            .checked_add(width)
            .ok_or_else(|| invalid("region endpoint overflow"))?;

        if start < 0 {
            return Err(invalid("negative execution region"));
        }

        if !clipped && end > size {
            return Err(invalid("unclipped execution region exceeds tensor"));
        }

        if start >= size {
            if clipped {
                return Ok(vec![]);
            }
            return Err(invalid("execution region starts outside tensor"));
        }

        axes.push((start, end.min(size)));
    }

    // Different views alias the same storage. Compare physical element intervals,
    // not unrelated logical coordinates from each view. Collapse the contiguous
    // suffix into one interval; only enumerate rows for a strided rectangle.
    let suffix = axes
        .iter()
        .zip(&access.shape)
        .rposition(|(&(start, end), &size)| start != 0 || end != size as i64)
        .unwrap_or(0);
    let stride = access.shape[suffix + 1..]
        .iter()
        .map(|&n| n as i64)
        .product::<i64>();
    fn rows(axes: &[(i64, i64)], shape: &[usize], prefix: i64, out: &mut Vec<i64>) {
        if axes.is_empty() {
            out.push(prefix);
            return;
        }
        for index in axes[0].0..axes[0].1 {
            rows(
                &axes[1..],
                &shape[1..],
                prefix * shape[0] as i64 + index,
                out,
            );
        }
    }
    let mut offsets = Vec::new();
    rows(&axes[..suffix], &access.shape[..suffix], 0, &mut offsets);
    Ok(offsets
        .into_iter()
        .map(|prefix| {
            let base = prefix * access.shape[suffix] as i64;
            Region {
                value: access.value,
                axes: vec![(
                    (base + axes[suffix].0) * stride,
                    (base + axes[suffix].1) * stride,
                )],
            }
        })
        .collect())
}

#[derive(Default)]
struct Effects {
    reads: Regions,
    writes: Regions,
}

struct Validator<'a, 'p> {
    combined: &'a CombinedPlan<'p>,
    available: Regions,
    effects: Effects,
}

impl Validator<'_, '_> {
    fn inputs(&mut self, access: &Access, coords: &BTreeMap<String, i64>) -> Result<(), EmitError> {
        for r in region(access, coords)? {
            if !self.available.covers(&r) {
                return Err(invalid(format!(
                    "value {} reads an unproduced region or requires another CTA in the same launch",
                    r.value.index()
                )));
            }
            self.effects.reads.insert(r);
        }

        Ok(())
    }

    fn outputs(
        &mut self,
        access: &Access,
        coords: &BTreeMap<String, i64>,
    ) -> Result<(), EmitError> {
        for r in region(access, coords)? {
            self.available.insert(r.clone());
            self.effects.writes.insert(r);
        }

        Ok(())
    }

    fn visit(
        &mut self,
        nodes: &[DeviceStatement<'_>],
        coords: &mut BTreeMap<String, i64>,
    ) -> Result<(), EmitError> {
        for node in nodes {
            match node {
                DeviceStatement::Sequential { domain, body } => {
                    for i in 0..domain.count {
                        coords.insert(domain.name.clone(), domain.start + i as i64 * domain.step);
                        self.visit(body, coords)?;
                    }
                    coords.remove(&domain.name);
                }
                DeviceStatement::Body(body) => {
                    for invocation in &body.roots {
                        let interface = self.combined.kernels[&invocation.operation]
                            .specification
                            .interface(body.requirements.block_threads);

                        if let Some(d) = &invocation.iteration {
                            let d = Domain::new(d)?;
                            for i in 0..d.count {
                                coords.insert(d.name.clone(), d.start + i as i64 * d.step);
                                for port in &interface.inputs {
                                    self.inputs(&port.access, coords)?;
                                }
                            }
                            coords.remove(&d.name);
                        } else {
                            for port in &interface.inputs {
                                self.inputs(&port.access, coords)?;
                            }
                        }

                        for port in &interface.outputs {
                            self.outputs(&port.access, coords)?;
                        }

                        // Register followers are validated inside their producer's
                        // output scope by combine; inspect their external effects.
                        for follower in &invocation.followers {
                            let interface = self.combined.kernels[follower]
                                .specification
                                .interface(body.requirements.block_threads);
                            for port in &interface.inputs {
                                self.inputs(&port.access, coords)?;
                            }
                            for port in &interface.outputs {
                                self.outputs(&port.access, coords)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub(super) fn validate(
    plan: &PhysicalPlan,
    combined: &CombinedPlan<'_>,
    execution: &Execution<'_>,
) -> Result<(), EmitError> {
    let full = |value| Region {
        value,
        axes: vec![(
            0,
            plan.value_instance(value)
                .unwrap()
                .shape()
                .iter()
                .map(|&n| n as i64)
                .product(),
        )],
    };

    let mut available = Regions::default();
    for input in plan.inputs() {
        available.insert(full(input.value()));
    }

    for kernel in &execution.kernels {
        let mut seen = Effects::default();
        for block in 0..kernel.blocks {
            let mut validator = Validator {
                combined,
                available: available.clone(),
                effects: Effects::default(),
            };

            validator.visit(&kernel.body, &mut coordinates(kernel, block))?;

            let effects = validator.effects;
            for write in &effects.writes.0 {
                if seen.writes.overlaps(write) || seen.reads.overlaps(write) {
                    return Err(invalid(
                        "unordered write/write or read/write overlap between CTAs",
                    ));
                }
            }

            for read in &effects.reads.0 {
                if seen.writes.overlaps(read) {
                    return Err(invalid("read requires another CTA in the same launch"));
                }
            }

            for r in effects.reads.0 {
                seen.reads.insert(r);
            }

            for r in effects.writes.0 {
                seen.writes.insert(r);
            }
        }

        for r in seen.writes.0 {
            available.insert(r);
        }
    }

    if !available.covers(&full(plan.output().value())) {
        return Err(invalid("output region is not fully produced"));
    }
    Ok(())
}
