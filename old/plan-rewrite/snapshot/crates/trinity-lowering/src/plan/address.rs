//! View/index parsing and bounds checking for Loop IR input.
use std::collections::BTreeMap;

use super::PhysicalInvariantError;

fn fail(message: impl Into<String>) -> PhysicalInvariantError {
    PhysicalInvariantError::InvalidProgram(message.into())
}
use crate::{Expression as E, PhysicalPlan};

#[derive(Debug, Clone)]
pub(crate) enum Axis {
    Full,
    Tile {
        variable: String,
        width: usize,
        clipped: bool,
    },
    Element(String),
}

pub(crate) struct Address {
    pub shape: Vec<usize>,
    pub axes: Vec<Axis>,
}

fn args(e: &E) -> Result<&[E], PhysicalInvariantError> {
    e.list()
        .filter(|a| !a.is_empty())
        .map(|a| &a[1..])
        .ok_or_else(|| fail("expected expression"))
}
fn atom(e: &E) -> Result<&str, PhysicalInvariantError> {
    e.atom().ok_or_else(|| fail("expected atom"))
}
fn number(e: &E) -> Result<usize, PhysicalInvariantError> {
    atom(e)?.parse().map_err(|_| fail("expected extent"))
}

pub(crate) fn parse(
    plan: &PhysicalPlan,
    view: &E,
    index: &E,
) -> Result<Address, PhysicalInvariantError> {
    let v = args(view)?;
    if view.operator() != Some("view") || v.len() != 2 {
        return Err(fail("invalid view"));
    }
    let base = args(&v[0])?;
    let name = base
        .first()
        .and_then(E::atom)
        .ok_or_else(|| fail("invalid tensor name"))?;
    let (_, value) = plan
        .value_instances()
        .find(|(_, v)| v.name() == Some(name))
        .ok_or_else(|| fail("unknown tensor name"))?;
    let layout = args(&v[1])?;
    if layout.len() != value.shape().len()
        || !(1..=3).contains(&layout.len())
        || index.operator() != Some("keyed_index")
    {
        return Err(fail("view/index rank mismatch"));
    }
    let mut slots = BTreeMap::new();
    for slot in args(index)? {
        let s = args(slot)?;
        if s.len() != 2 || slots.insert(atom(&s[0])?, &s[1]).is_some() {
            return Err(fail("duplicate/invalid index slot"));
        }
    }
    let mut axes = Vec::new();
    for (axis, &size) in layout.iter().zip(value.shape()) {
        let a = args(axis)?;
        if a.len() != 2 || number(&a[1])? != size {
            return Err(fail("view extent differs from value"));
        }
        let parsed = match slots.remove(atom(&a[0])?) {
            None => Axis::Full,
            Some(p) if p.atom() == Some("fulltile") => Axis::Full,
            Some(p) => {
                let xs = args(p)?;
                let variable =
                    atom(xs.first().ok_or_else(|| fail("missing index variable"))?)?.to_owned();
                match p.operator() {
                    Some("tile" | "clipped_tile") if xs.len() == 2 => {
                        let width = number(&xs[1])?;
                        if width == 0 {
                            return Err(fail("zero tile width"));
                        }
                        Axis::Tile {
                            variable,
                            width,
                            clipped: p.operator() == Some("clipped_tile"),
                        }
                    }
                    Some("elem") if xs.len() == 1 => Axis::Element(variable),
                    _ => return Err(fail("unsupported access index")),
                }
            }
        };
        axes.push(parsed);
    }
    if !slots.is_empty() {
        return Err(fail("index slot absent from view"));
    }
    Ok(Address {
        shape: value.shape().to_vec(),
        axes,
    })
}

impl Address {
    pub fn resolve(
        &self,
        bindings: &BTreeMap<String, i64>,
        steps: &BTreeMap<String, i64>,
    ) -> Result<Bounds, PhysicalInvariantError> {
        let mut origin = [0; 3];
        let mut extent = [1; 3];
        for (i, (axis, &size)) in self.axes.iter().zip(&self.shape).enumerate() {
            let coordinate = |v: &str| -> Result<usize, PhysicalInvariantError> {
                usize::try_from(
                    *bindings
                        .get(v)
                        .ok_or_else(|| fail(format!("unbound index {v}")))?,
                )
                .map_err(|_| fail("negative index"))
            };
            match axis {
                Axis::Full => extent[i] = size,
                Axis::Tile {
                    variable,
                    width,
                    clipped,
                } => {
                    origin[i] = coordinate(variable)?;
                    extent[i] = if *clipped {
                        (*width).min(size.saturating_sub(origin[i]))
                    } else {
                        *width
                    };
                }
                Axis::Element(variable) => {
                    let step = steps
                        .get(variable)
                        .copied()
                        .filter(|s| *s > 0)
                        .ok_or_else(|| fail("invalid/missing elem step"))?;
                    origin[i] = coordinate(variable)? / step as usize;
                    extent[i] = 1;
                }
            }
            if extent[i] == 0
                || origin[i]
                    .checked_add(extent[i])
                    .is_none_or(|end| end > size)
            {
                return Err(fail("out-of-bounds execution region"));
            }
        }
        Ok(Bounds { origin })
    }
}

/// Resolved origin used by target-specific input alignment checks.
pub(crate) struct Bounds {
    pub origin: [usize; 3],
}
