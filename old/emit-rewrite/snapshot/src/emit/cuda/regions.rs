use super::{EmitError, Region, access::fail};
use crate::{PhysicalPlan, ValueInstanceId};

pub(super) fn overlaps(a: Region, b: Region) -> bool {
    a.value == b.value
        && a.rank == b.rank
        && (0..3).all(|i| {
            a.origin[i] < b.origin[i] + b.extent[i] && b.origin[i] < a.origin[i] + a.extent[i]
        })
}
/// Exact rectangular subtraction; disjoint pieces retain the previous version.
pub(super) fn subtract(region: Region, cut: Region) -> Vec<Region> {
    if !overlaps(region, cut) {
        return vec![region];
    }
    let mut core = region;
    let mut pieces = Vec::new();
    for i in 0..3 {
        let lo = core.origin[i].max(cut.origin[i]);
        let hi = (core.origin[i] + core.extent[i]).min(cut.origin[i] + cut.extent[i]);
        if core.origin[i] < lo {
            let mut p = core;
            p.extent[i] = lo - core.origin[i];
            pieces.push(p);
            core.extent[i] -= lo - core.origin[i];
            core.origin[i] = lo;
        }
        if core.origin[i] + core.extent[i] > hi {
            let mut p = core;
            p.origin[i] = hi;
            p.extent[i] = core.origin[i] + core.extent[i] - hi;
            pieces.push(p);
            core.extent[i] = hi - core.origin[i];
        }
    }
    pieces
}
pub(super) fn validate_region(plan: &PhysicalPlan, r: Region) -> Result<(), EmitError> {
    let value = plan
        .value_instance(r.value)
        .ok_or_else(|| fail("unknown value"))?;
    let shape = region_shape(value.shape())?;
    if r.rank >= plan.world_size()
        || r.extent.contains(&0)
        || (0..3).any(|i| {
            r.origin[i]
                .checked_add(r.extent[i])
                .is_none_or(|end| end > shape[i])
        })
    {
        return Err(fail("out-of-bounds execution region"));
    }
    Ok(())
}
pub(crate) fn full_region(plan: &PhysicalPlan, value: ValueInstanceId, rank: usize) -> Region {
    Region::new(
        value,
        rank,
        [0; 3],
        region_shape(plan.value_instance(value).unwrap().shape()).expect("validated tensor rank"),
    )
}
pub(crate) fn region_shape(shape: &[usize]) -> Result<[usize; 3], EmitError> {
    if !(1..=3).contains(&shape.len()) {
        return Err(fail("supported tensor ranks are 1, 2, 3"));
    }
    let mut out = [1; 3];
    out[..shape.len()].copy_from_slice(shape);
    Ok(out)
}
