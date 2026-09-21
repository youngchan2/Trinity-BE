//! Triton block shapes. Logical views and scalar evaluation live in analysis.
use super::{Error, Options, invalid};
pub use crate::analysis::access::AxisAccess;
use crate::analysis::{IndexExpr, ScheduledIr, ScopeId, access::ResolvedAccess};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileAccess {
    pub axes: Vec<AxisAccess>,
    /// Padded Triton block shape, distinct from ResolvedAccess's logical widths.
    pub shape: Vec<usize>,
}

pub(crate) fn tile(access: &ResolvedAccess) -> Result<TileAccess, Error> {
    let shape: Vec<_> = access
        .shape
        .iter()
        .map(|width| {
            width
                .checked_next_power_of_two()
                .ok_or_else(|| invalid("tile overflow"))
        })
        .collect::<Result<_, _>>()?;
    if product(&shape)? > 1_048_576 {
        return Err(invalid("tile exceeds Triton's element limit"));
    }
    Ok(TileAccess {
        axes: access.axes.clone(),
        shape,
    })
}

// The public Options type still accepts concrete shapes and symbols. These thin
// adapters keep codegen independent of how its input configuration was supplied.
pub(crate) fn constant(expr: &IndexExpr, options: &Options) -> Result<i64, Error> {
    Ok(crate::analysis::scalar::constant(expr, &options.symbols)?)
}
pub(crate) fn positive(expr: &IndexExpr, options: &Options) -> Result<usize, Error> {
    Ok(crate::analysis::scalar::positive(expr, &options.symbols)?)
}
pub(crate) fn loop_range(
    ir: &ScheduledIr,
    id: ScopeId,
    options: &Options,
) -> Result<(i64, i64, usize), Error> {
    Ok(crate::analysis::scalar::loop_range(
        ir,
        id,
        &options.symbols,
    )?)
}
pub(crate) fn product(shape: &[usize]) -> Result<usize, Error> {
    Ok(crate::analysis::scalar::product(shape)?)
}
