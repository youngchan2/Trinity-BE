//! Validate scalar loop expressions separately from static tile extents.
use super::super::shape::{constant, positive};
use super::super::{Error, Options, invalid};
use crate::analysis::*;

pub(super) fn validate(ir: &ProgramAnalysis, id: ScopeId, options: &Options) -> Result<(), Error> {
    let s = ir.scope(id);
    let info = s.loop_info.as_ref().unwrap();
    positive(&info.step, options)?;
    for expr in [&info.start, &info.end] {
        validate_expr(expr, options)?;
        for dependency in expr.loop_dependencies() {
            if dependency == id || !ir.is_within(id, dependency) {
                return Err(invalid("loop bound references an unavailable binding"));
            }
        }
    }
    if s.kind.is_parallel()
        && (!info.start.loop_dependencies().is_empty() || !info.end.loop_dependencies().is_empty())
    {
        return Err(invalid(
            "parallel grid bounds must be program-wide scalar parameters",
        ));
    }
    if let (Ok(start), Ok(end)) = (constant(&info.start, options), constant(&info.end, options))
        && (start < 0 || end <= start)
    {
        return Err(invalid("require nonempty nonnegative loop bounds"));
    }
    Ok(())
}

pub(crate) fn validate_expr(expr: &IndexExpr, options: &Options) -> Result<(), Error> {
    match expr {
        IndexExpr::LoopVar(_) => Ok(()),
        IndexExpr::Apply(op, args)
            if args.len() == 2 && ["+", "-", "*", "/", "//"].contains(&op.as_str()) =>
        {
            for arg in args {
                validate_expr(arg, options)?;
            }
            if op == "/" || op == "//" {
                positive(&args[1], options)?;
            }
            Ok(())
        }
        _ => constant(expr, options).map(|_| ()),
    }
}
