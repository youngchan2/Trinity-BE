//! Validate scalar loop expressions separately from static tile extents.
use super::scalar::{constant, positive};
use super::*;

pub fn validate(ir: &ScheduledIr, id: ScopeId, bindings: &Bindings) -> Result<(), ResolveError> {
    let s = ir.scope(id);
    let info = s.loop_info.as_ref().unwrap();
    positive(&info.step, &bindings.symbols)?;
    for expr in [&info.start, &info.end] {
        validate_expr(expr, bindings)?;
        for dependency in expr.loop_dependencies() {
            if dependency == id || !ir.is_within(id, dependency) {
                return Err(invalid("loop bound references an unavailable binding"));
            }
        }
    }
    if let (Ok(start), Ok(end)) = (
        constant(&info.start, &bindings.symbols),
        constant(&info.end, &bindings.symbols),
    ) && (start < 0 || end <= start)
    {
        return Err(invalid("require nonempty nonnegative loop bounds"));
    }
    Ok(())
}

pub fn validate_expr(expr: &IndexExpr, bindings: &Bindings) -> Result<(), ResolveError> {
    match expr {
        IndexExpr::LoopVar(_) => Ok(()),
        IndexExpr::Apply(op, args)
            if args.len() == 2 && ["+", "-", "*", "/", "//"].contains(&op.as_str()) =>
        {
            for arg in args {
                validate_expr(arg, bindings)?;
            }
            if op == "/" || op == "//" {
                positive(&args[1], &bindings.symbols)?;
            }
            Ok(())
        }
        _ => constant(expr, &bindings.symbols).map(|_| ()),
    }
}
