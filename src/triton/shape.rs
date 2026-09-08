use super::{Error, Options, invalid};
use crate::analyzer::{AccessInfo, IndexDim, IndexExpr, ProgramAnalysis, ScopeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisAccess {
    pub start: IndexExpr,
    pub width: usize,
    pub extent: usize,
    pub stride: usize,
    /// For a tile of a loop, mask against its IR end as well as tensor extent.
    pub loop_end: Option<IndexExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileAccess {
    pub axes: Vec<AxisAccess>,
    pub shape: Vec<usize>,
}

pub(crate) fn constant(expr: &IndexExpr, options: &Options) -> Result<i64, Error> {
    match expr {
        IndexExpr::Integer(v) => Ok(*v),
        IndexExpr::Symbol(name) => options
            .symbols
            .get(name)
            .copied()
            .ok_or_else(|| invalid(format!("missing symbol {name}"))),
        IndexExpr::Apply(op, args) if args.len() == 2 => {
            let a = constant(&args[0], options)?;
            let b = constant(&args[1], options)?;
            let value = match op.as_str() {
                "+" => a.checked_add(b),
                "-" => a.checked_sub(b),
                "*" => a.checked_mul(b),
                "/" | "//" if b > 0 => Some(a.div_euclid(b)),
                _ => None,
            };
            value.ok_or_else(|| invalid(format!("invalid constant expression {expr:?}")))
        }
        _ => Err(invalid(format!(
            "expected compile-time integer, got {expr:?}"
        ))),
    }
}

pub(crate) fn positive(expr: &IndexExpr, options: &Options) -> Result<usize, Error> {
    let v = constant(expr, options)?;
    if v <= 0 {
        return Err(invalid(format!("expected positive extent, got {v}")));
    }
    usize::try_from(v).map_err(|_| invalid("extent overflow"))
}

pub(crate) fn tile(
    a: &AccessInfo,
    ir: &ProgramAnalysis,
    options: &Options,
) -> Result<TileAccess, Error> {
    let name = &ir.tensor(a.tensor).name;
    let base_shape = options
        .shapes
        .get(name)
        .ok_or_else(|| invalid(format!("missing shape for {name}")))?;
    let view_shape = a
        .view_shape
        .as_ref()
        .map(|shape| {
            shape
                .iter()
                .map(|s| positive(s, options))
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
                (start.clone(), positive(width, options)?, end)
            }
            IndexDim::ConstTile { start, width } => {
                (start.clone(), positive(width, options)?, None)
            }
        };
        validate_index(&start, options)?;
        axes.push(AxisAccess {
            start,
            width,
            extent: shape[i],
            stride: product(&shape[i + 1..])?,
            loop_end,
        });
    }
    let shape: Vec<_> = axes
        .iter()
        .map(|a| {
            a.width
                .checked_next_power_of_two()
                .ok_or_else(|| invalid("tile overflow"))
        })
        .collect::<Result<_, _>>()?;
    // Avoid emitting a block that Triton cannot represent, before allocating source.
    if product(&shape)? > 1_048_576 {
        return Err(invalid(format!(
            "{name}: tile exceeds Triton's element limit"
        )));
    }
    Ok(TileAccess { axes, shape })
}

pub(crate) fn product(shape: &[usize]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b)
            .filter(|n| *n <= i64::MAX as usize)
            .ok_or_else(|| invalid("shape product overflow"))
    })
}

fn validate_index(expr: &IndexExpr, options: &Options) -> Result<(), Error> {
    match expr {
        IndexExpr::LoopVar(_) => Ok(()),
        IndexExpr::Apply(op, args)
            if args.len() == 2 && ["+", "-", "*", "/", "//"].contains(&op.as_str()) =>
        {
            for arg in args {
                validate_index(arg, options)?;
            }
            if ["/", "//"].contains(&op.as_str()) {
                positive(&args[1], options)?;
            }
            Ok(())
        }
        _ => constant(expr, options).map(|_| ()),
    }
}

pub(crate) fn loop_range(
    ir: &ProgramAnalysis,
    id: ScopeId,
    options: &Options,
) -> Result<(i64, i64, usize), Error> {
    let info = ir.scope(id).loop_info.as_ref().unwrap();
    let start = constant(&info.start, options)?;
    let end = constant(&info.end, options)?;
    let step = positive(&info.step, options)?;
    if start < 0 || end <= start {
        return Err(invalid(format!(
            "{}: require nonempty nonnegative static loop bounds",
            info.variable
        )));
    }
    Ok((start, end, step))
}
