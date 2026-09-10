//! Scalar, pointwise, reduction, transform and dot expressions with value validity.
use super::super::plan::{Expr, ExprKind};
use super::super::{KernelPlan, ProgramPlan};
use super::context::{CodegenContext, EmittedValue, tuple};

impl ProgramPlan {
    pub(super) fn expression(
        &self,
        expr: &Expr,
        kernel: &KernelPlan,
        w: &mut CodegenContext,
    ) -> EmittedValue {
        match &expr.kind {
            ExprKind::Scalar(v) => EmittedValue {
                code: v.strip_suffix(".0").unwrap_or(v).to_owned(),
                shape: vec![],
                valid: vec![],
                zero_invalid: false,
            },
            ExprKind::Index(v) => EmittedValue {
                code: self.index(v),
                shape: vec![],
                valid: vec![],
                zero_invalid: false,
            },
            ExprKind::Load(id) => {
                let a = self.analysis.access(*id);
                if kernel.register_accesses.contains(id) {
                    return self.local_load(*id, kernel, w);
                }
                if self.options.managed {
                    return self.load(*id, kernel, w);
                }
                let key = format!("{}:{:?}:{:?}", a.tensor.index(), a.index, a.view_shape);
                if let Some(value) = w.loads.get(&key) {
                    return value.clone();
                }
                let value = self.load(*id, kernel, w);
                w.loads.insert(key, value.clone());
                value
            }
            ExprKind::Unary(op, child) => {
                let a = self.expression(child, kernel, w);
                let cast = if a.shape.is_empty() {
                    format!("tl.full((), {}, tl.float32)", a.code)
                } else {
                    format!("({}).to(tl.float32)", a.code)
                };
                let code = match op.as_str() {
                    "sqr" => format!("({0} * {0})", a.code),
                    "abs" => format!("tl.abs({})", a.code),
                    "erf" => format!("tl.math.erf({cast})"),
                    _ => format!("tl.{op}({cast})"),
                };
                let zero_invalid =
                    a.zero_invalid && matches!(op.as_str(), "sqr" | "abs" | "sqrt" | "erf");
                EmittedValue {
                    code,
                    zero_invalid,
                    ..a
                }
            }
            ExprKind::Binary(op, left, right) => {
                let a = self.expression(left, kernel, w);
                let b = self.expression(right, kernel, w);
                let code = match op.as_str() {
                    "max" => format!("tl.maximum({}, {})", a.code, b.code),
                    "min" => format!("tl.minimum({}, {})", a.code, b.code),
                    _ => format!("({} {op} {})", a.code, b.code),
                };
                let shape = broadcast_shape(&a.shape, &b.shape);
                let zero_invalid = a.zero_invalid
                    && b.zero_invalid
                    && a.valid == b.valid
                    && a.shape == b.shape
                    && matches!(op.as_str(), "+" | "-" | "*");
                let valid = broadcast_validity(&a.valid, &b.valid);
                EmittedValue {
                    code,
                    shape,
                    valid,
                    zero_invalid,
                }
            }
            ExprKind::Reduce(op, axis, child) => {
                let a = self.expression(child, kernel, w);
                let mut shape = a.shape.clone();
                shape.remove(*axis);
                let (function, identity) = match op.as_str() {
                    "rsum" => ("sum", "0.0"),
                    "rmax" => ("max", "float('-inf')"),
                    _ => ("min", "float('inf')"),
                };
                let code = neutralized(&a, *axis, identity);
                let code = format!(
                    "tl.{function}({code}, axis={axis}{})",
                    if op == "rsum" {
                        ", dtype=tl.float32"
                    } else {
                        ""
                    }
                );
                let mut valid = a.valid;
                valid.remove(*axis);
                EmittedValue {
                    code,
                    shape,
                    valid,
                    zero_invalid: op == "rsum" && a.zero_invalid,
                }
            }
            ExprKind::Permute(order, child) => {
                let a = self.expression(child, kernel, w);
                let shape = order.iter().map(|i| a.shape[*i].clone()).collect();
                let code = w.temporary(format!("tl.permute({}, {})", a.code, tuple(order)));
                let valid = order.iter().map(|i| a.valid[*i].clone()).collect();
                EmittedValue {
                    code,
                    shape,
                    valid,
                    zero_invalid: a.zero_invalid,
                }
            }
            ExprKind::Transform(op, axis, child) => {
                let a = self.expression(child, kernel, w);
                let mut shape = a.shape.clone();
                if op == "squeeze" {
                    shape.remove(*axis);
                } else {
                    shape.insert(*axis, "1".into());
                }
                let code = if op == "bcast" {
                    let slice = (0..shape.len())
                        .map(|i| if i == *axis { "None" } else { ":" })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({})[{slice}]", a.code)
                } else if op == "unsqueeze" {
                    w.temporary(format!("tl.expand_dims({}, {axis})", a.code))
                } else {
                    w.temporary(format!("tl.reshape({}, {})", a.code, tuple(&shape)))
                };
                let mut valid = a.valid;
                if op == "squeeze" {
                    // A singleton can be masked (e.g. elem of a padded batch).
                    // Keep that scalar condition on every remaining axis.
                    let removed = valid.remove(*axis);
                    if let Some(predicate) = removed {
                        for mask in &mut valid {
                            *mask = merge_masks(mask.take(), Some(predicate.clone()));
                        }
                    }
                } else {
                    valid.insert(*axis, None);
                }
                EmittedValue {
                    code,
                    shape,
                    valid,
                    zero_invalid: a.zero_invalid,
                }
            }
            ExprKind::Dot(left, right) => {
                let a = self.expression(left, kernel, w);
                let b = self.expression(right, kernel, w);
                let rank = a.shape.len();
                let small = [&a.shape[rank - 2], &a.shape[rank - 1], &b.shape[rank - 1]]
                    .iter()
                    .any(|s| s.parse::<usize>().is_ok_and(|v| v < 16));
                let dtype = if small && (self.options.managed || rank > 3) {
                    "float32"
                } else {
                    "float16"
                };
                let av = format!("({}).to(tl.{dtype})", neutralized(&a, rank - 1, "0.0"));
                let bv = format!("({}).to(tl.{dtype})", neutralized(&b, rank - 2, "0.0"));
                let batch = broadcast_shape(&a.shape[..rank - 2], &b.shape[..rank - 2]);
                let mut code = if small {
                    format!(
                        "tl.sum(tl.expand_dims({av}, {rank}) * tl.expand_dims({bv}, {}), axis={})",
                        rank - 2,
                        rank - 1
                    )
                } else if rank > 3 {
                    let mut ashape = batch.clone();
                    ashape.extend_from_slice(&a.shape[rank - 2..]);
                    let mut bshape = batch.clone();
                    bshape.extend_from_slice(&b.shape[rank - 2..]);
                    let batches = batch.join(" * ");
                    let mut output = batch.clone();
                    output.extend([a.shape[rank - 2].clone(), b.shape[rank - 1].clone()]);
                    format!(
                        "tl.reshape(tl.dot(tl.reshape(tl.broadcast_to({av}, {}), ({batches}, {}, {})), tl.reshape(tl.broadcast_to({bv}, {}), ({batches}, {}, {}))), {})",
                        tuple(ashape),
                        a.shape[rank - 2],
                        a.shape[rank - 1],
                        tuple(bshape),
                        b.shape[rank - 2],
                        b.shape[rank - 1],
                        tuple(output)
                    )
                } else {
                    format!("tl.dot({av}, {bv})")
                };
                if self.options.managed
                    && !small
                    && [&a.shape[rank - 2], &a.shape[rank - 1], &b.shape[rank - 1]]
                        .iter()
                        .any(|s| s.parse::<usize>().is_err())
                {
                    // The chosen profile can cross Triton's dot-size boundary.
                    // Dispatch on constexpr dimensions, not the validation sample.
                    let av = format!("({}).to(tl.float32)", neutralized(&a, rank - 1, "0.0"));
                    let bv = format!("({}).to(tl.float32)", neutralized(&b, rank - 2, "0.0"));
                    let small_code = format!(
                        "tl.sum(tl.expand_dims({av}, {rank}) * tl.expand_dims({bv}, {}), axis={})",
                        rank - 2,
                        rank - 1
                    );
                    let name = format!("temp_{}", w.temp);
                    w.temp += 1;
                    w.line(format!(
                        "if {} < 16 or {} < 16 or {} < 16:",
                        a.shape[rank - 2],
                        a.shape[rank - 1],
                        b.shape[rank - 1]
                    ));
                    w.indent += 1;
                    w.line(format!("{name} = {small_code}"));
                    w.indent -= 1;
                    w.line("else:");
                    w.indent += 1;
                    w.line(format!("{name} = {code}"));
                    w.indent -= 1;
                    code = name;
                }
                let mut shape = broadcast_shape(&a.shape[..rank - 2], &b.shape[..rank - 2]);
                shape.extend([a.shape[rank - 2].clone(), b.shape[rank - 1].clone()]);
                let mut valid = broadcast_validity(&a.valid[..rank - 2], &b.valid[..rank - 2]);
                valid.extend([a.valid[rank - 2].clone(), b.valid[rank - 1].clone()]);
                EmittedValue {
                    code,
                    shape,
                    valid,
                    zero_invalid: false,
                }
            }
            ExprKind::Cast(dtype, child) => {
                let mut value = self.expression(child, kernel, w);
                value.code = if value.shape.is_empty() {
                    format!("tl.full((), {}, tl.{dtype})", value.code)
                } else {
                    format!("({}).to(tl.{dtype})", value.code)
                };
                value
            }
            ExprKind::Concat(axis, left, right) => {
                let a = self.expression(left, kernel, w);
                let b = self.expression(right, kernel, w);
                let mut shape = a.shape.clone();
                let aw = self.logical_extent(left, *axis);
                let bw = self.logical_extent(right, *axis);
                shape[*axis] = format!("triton.next_power_of_2(({aw}) + ({bw}))");
                let coordinate = format!("tl.arange(0, {})", shape[*axis]);
                let ai = format!("tl.minimum({coordinate}, {} - 1)", a.shape[*axis]);
                let bi = format!(
                    "tl.minimum(tl.maximum({coordinate} - ({aw}), 0), {} - 1)",
                    b.shape[*axis]
                );
                let expand = |index: &str| {
                    format!(
                        "tl.broadcast_to(({index})[{}], {})",
                        Self::slice(shape.len(), *axis),
                        tuple(&shape)
                    )
                };
                let ga = w.temporary(format!(
                    "tl.gather({}, {}, axis={axis})",
                    a.code,
                    expand(&ai)
                ));
                let gb = w.temporary(format!(
                    "tl.gather({}, {}, axis={axis})",
                    b.code,
                    expand(&bi)
                ));
                let mut valid = broadcast_validity(&a.valid, &b.valid);
                let av = a.valid[*axis]
                    .as_ref()
                    .map(|p| format!("tl.gather({p}, {ai}, axis=0)"))
                    .unwrap_or("True".into());
                let bv = b.valid[*axis]
                    .as_ref()
                    .map(|p| format!("tl.gather({p}, {bi}, axis=0)"))
                    .unwrap_or("True".into());
                valid[*axis] = Some(format!(
                    "({coordinate} < ({aw}) + ({bw})) & tl.where({coordinate} < ({aw}), {av}, {bv})"
                ));
                EmittedValue {
                    code: format!(
                        "tl.where(({coordinate} < ({aw}))[{}], {ga}, {gb})",
                        Self::slice(shape.len(), *axis)
                    ),
                    shape,
                    valid,
                    zero_invalid: false,
                }
            }
        }
    }

    fn logical_extent(&self, expr: &Expr, axis: usize) -> String {
        match &expr.kind {
            ExprKind::Load(id) => match &self.analysis.access(*id).index[axis] {
                crate::analysis::IndexDim::FullTile => self.view_shape(*id)[axis].clone(),
                crate::analysis::IndexDim::Elem(_) => "1".into(),
                crate::analysis::IndexDim::Tile { width, .. }
                | crate::analysis::IndexDim::ConstTile { width, .. } => self.index(width),
            },
            ExprKind::Unary(_, x) | ExprKind::Cast(_, x) => self.logical_extent(x, axis),
            ExprKind::Permute(order, x) => self.logical_extent(x, order[axis]),
            ExprKind::Transform(op, ax, x) if op == "squeeze" => {
                self.logical_extent(x, axis + usize::from(axis >= *ax))
            }
            ExprKind::Transform(_, ax, x) => {
                if axis == *ax {
                    "1".into()
                } else {
                    self.logical_extent(x, axis - usize::from(axis > *ax))
                }
            }
            ExprKind::Reduce(_, ax, x) => self.logical_extent(x, axis + usize::from(axis >= *ax)),
            ExprKind::Concat(ax, a, b) if axis == *ax => format!(
                "({} + {})",
                self.logical_extent(a, axis),
                self.logical_extent(b, axis)
            ),
            ExprKind::Concat(_, a, _) => self.logical_extent(a, axis),
            ExprKind::Binary(_, a, b) => {
                let width = |x: &Expr| {
                    if axis + x.shape.len() < expr.shape.len() {
                        "1".into()
                    } else {
                        self.logical_extent(x, axis + x.shape.len() - expr.shape.len())
                    }
                };
                let a = width(a);
                let b = width(b);
                if a == "1" { b } else { a }
            }
            ExprKind::Dot(a, b) => {
                let n = expr.shape.len();
                if axis == n - 1 {
                    self.logical_extent(b, axis)
                } else {
                    let av = self.logical_extent(a, axis);
                    if axis < n - 2 && av == "1" {
                        self.logical_extent(b, axis)
                    } else {
                        av
                    }
                }
            }
            _ => expr.shape[axis].to_string(),
        }
    }
}

fn broadcast_shape(a: &[String], b: &[String]) -> Vec<String> {
    let rank = a.len().max(b.len());
    (0..rank)
        .map(|i| {
            let x = a
                .get(i.wrapping_sub(rank - a.len()))
                .map(String::as_str)
                .unwrap_or("1");
            let y = b
                .get(i.wrapping_sub(rank - b.len()))
                .map(String::as_str)
                .unwrap_or("1");
            if x == "1" { y.to_owned() } else { x.to_owned() }
        })
        .collect()
}

fn merge_masks(a: Option<String>, b: Option<String>) -> Option<String> {
    match (a, b) {
        (Some(a), Some(b)) => Some(format!("({a}) & ({b})")),
        (a, None) | (None, a) => a,
    }
}

fn broadcast_validity(a: &[Option<String>], b: &[Option<String>]) -> Vec<Option<String>> {
    let rank = a.len().max(b.len());
    (0..rank)
        .map(|i| {
            merge_masks(
                a.get(i.wrapping_sub(rank - a.len())).cloned().flatten(),
                b.get(i.wrapping_sub(rank - b.len())).cloned().flatten(),
            )
        })
        .collect()
}

fn neutralized(value: &EmittedValue, axis: usize, identity: &str) -> String {
    if identity == "0.0" && value.zero_invalid {
        return value.code.clone();
    }
    match &value.valid[axis] {
        Some(mask) => {
            let mask = if value.shape.len() > 1 {
                format!("({mask})[{}]", ProgramPlan::slice(value.shape.len(), axis))
            } else {
                mask.clone()
            };
            format!("tl.where({mask}, {}, {identity})", value.code)
        }
        None => value.code.clone(),
    }
}
