//! Trinity/backend/codegen's source conventions and benchmark ABI.
use super::plan::{Expr, ExprKind};
use super::shape::{constant, loop_range};
use super::{InitialValue, KernelPlan, TritonPlan};
use crate::analyzer::*;
use std::collections::{BTreeMap, BTreeSet};

fn tuple(values: impl IntoIterator<Item = impl ToString>) -> String {
    let values: Vec<_> = values.into_iter().map(|v| v.to_string()).collect();
    format!(
        "({}{})",
        values.join(", "),
        if values.len() == 1 { "," } else { "" }
    )
}

#[derive(Default)]
struct Writer {
    source: String,
    indent: usize,
    temp: usize,
    offset: usize,
    mask: usize,
    indices: BTreeSet<String>,
    loads: BTreeMap<String, Value>,
}
impl Writer {
    fn line(&mut self, text: impl AsRef<str>) {
        if !text.as_ref().is_empty() {
            self.source.push_str(&"    ".repeat(self.indent));
        }
        self.source.push_str(text.as_ref());
        self.source.push('\n');
    }
    fn temporary(&mut self, expression: impl AsRef<str>) -> String {
        let name = format!("temp_{}", self.temp);
        self.temp += 1;
        self.line(format!("{name} = {}", expression.as_ref()));
        name
    }
}
#[derive(Clone)]
struct Value {
    code: String,
    shape: Vec<String>,
    /// Rectangular validity, one one-dimensional predicate per logical axis.
    /// Contracting an axis removes its predicate; it never reduces mask tensors.
    valid: Vec<Option<String>>,
    /// A masked load already supplies zero. Pointwise operations may change it.
    zero_invalid: bool,
}

impl TritonPlan {
    pub fn emit(&self) -> String {
        let mut w = Writer::default();
        w.line("import triton\nimport triton.language as tl\nimport torch\n");
        for (ki, kernel) in self.kernels.iter().enumerate() {
            self.autotune(kernel, &mut w);
            w.line("@triton.jit");
            w.line(format!("def kernel_{ki}("));
            w.indent = 1;
            let mut params = Vec::new();
            for tensor in self
                .tensor_order(kernel)
                .into_iter()
                .filter(|t| kernel.tensors[t].has_global())
            {
                let name = &self.analysis.tensor(tensor).name;
                params.push(format!("{name}_ptr"));
                for axis in 0..self.kernel_shape(kernel, tensor).len() {
                    params.push(format!("{name}_stride{axis}: tl.constexpr"));
                }
            }
            for id in self.loops(kernel) {
                params.push(format!("{}: tl.constexpr", self.block(id)));
            }
            for (i, param) in params.iter().enumerate() {
                w.line(format!(
                    "{param}{}",
                    if i + 1 < params.len() { "," } else { "" }
                ));
            }
            w.indent = 0;
            w.line("):");
            w.indent = 1;
            w.temp = 0;
            w.offset = 0;
            w.indices.clear();
            w.loads.clear();
            if self.analysis.scope(kernel.root_scope).children.is_empty() {
                w.line("pass");
            }
            self.scope(kernel.root_scope, kernel, &mut w);
            w.indent = 0;
            w.line("\n");
        }
        self.wrapper(&mut w);
        w.source
    }
    fn tensor_order(&self, kernel: &KernelPlan) -> Vec<TensorId> {
        let mut ids: Vec<_> = kernel.tensors.keys().copied().collect();
        ids.sort_by_key(|id| &self.analysis.tensor(*id).name);
        ids
    }
    fn loop_name(&self, id: ScopeId) -> String {
        let s = self.analysis.scope(id);
        let info = s.loop_info.as_ref().unwrap();
        let conflict = self.analysis.scopes().iter().enumerate().any(|(i, other)| {
            i < id.index()
                && other.kernel == s.kernel
                && other.loop_info.as_ref().is_some_and(|l| {
                    l.variable == info.variable
                        && (l.step != info.step || self.analysis.is_within(id, ScopeId(i)))
                })
        });
        if conflict {
            format!("{}_{}", info.variable, id.index())
        } else {
            info.variable.clone()
        }
    }
    fn block(&self, id: ScopeId) -> String {
        format!("BLOCK_{}", self.loop_name(id).to_uppercase())
    }
    fn loops(&self, kernel: &KernelPlan) -> Vec<ScopeId> {
        let mut seen = BTreeSet::new();
        self.analysis
            .scopes()
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let id = ScopeId(i);
                if s.kernel == self.analysis.scope(kernel.root_scope).kernel
                    && s.loop_info.is_some()
                    && seen.insert(self.block(id))
                {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }
    fn tunable(&self, id: ScopeId) -> bool {
        matches!(
            self.analysis.scope(id).loop_info.as_ref().unwrap().step,
            IndexExpr::Symbol(_)
        )
    }
    fn index(&self, expr: &IndexExpr) -> String {
        match expr {
            IndexExpr::LoopVar(id) => self.loop_name(*id),
            IndexExpr::Apply(op, args) => format!(
                "({} {} {})",
                self.index(&args[0]),
                if op == "/" { "//" } else { op },
                self.index(&args[1])
            ),
            _ => constant(expr, &self.options).unwrap().to_string(),
        }
    }
    fn width(&self, dim: &IndexDim, fallback: usize) -> String {
        match dim {
            IndexDim::Tile {
                start: IndexExpr::LoopVar(id),
                width,
            } if *width == self.analysis.scope(*id).loop_info.as_ref().unwrap().step
                && !matches!(width, IndexExpr::Integer(n) if !(*n as usize).is_power_of_two()) =>
            {
                self.block(*id)
            }
            _ => fallback.to_string(),
        }
    }
    fn tile_shape(&self, id: AccessId) -> Vec<String> {
        self.analysis
            .access(id)
            .index
            .iter()
            .zip(&self.accesses[id.index()].shape)
            .map(|(dim, size)| self.width(dim, *size))
            .collect()
    }
    fn kernel_shape(&self, kernel: &KernelPlan, tensor: TensorId) -> Vec<usize> {
        let ki = self.analysis.scope(kernel.root_scope).kernel;
        self.analysis
            .accesses()
            .iter()
            .enumerate()
            .filter(|(_, a)| a.tensor == tensor && self.analysis.scope(a.scope).kernel == ki)
            .max_by_key(|(_, a)| a.index.len())
            .map(|(i, _)| self.accesses[i].axes.iter().map(|a| a.extent).collect())
            .unwrap()
    }
    fn autotune(&self, kernel: &KernelPlan, w: &mut Writer) {
        let loops: Vec<_> = self
            .loops(kernel)
            .into_iter()
            .filter(|id| self.tunable(*id))
            .collect();
        if loops.is_empty() {
            return;
        }
        let mut configs: Vec<Vec<usize>> = vec![vec![]];
        for id in &loops {
            let (start, end, _) = loop_range(&self.analysis, *id, &self.options).unwrap();
            let size = (end - start) as usize;
            let choices = if size < 16 {
                vec![1, 2, 4, 8]
            } else {
                vec![16, 32, 64, 128]
            };
            configs = configs
                .into_iter()
                .flat_map(|prefix| {
                    choices.iter().filter(|v| **v <= size).map(move |v| {
                        let mut row = prefix.clone();
                        row.push(*v);
                        row
                    })
                })
                .collect();
            configs.truncate(16);
        }
        w.line("@triton.autotune(\n    configs = [");
        for (i, config) in configs.iter().enumerate() {
            let values = loops
                .iter()
                .zip(config)
                .map(|(id, v)| format!("'{}': {v}", self.block(*id)))
                .collect::<Vec<_>>()
                .join(", ");
            w.line(format!(
                "        triton.Config({{{values}}}){}",
                if i + 1 < configs.len() { "," } else { "" }
            ));
        }
        w.line("    ], key=[]\n)");
    }
    fn initializations(&self, id: ScopeId, kernel: &KernelPlan, zero: bool, w: &mut Writer) {
        for tensor in self.tensor_order(kernel) {
            let tp = &kernel.tensors[&tensor];
            if let Some(init) = &tp.initialization
                && init.scope == id
                && (init.value == InitialValue::Zero) == zero
            {
                let name = &self.analysis.tensor(tensor).name;
                if zero {
                    w.line(format!(
                        "{name} = tl.zeros({}, dtype=tl.float32)",
                        tuple(self.tile_shape(init.access))
                    ));
                } else {
                    let value = self.load(init.access, kernel, w);
                    w.line(format!("{name} = {}", value.code));
                }
            }
        }
    }
    fn scope(&self, id: ScopeId, kernel: &KernelPlan, w: &mut Writer) {
        let scope = self.analysis.scope(id);
        let previous_indices = w.indices.clone();
        if scope.kind == ScopeKind::SequentialLoop && scope.children.is_empty() {
            w.line("# Skipped empty sloop with dummy body");
            return;
        }
        if scope.kind == ScopeKind::ParallelLoop {
            self.initializations(id, kernel, true, w);
        }
        if scope.kind != ScopeKind::Kernel {
            let (start, end, _) = loop_range(&self.analysis, id, &self.options).unwrap();
            let variable = self.loop_name(id);
            let block = self.block(id);
            if scope.kind == ScopeKind::ParallelLoop {
                let axis = kernel.parallel_loops.iter().position(|s| *s == id).unwrap();
                w.line(format!(
                    "# Parallel loop {variable} from {start} to {end} with tile size {block}"
                ));
                w.line(format!("# Executed across grid dimension {axis}"));
                w.line(format!(
                    "{variable} = {start} + tl.program_id({axis}) * {block}"
                ));
            } else {
                w.line(format!(
                    "# Sequential loop {variable} from {start} to {end} with tile size {block}"
                ));
                w.line(format!("for {variable} in range({start}, {end}, {block}):"));
                w.indent += 1;
                w.indices.clear();
                w.loads.clear();
                if scope.children.is_empty() {
                    w.line("pass");
                }
            }
        }
        if scope.kind != ScopeKind::ParallelLoop {
            self.initializations(id, kernel, true, w);
        }
        self.initializations(id, kernel, false, w);
        for item in &scope.children {
            match *item {
                ScopeItem::Scope(child) => self.scope(child, kernel, w),
                ScopeItem::Statement(statement) => {
                    let info = self.analysis.statement(statement);
                    let access = *info.accesses.last().unwrap();
                    let tensor = self.analysis.access(access).tensor;
                    let value = self.expression(&self.expressions[statement.index()], kernel, w);
                    // Stores have their own address mask. Register lanes only
                    // need neutralization when a later operation reduces them.
                    let mut code = value.code;
                    if value.shape.is_empty() {
                        code = format!("tl.full((), {code}, tl.float32)");
                    }
                    if kernel.register_accesses.contains(&access) {
                        let target_shape = self.tile_shape(access);
                        if self.expressions[statement.index()].shape
                            != self.accesses[access.index()].shape
                        {
                            code = format!("tl.broadcast_to({code}, {})", tuple(target_shape));
                        }
                        w.line(format!("{} = {code}", self.analysis.tensor(tensor).name));
                    } else {
                        self.store(access, &code, kernel, w);
                    }
                    w.loads
                        .retain(|key, _| !key.starts_with(&format!("{}:", tensor.index())));
                }
            }
        }
        for tensor in self.tensor_order(kernel) {
            let tp = &kernel.tensors[&tensor];
            if tp.export_scope == Some(id) {
                self.store(
                    tp.representative,
                    &self.analysis.tensor(tensor).name,
                    kernel,
                    w,
                );
            }
        }
        if scope.kind == ScopeKind::SequentialLoop {
            w.indent -= 1;
            w.indices = previous_indices;
            w.loads.clear();
        }
    }
    fn coordinate(&self, id: AccessId, axis: usize) -> String {
        let access = self.analysis.access(id);
        let tile = &self.accesses[id.index()];
        match &access.index[axis] {
            IndexDim::FullTile => format!("tl.arange(0, {})", tile.shape[axis]),
            IndexDim::Elem(IndexExpr::LoopVar(scope)) => format!(
                "({} // {}) + tl.arange(0, 1)",
                self.loop_name(*scope),
                self.block(*scope)
            ),
            _ => format!(
                "{} + tl.arange(0, {})",
                self.index(&tile.axes[axis].start),
                self.width(&access.index[axis], tile.shape[axis])
            ),
        }
    }
    fn slice(rank: usize, axis: usize) -> String {
        (0..rank)
            .map(|i| if i == axis { ":" } else { "None" })
            .collect::<Vec<_>>()
            .join(", ")
    }
    fn mask(&self, id: AccessId, w: &mut Writer) -> String {
        let access = self.analysis.access(id);
        let tile = &self.accesses[id.index()];
        let mut masks = Vec::new();
        for (axis, dim) in access.index.iter().enumerate() {
            let info = &tile.axes[axis];
            if matches!(dim, IndexDim::FullTile) && info.extent.is_power_of_two() {
                continue;
            }
            if matches!(dim, IndexDim::ConstTile { .. })
                && info.width.is_power_of_two()
                && constant(&info.start, &self.options)
                    .is_ok_and(|start| start >= 0 && start as usize + info.width <= info.extent)
            {
                continue;
            }
            let coordinate = if let IndexDim::Tile {
                start: IndexExpr::LoopVar(scope),
                ..
            } = dim
            {
                let width = self.width(dim, tile.shape[axis]);
                let suffix = if width == self.block(*scope) {
                    String::new()
                } else {
                    format!("_{width}")
                };
                let name = format!("{}_indices{suffix}", self.loop_name(*scope));
                if w.indices.insert(name.clone()) {
                    w.line(format!("{name} = {}", self.coordinate(id, axis)));
                }
                name
            } else if let IndexDim::Elem(IndexExpr::LoopVar(scope)) = dim {
                let name = format!("elem_{}_indices", self.loop_name(*scope));
                if w.indices.insert(name.clone()) {
                    w.line(format!("{name} = {}", self.coordinate(id, axis)));
                }
                name
            } else {
                format!("({})", self.coordinate(id, axis))
            };
            let end = if self.padded(id) {
                info.loop_end
                    .as_ref()
                    .map(|e| constant(e, &self.options).unwrap() as usize)
                    .unwrap_or(info.extent)
                    .min(info.extent)
            } else {
                info.extent
            };
            let mut condition = format!("({coordinate} < {end})");
            if !info.width.is_power_of_two() {
                condition.push_str(&format!(
                    " & (tl.arange(0, {}) < {})",
                    tile.shape[axis], info.width
                ));
            }
            if access.index.len() > 1 {
                condition = format!("({condition})[{}]", Self::slice(access.index.len(), axis));
            }
            masks.push(condition);
        }
        if masks.is_empty() {
            return "True".into();
        }
        let name = format!("mask_{}", w.mask);
        w.mask += 1;
        w.line(format!("{name} = {}", masks.join(" & ")));
        name
    }
    fn address(&self, id: AccessId, kernel: &KernelPlan, w: &mut Writer) -> (String, String) {
        let access = self.analysis.access(id);
        let tensor = &self.analysis.tensor(access.tensor).name;
        let kernel_shape = self.kernel_shape(kernel, access.tensor);
        let rank = access.index.len();
        let offset = (0..rank)
            .map(|axis| {
                let stride = if self.accesses[id.index()]
                    .axes
                    .iter()
                    .map(|a| a.extent)
                    .eq(kernel_shape.iter().copied())
                {
                    format!("{tensor}_stride{axis}")
                } else {
                    self.accesses[id.index()].axes[axis].stride.to_string()
                };
                let coordinate = self.coordinate(id, axis);
                if rank == 1 {
                    format!("({coordinate}) * {stride}")
                } else {
                    format!("({coordinate})[{}] * {stride}", Self::slice(rank, axis))
                }
            })
            .collect::<Vec<_>>()
            .join(" + ");
        let name = format!("offset_{}", w.offset);
        w.offset += 1;
        w.line(format!("{name} = {offset}"));
        let mask = self.mask(id, w);
        (name, mask)
    }
    fn padded(&self, id: AccessId) -> bool {
        (0..self.analysis.access(id).index.len()).any(|axis| self.axis_padded(id, axis))
    }
    fn axis_padded(&self, id: AccessId, axis: usize) -> bool {
        let access = self.analysis.access(id);
        let tile = &self.accesses[id.index()];
        let dim = &access.index[axis];
        let info = &tile.axes[axis];
        let width = if let IndexDim::Tile {
            start: IndexExpr::LoopVar(scope),
            width,
        } = dim
        {
            if self.tunable(*scope)
                && *width == self.analysis.scope(*scope).loop_info.as_ref().unwrap().step
            {
                128.min(loop_range(&self.analysis, *scope, &self.options).unwrap().1 as usize)
                    .next_power_of_two()
            } else {
                info.width
            }
        } else {
            info.width
        };
        !info.width.is_power_of_two()
            || !info.extent.is_multiple_of(width)
            || info.loop_end.as_ref().is_some_and(|e| {
                !(constant(e, &self.options).unwrap() as usize).is_multiple_of(width)
            })
    }
    fn validity(&self, id: AccessId) -> Vec<Option<String>> {
        let tile = &self.accesses[id.index()];
        tile.axes
            .iter()
            .enumerate()
            .map(|(axis, info)| {
                if !self.axis_padded(id, axis) {
                    return None;
                }
                let end = info
                    .loop_end
                    .as_ref()
                    .map(|e| constant(e, &self.options).unwrap() as usize)
                    .unwrap_or(info.extent)
                    .min(info.extent);
                let mut predicate = format!("({} < {end})", self.coordinate(id, axis));
                if !info.width.is_power_of_two() {
                    predicate.push_str(&format!(
                        " & (tl.arange(0, {}) < {})",
                        tile.shape[axis], info.width
                    ));
                }
                Some(predicate)
            })
            .collect()
    }
    fn load(&self, id: AccessId, kernel: &KernelPlan, w: &mut Writer) -> Value {
        let access = self.analysis.access(id);
        let (offset, mask) = self.address(id, kernel, w);
        let code = w.temporary(format!(
            "tl.load({}_ptr + {offset}, mask={mask}, other=0.0).to(tl.float32)",
            self.analysis.tensor(access.tensor).name
        ));
        Value {
            code,
            shape: self.tile_shape(id),
            valid: self.validity(id),
            zero_invalid: true,
        }
    }
    fn store(&self, id: AccessId, value: &str, kernel: &KernelPlan, w: &mut Writer) {
        let (offset, mask) = self.address(id, kernel, w);
        let value = if value.parse::<f64>().is_ok() {
            format!("tl.full((), {value}, tl.float32)")
        } else {
            format!("({value})")
        };
        w.line(format!(
            "tl.store({}_ptr + {offset}, {value}.to(tl.float16), mask={mask})",
            self.analysis.tensor(self.analysis.access(id).tensor).name
        ));
    }
    fn expression(&self, expr: &Expr, kernel: &KernelPlan, w: &mut Writer) -> Value {
        match &expr.kind {
            ExprKind::Scalar(v) => Value {
                code: v.strip_suffix(".0").unwrap_or(v).to_owned(),
                shape: vec![],
                valid: vec![],
                zero_invalid: false,
            },
            ExprKind::Index(v) => Value {
                code: self.index(v),
                shape: vec![],
                valid: vec![],
                zero_invalid: false,
            },
            ExprKind::Load(id) => {
                let a = self.analysis.access(*id);
                if kernel.register_accesses.contains(id) {
                    return Value {
                        code: self.analysis.tensor(a.tensor).name.clone(),
                        shape: self.tile_shape(*id),
                        valid: self.validity(*id),
                        zero_invalid: false,
                    };
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
                Value {
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
                Value {
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
                Value {
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
                Value {
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
                Value {
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
                let av = format!("({}).to(tl.float16)", neutralized(&a, rank - 1, "0.0"));
                let bv = format!("({}).to(tl.float16)", neutralized(&b, rank - 2, "0.0"));
                let small = [&a.shape[rank - 2], &a.shape[rank - 1], &b.shape[rank - 1]]
                    .iter()
                    .any(|s| s.parse::<usize>().is_ok_and(|v| v < 16));
                let code = if small {
                    format!(
                        "tl.sum(tl.expand_dims({av}, {rank}) * tl.expand_dims({bv}, {}), axis={})",
                        rank - 2,
                        rank - 1
                    )
                } else {
                    format!("tl.dot({av}, {bv})")
                };
                let mut shape = broadcast_shape(&a.shape[..rank - 2], &b.shape[..rank - 2]);
                shape.extend([a.shape[rank - 2].clone(), b.shape[rank - 1].clone()]);
                let mut valid = broadcast_validity(&a.valid[..rank - 2], &b.valid[..rank - 2]);
                valid.extend([a.valid[rank - 2].clone(), b.valid[rank - 1].clone()]);
                Value {
                    code,
                    shape,
                    valid,
                    zero_invalid: false,
                }
            }
        }
    }
    fn wrapper(&self, w: &mut Writer) {
        let mut tensors: Vec<_> = self.globals.iter().copied().collect();
        tensors.sort_by_key(|t| &self.analysis.tensor(*t).name);
        let names: Vec<_> = tensors
            .iter()
            .map(|t| self.analysis.tensor(*t).name.clone())
            .collect();
        let blocks: BTreeSet<_> = self
            .kernels
            .iter()
            .flat_map(|k| self.loops(k))
            .filter(|id| self.tunable(*id))
            .map(|id| format!("block_{}", self.loop_name(id)))
            .collect();
        w.line("# Metadata for benchmark.py");
        w.line(format!(
            "TENSOR_PARAMS = [{}]",
            names
                .iter()
                .map(|s| format!("'{s}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        w.line(format!(
            "BLOCK_PARAMS = [{}]\n",
            blocks
                .iter()
                .map(|s| format!("'{s}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        let mut params = names;
        params.extend(blocks.iter().map(|s| format!("{s}=16")));
        w.line(format!("def forward({}):", params.join(", ")));
        w.indent = 1;
        w.line("\"\"\"Wrapper function that executes all kernels sequentially.\"\"\"");
        for (ki, kernel) in self.kernels.iter().enumerate() {
            let mut arguments = Vec::new();
            for tensor in self
                .tensor_order(kernel)
                .into_iter()
                .filter(|t| kernel.tensors[t].has_global())
            {
                let name = &self.analysis.tensor(tensor).name;
                let shape = self.kernel_shape(kernel, tensor);
                let arg = if shape != self.options.shapes[name] {
                    let arg = format!("{name}_view_{ki}");
                    w.line(format!("{arg} = {name}.view{}", tuple(&shape)));
                    arg
                } else {
                    name.clone()
                };
                arguments.push(arg.clone());
                for axis in 0..shape.len() {
                    arguments.push(format!("{arg}.stride({axis})"));
                }
            }
            let dynamic = kernel.parallel_loops.iter().any(|id| self.tunable(*id));
            let grid: Vec<_> = kernel
                .parallel_loops
                .iter()
                .map(|id| {
                    let (start, end, step) =
                        loop_range(&self.analysis, *id, &self.options).unwrap();
                    let block = if self.tunable(*id) {
                        format!("meta[\"{}\"]", self.block(*id))
                    } else {
                        step.to_string()
                    };
                    format!("({end} - {start} + {block} - 1) // {block}")
                })
                .collect();
            let grid = if grid.is_empty() {
                "(1,)".into()
            } else {
                tuple(grid)
            };
            w.line(format!(
                "kernel_{ki}[{}{grid}](",
                if dynamic { "lambda meta: " } else { "" }
            ));
            w.indent += 1;
            for arg in arguments {
                w.line(format!("{arg},"));
            }
            for id in self.loops(kernel) {
                let block = self.block(id);
                if self.tunable(id) {
                    w.line(format!("# {block} is automatically set by autotune"));
                } else {
                    w.line(format!(
                        "{block}={},",
                        loop_range(&self.analysis, id, &self.options).unwrap().2
                    ));
                }
            }
            w.indent -= 1;
            w.line(")\n");
        }
        w.line("# Return output tensors if needed");
        w.line("# This depends on your specific use case");
        w.line("pass");
        w.indent = 0;
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

fn neutralized(value: &Value, axis: usize, identity: &str) -> String {
    if identity == "0.0" && value.zero_invalid {
        return value.code.clone();
    }
    match &value.valid[axis] {
        Some(mask) => {
            let mask = if value.shape.len() > 1 {
                format!("({mask})[{}]", TritonPlan::slice(value.shape.len(), axis))
            } else {
                mask.clone()
            };
            format!("tl.where({mask}, {}, {identity})", value.code)
        }
        None => value.code.clone(),
    }
}
