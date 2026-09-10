//! Reader for the owned, scheduled Loop IR boundary.
use crate::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct LoopIrConfig {
    pub target: TargetCapability,
    pub world_size: usize,
    pub symbols: BTreeMap<String, i64>,
    pub dtypes: BTreeMap<String, DType>,
}
impl Default for LoopIrConfig {
    fn default() -> Self {
        Self {
            target: TargetCapability::Cuda(CudaTargetCapability::Hopper),
            world_size: 1,
            symbols: BTreeMap::new(),
            dtypes: BTreeMap::new(),
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("Loop IR at byte {offset}: {message}")]
pub struct LoopIrError {
    pub offset: usize,
    pub message: String,
}
fn error(offset: usize, message: impl Into<String>) -> LoopIrError {
    LoopIrError {
        offset,
        message: message.into(),
    }
}

#[derive(Clone)]
struct Node {
    expr: Expression,
    offset: usize,
    children: Vec<Node>,
}
impl Node {
    fn op(&self) -> &str {
        self.expr.operator().unwrap_or("")
    }
    fn atom(&self) -> Result<&str, LoopIrError> {
        self.expr
            .atom()
            .ok_or_else(|| error(self.offset, "expected atom"))
    }
    fn args(&self, count: usize) -> Result<&[Node], LoopIrError> {
        if self.children.len() != count + 1 {
            return Err(error(
                self.offset,
                format!("{} expects {count} arguments", self.op()),
            ));
        }
        Ok(&self.children[1..])
    }
}
fn parse(text: &str) -> Result<Node, LoopIrError> {
    fn one(text: &str, cursor: &mut usize) -> Result<Node, LoopIrError> {
        while text
            .as_bytes()
            .get(*cursor)
            .is_some_and(u8::is_ascii_whitespace)
        {
            *cursor += 1;
        }
        let offset = *cursor;
        match text.as_bytes().get(*cursor) {
            Some(b'(') => {
                *cursor += 1;
                let mut children = Vec::new();
                loop {
                    while text
                        .as_bytes()
                        .get(*cursor)
                        .is_some_and(u8::is_ascii_whitespace)
                    {
                        *cursor += 1;
                    }
                    if text.as_bytes().get(*cursor) == Some(&b')') {
                        *cursor += 1;
                        break;
                    }
                    if *cursor == text.len() {
                        return Err(error(offset, "unclosed expression"));
                    }
                    children.push(one(text, cursor)?);
                }
                if children.is_empty() {
                    return Err(error(offset, "empty expression"));
                }
                Ok(Node {
                    expr: Expression::List(children.iter().map(|n| n.expr.clone()).collect()),
                    children,
                    offset,
                })
            }
            Some(b')') | None => Err(error(offset, "expected expression")),
            Some(_) => {
                while text
                    .as_bytes()
                    .get(*cursor)
                    .is_some_and(|b| !b.is_ascii_whitespace() && *b != b'(' && *b != b')')
                {
                    *cursor += 1;
                }
                Ok(Node {
                    expr: Expression::Atom(text[offset..*cursor].into()),
                    children: Vec::new(),
                    offset,
                })
            }
        }
    }
    let mut cursor = 0;
    let node = one(text, &mut cursor)?;
    if !text[cursor..].trim().is_empty() {
        return Err(error(cursor, "trailing input"));
    }
    Ok(node)
}

struct Reader<'a> {
    config: &'a LoopIrConfig,
    builder: PhysicalPlanBuilder,
    tensors: BTreeMap<String, (ValueInstanceId, String, Vec<usize>)>,
    output: Option<(String, ValueInstanceId)>,
    positions: Vec<usize>,
}
impl Reader<'_> {
    fn gemm_instance(
        &self,
        e: &Expression,
        offset: usize,
    ) -> Result<ImplementationInstance, LoopIrError> {
        fn find(e: &Expression) -> Option<&Expression> {
            if e.operator() == Some("@") {
                Some(e)
            } else {
                e.list()?.iter().find_map(find)
            }
        }
        fn tile(
            view: &Expression,
            index: &Expression,
            config: &LoopIrConfig,
        ) -> Option<(DType, Vec<usize>)> {
            let view = view.list()?;
            let name = view.get(1)?.list()?.get(1)?.atom()?;
            let dtype = *config.dtypes.get(name)?;
            let layout = view.get(2)?.list()?;
            let slots = index.list()?;
            let mut shape = Vec::new();
            for axis in &layout[1..] {
                let axis = axis.list()?;
                let name = axis.get(1)?.atom()?;
                let size = axis.get(2)?.atom()?.parse::<usize>().ok()?;
                let part = slots.iter().skip(1).find_map(|s| {
                    let s = s.list()?;
                    (s.get(1)?.atom() == Some(name)).then(|| s.get(2)).flatten()
                });
                shape.push(match part {
                    None => size,
                    Some(p) if p.atom() == Some("fulltile") => size,
                    Some(p) if p.operator() == Some("elem") => 1,
                    Some(p) if p.operator() == Some("tile") => {
                        p.list()?.get(2)?.atom()?.parse().ok()?
                    }
                    _ => return None,
                });
            }
            Some((dtype, shape))
        }
        let failure = || {
            error(
                offset,
                "no implementation supports the scheduled GEMM tiles/dtypes",
            )
        };
        let m = find(e).and_then(Expression::list).ok_or_else(failure)?;
        let a = m
            .get(1)
            .and_then(Expression::list)
            .filter(|a| a.len() == 3 && a[0].atom() == Some("load"))
            .ok_or_else(failure)?;
        let b = m
            .get(2)
            .and_then(Expression::list)
            .filter(|a| a.len() == 3 && a[0].atom() == Some("load"))
            .ok_or_else(failure)?;
        let st = e.list().ok_or_else(failure)?;
        let (ad, ashape) = tile(&a[1], &a[2], self.config).ok_or_else(failure)?;
        let (bd, bshape) = tile(&b[1], &b[2], self.config).ok_or_else(failure)?;
        let (cd, cshape) = tile(&st[1], &st[3], self.config).ok_or_else(failure)?;
        gemm_implementations(self.config.target)
            .iter()
            .flat_map(|i| i.enumerate_scheduled([ad, bd, cd], [&ashape, &bshape, &cshape]))
            .next()
            .ok_or_else(failure)
    }

    fn number(&self, n: &Node) -> Result<i64, LoopIrError> {
        let atom = n.atom()?;
        atom.parse()
            .ok()
            .or_else(|| self.config.symbols.get(atom).copied())
            .ok_or_else(|| error(n.offset, format!("unresolved symbol {atom}")))
    }
    fn index(&self, n: &Node, scope: &BTreeSet<String>) -> Result<IndexExpr, LoopIrError> {
        if let Some(a) = n.expr.atom() {
            return if scope.contains(a) {
                Ok(IndexExpr::Variable(a.into()))
            } else {
                self.number(n).map(IndexExpr::Constant)
            };
        }
        let args = n.args(2)?;
        let a = Box::new(self.index(&args[0], scope)?);
        let b = Box::new(self.index(&args[1], scope)?);
        match n.op() {
            "+" => Ok(IndexExpr::Add(a, b)),
            "-" => Ok(IndexExpr::Sub(a, b)),
            "*" => Ok(IndexExpr::Mul(a, b)),
            "/" => Ok(IndexExpr::Div(a, b)),
            _ => Err(error(n.offset, "unsupported range expression")),
        }
    }
    fn collect(&mut self, n: &Node) -> Result<(), LoopIrError> {
        if n.op() == "view" {
            let args = n.args(2)?;
            let base = args[0].args(1)?;
            let role = args[0].op();
            if !matches!(role, "input" | "tensor" | "output") {
                return Err(error(n.offset, "unsupported view base"));
            }
            let name = base[0].atom()?.to_owned();
            if args[1].op() != "layout" {
                return Err(error(n.offset, "expected layout"));
            }
            let mut shape = Vec::new();
            let mut axes = BTreeSet::new();
            for axis in &args[1].children[1..] {
                if axis.op() != "axis" {
                    return Err(error(axis.offset, "expected axis"));
                }
                let a = axis.args(2)?;
                if !axes.insert(a[0].atom()?) {
                    return Err(error(axis.offset, "duplicate layout axis"));
                }
                let size = self.number(&a[1])?;
                if size <= 0 {
                    return Err(error(axis.offset, "nonpositive tensor extent"));
                }
                shape.push(size as usize);
            }
            if !(1..=3).contains(&shape.len()) {
                return Err(error(n.offset, "supported tensor ranks are 1, 2, 3"));
            }
            if let Some((_, old_role, old_shape)) = self.tensors.get(&name) {
                if old_role != role || old_shape != &shape {
                    return Err(error(n.offset, format!("inconsistent view for {name}")));
                }
            } else {
                let dtype = *self
                    .config
                    .dtypes
                    .get(&name)
                    .ok_or_else(|| error(n.offset, format!("missing dtype for {name}")))?;
                let storage = if role == "tensor" {
                    Storage::Global
                } else {
                    Storage::External
                };
                let id = self
                    .builder
                    .add_named_value(&name, dtype, shape.iter().copied(), storage);
                if role == "input" {
                    self.builder.bind_input(&name, id);
                }
                if role == "output" {
                    if self.output.is_some() {
                        return Err(error(n.offset, "one output tensor is supported"));
                    }
                    self.output = Some((name.clone(), id));
                }
                self.tensors.insert(name, (id, role.into(), shape));
            }
        }
        for child in &n.children {
            self.collect(child)?;
        }
        Ok(())
    }
    fn notation(&self, n: &Node, scope: &BTreeSet<String>) -> Result<Expression, LoopIrError> {
        if n.expr.atom().is_some() {
            return Ok(Expression::Atom(self.number(n)?.to_string()));
        }
        let op = n.op();
        let count = match op {
            "store" => 3,
            "load" | "view" | "axis" | "slot" | "tile" | "@" | "+" | "-" | "*" | "/" | "rsum"
            | "bcast" | "unsqueeze" => 2,
            "input" | "tensor" | "output" | "elem" | "sqr" | "sqrt" | "sigmoid" => 1,
            "layout" | "keyed_index" => n.children.len() - 1,
            _ => return Err(error(n.offset, format!("unsupported operation {op}"))),
        };
        let args = n.args(count)?;
        let mut result = vec![Expression::Atom(op.into())];
        for (i, arg) in args.iter().enumerate() {
            let literal = matches!(op, "input" | "tensor" | "output")
                || (matches!(op, "axis" | "slot" | "tile" | "elem") && i == 0)
                || arg.expr.atom() == Some("fulltile");
            if literal {
                let a = arg.atom()?;
                if matches!(op, "tile" | "elem") && !scope.contains(a) {
                    return Err(error(arg.offset, format!("unbound index {a}")));
                }
                result.push(arg.expr.clone());
            } else {
                result.push(self.notation(arg, scope)?);
            }
        }
        Ok(Expression::List(result))
    }
    fn statements(
        &mut self,
        n: &Node,
        scope: &BTreeSet<String>,
        serial: bool,
    ) -> Result<Vec<Statement>, LoopIrError> {
        match n.op() {
            "seq" => {
                let mut out = Vec::new();
                for child in n.args(2)? {
                    out.extend(self.statements(child, scope, serial)?);
                }
                Ok(out)
            }
            "ploop" | "sloop" | "mloop" => {
                let split = n.op() == "mloop";
                let args = n.args(if split { 7 } else { 5 })?;
                let parallel = n.op() != "sloop";
                if serial && parallel {
                    return Err(error(
                        n.offset,
                        "parallel work inside a sequential loop is unsupported",
                    ));
                }
                let variable = args[3].atom()?.to_owned();
                let domain = LoopDomain {
                    variable: variable.clone(),
                    start: self.index(&args[0], scope)?,
                    stop: self.index(&args[1], scope)?,
                    step: self.index(&args[2], scope)?,
                };
                let mut child_scope = scope.clone();
                child_scope.insert(variable);
                if split {
                    let split_var = args[4].atom()?.to_owned();
                    if child_scope.contains(&split_var) {
                        return Err(error(
                            args[4].offset,
                            "split variable shadows an active loop",
                        ));
                    }
                    let count = self.number(&args[5])?;
                    let empty = BTreeMap::new();
                    let start = domain
                        .start
                        .evaluate(&empty)
                        .map_err(|e| error(n.offset, e))?;
                    let stop = domain
                        .stop
                        .evaluate(&empty)
                        .map_err(|e| error(n.offset, e))?;
                    let step = domain
                        .step
                        .evaluate(&empty)
                        .map_err(|e| error(n.offset, e))?;
                    if count <= 0
                        || step <= 0
                        || stop <= start
                        || (stop - start) % count != 0
                        || ((stop - start) / count) % step != 0
                    {
                        return Err(error(
                            n.offset,
                            "mloop split must exactly divide the iteration range",
                        ));
                    }
                    let chunk = (stop - start) / count;
                    let inner_start = IndexExpr::Add(
                        Box::new(IndexExpr::Constant(start)),
                        Box::new(IndexExpr::Mul(
                            Box::new(IndexExpr::Variable(split_var.clone())),
                            Box::new(IndexExpr::Constant(chunk)),
                        )),
                    );
                    let inner_stop = IndexExpr::Add(
                        Box::new(inner_start.clone()),
                        Box::new(IndexExpr::Constant(chunk)),
                    );
                    child_scope.insert(split_var.clone());
                    let body = self.statements(&args[6], &child_scope, true)?;
                    Ok(vec![Statement::Loop(Loop {
                        kind: LoopKind::Parallel,
                        domain: LoopDomain {
                            variable: split_var,
                            start: IndexExpr::Constant(0),
                            stop: IndexExpr::Constant(count),
                            step: IndexExpr::Constant(1),
                        },
                        body: vec![Statement::Loop(Loop {
                            kind: LoopKind::Sequential,
                            domain: LoopDomain {
                                start: inner_start,
                                stop: inner_stop,
                                ..domain
                            },
                            body,
                        })],
                    })])
                } else {
                    let body = self.statements(&args[4], &child_scope, serial || !parallel)?;
                    Ok(vec![Statement::Loop(Loop {
                        kind: if parallel {
                            LoopKind::Parallel
                        } else {
                            LoopKind::Sequential
                        },
                        domain,
                        body,
                    })])
                }
            }
            "store" => {
                let expression = self.notation(n, scope)?;
                let args = n.args(3)?;
                let name = args[0].args(2)?[0].args(1)?[0].atom()?;
                let &(output, ref role, _) = self
                    .tensors
                    .get(name)
                    .ok_or_else(|| error(n.offset, "unknown store tensor"))?;
                if role == "input" {
                    return Err(error(n.offset, "cannot store to input"));
                }
                fn loads(n: &Node, out: &mut BTreeSet<String>) -> Result<(), LoopIrError> {
                    if n.op() == "load" {
                        out.insert(n.args(2)?[0].args(2)?[0].args(1)?[0].atom()?.into());
                    }
                    for c in &n.children {
                        loads(c, out)?;
                    }
                    Ok(())
                }
                let mut names = BTreeSet::new();
                loads(&args[1], &mut names)?;
                let inputs = names
                    .iter()
                    .map(|name| self.tensors[name].0)
                    .collect::<Vec<_>>();
                fn has_matmul(e: &Expression) -> bool {
                    e.operator() == Some("@")
                        || e.list().is_some_and(|xs| xs.iter().any(has_matmul))
                }
                let implementation = if has_matmul(&expression) {
                    self.gemm_instance(&expression, n.offset)?
                } else {
                    crate::implementation::expression_instance(self.config.target)
                };
                let id = self
                    .builder
                    .add_expression(inputs, [output], expression, implementation);
                self.positions.push(n.offset);
                Ok(vec![Statement::Operation(id)])
            }
            other => Err(error(n.offset, format!("unsupported program node {other}"))),
        }
    }
}

/// Read a scheduled program without invoking the optimizer or replacing its schedule.
pub fn lower_loop_ir(text: &str, config: &LoopIrConfig) -> Result<Vec<PhysicalPlan>, LoopIrError> {
    let node = parse(text)?;
    let mut reader = Reader {
        config,
        builder: PhysicalPlanBuilder::new(config.target, config.world_size),
        tensors: BTreeMap::new(),
        output: None,
        positions: Vec::new(),
    };
    reader.collect(&node)?;
    for statement in reader.statements(&node, &BTreeSet::new(), false)? {
        reader.builder.add_statement(statement);
    }
    let (name, output) = reader
        .output
        .ok_or_else(|| error(0, "missing output tensor"))?;
    let plan = reader
        .builder
        .finalize(name, output)
        .map_err(|e| error(0, e.to_string()))?;
    validate_indices(&plan, &reader.positions)?;
    Ok(vec![plan])
}

/// Validate notation addresses under concrete lexical bindings, not overlap or
/// schedule legality. This prevents an accepted program from emitting unsafe loads.
fn validate_indices(plan: &PhysicalPlan, positions: &[usize]) -> Result<(), LoopIrError> {
    fn expr(
        e: &Expression,
        bindings: &BTreeMap<String, i64>,
        steps: &BTreeMap<String, i64>,
        offset: usize,
        mma_operand: bool,
    ) -> Result<(), LoopIrError> {
        if let Some(xs) = e.list() {
            if matches!(e.operator(), Some("load" | "store")) {
                let view = xs[1].list().ok_or_else(|| error(offset, "expected view"))?;
                if view.len() != 3 {
                    return Err(error(offset, "invalid view"));
                }
                let layout = view[2]
                    .list()
                    .ok_or_else(|| error(offset, "expected layout"))?;
                let index = xs
                    .last()
                    .unwrap()
                    .list()
                    .ok_or_else(|| error(offset, "expected keyed_index"))?;
                let mut slots = BTreeMap::new();
                for slot in &index[1..] {
                    let s = slot.list().ok_or_else(|| error(offset, "expected slot"))?;
                    if s.len() != 3 || slots.insert(s[1].atom().unwrap_or(""), &s[2]).is_some() {
                        return Err(error(offset, "duplicate/invalid index slot"));
                    }
                }
                for (axis_position, axis) in layout[1..].iter().enumerate() {
                    let a = axis.list().ok_or_else(|| error(offset, "expected axis"))?;
                    let extent: i64 = a[2]
                        .atom()
                        .unwrap_or("")
                        .parse()
                        .map_err(|_| error(offset, "invalid extent"))?;
                    let contiguous_axis = axis_position + 2 == layout.len();
                    if mma_operand && contiguous_axis && extent % 8 != 0 {
                        return Err(error(
                            offset,
                            "WGMMA operand stride must be aligned to eight BF16 elements",
                        ));
                    }
                    let Some(part) = slots.remove(a[1].atom().unwrap_or("")) else {
                        continue;
                    };
                    if part.atom() == Some("fulltile") {
                        continue;
                    }
                    let p = part
                        .list()
                        .ok_or_else(|| error(offset, "invalid tile index"))?;
                    let var = p
                        .get(1)
                        .and_then(Expression::atom)
                        .ok_or_else(|| error(offset, "missing index variable"))?;
                    let value = *bindings
                        .get(var)
                        .ok_or_else(|| error(offset, format!("unbound index {var}")))?;
                    let (start, width) = match part.operator() {
                        Some("tile") => (
                            value,
                            p.get(2)
                                .and_then(Expression::atom)
                                .and_then(|s| s.parse::<i64>().ok())
                                .ok_or_else(|| error(offset, "invalid tile width"))?,
                        ),
                        Some("elem") => (value / steps[var], 1),
                        _ => return Err(error(offset, "unsupported index")),
                    };
                    if mma_operand && contiguous_axis && start % 8 != 0 {
                        return Err(error(
                            offset,
                            "WGMMA operand offset must be aligned to eight BF16 elements",
                        ));
                    }
                    if start < 0
                        || width <= 0
                        || start.checked_add(width).is_none_or(|end| end > extent)
                    {
                        return Err(error(
                            offset,
                            format!(
                                "out-of-bounds tile: {start} + {width} exceeds axis extent {extent}"
                            ),
                        ));
                    }
                }
                if !slots.is_empty() {
                    return Err(error(offset, "index slot absent from view layout"));
                }
            }
            for x in &xs[1..] {
                expr(x, bindings, steps, offset, e.operator() == Some("@"))?;
            }
        }
        Ok(())
    }
    fn visit(
        statements: &[Statement],
        plan: &PhysicalPlan,
        positions: &[usize],
        bindings: &BTreeMap<String, i64>,
        steps: &BTreeMap<String, i64>,
    ) -> Result<(), LoopIrError> {
        for a in statements {
            match a {
                Statement::Operation(id) => expr(
                    plan.operation(*id).unwrap().expression().unwrap(),
                    bindings,
                    steps,
                    positions[id.index()],
                    false,
                )?,
                Statement::Loop(l) => {
                    let offset = l
                        .body
                        .iter()
                        .flat_map(Statement::operations)
                        .next()
                        .map(|id| positions[id.index()])
                        .unwrap_or(0);
                    let step = l
                        .domain
                        .step
                        .evaluate(bindings)
                        .map_err(|e| error(offset, e))?;
                    for v in l.domain.values(bindings).map_err(|e| error(offset, e))? {
                        let mut b = bindings.clone();
                        let mut s = steps.clone();
                        b.insert(l.domain.variable.clone(), v);
                        s.insert(l.domain.variable.clone(), step);
                        visit(&l.body, plan, positions, &b, &s)?;
                    }
                }
            }
        }
        Ok(())
    }
    for statement in plan.statements() {
        visit(
            std::slice::from_ref(statement),
            plan,
            positions,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )?;
    }
    Ok(())
}
