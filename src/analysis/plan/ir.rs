//! Read IR into an explicit physical program, preserving its loops and accesses.
use super::{
    AccessIndex, Constant, Expression, IndexExpr, Loop, LoopDomain, LoopKind, PhysicalPlan,
    PhysicalPlanBuilder, Statement, Storage, TensorAccess, TileWidth, ValueInstanceId,
};
use crate::{CudaTargetCapability, DType, TargetCapability};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct IrConfig {
    pub target: TargetCapability,
    pub world_size: usize,
    pub symbols: BTreeMap<String, i64>,
    pub dtypes: BTreeMap<String, DType>,
}

impl Default for IrConfig {
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
#[error("IR at byte {offset}: {message}")]
pub struct IrError {
    pub offset: usize,
    pub message: String,
}
fn error(offset: usize, message: impl Into<String>) -> IrError {
    IrError {
        offset,
        message: message.into(),
    }
}

// Reader-specific arity and atom checks over the common syntax tree. These
// helpers interpret syntax; they do not tokenize or create a second AST.
use crate::analysis::IrNode as Node;

trait ReaderNode {
    fn op(&self) -> &str;
    fn text(&self) -> Option<&str>;
    fn offset(&self) -> usize;
    fn as_atom(&self) -> Result<&str, IrError>;
    fn expect_args(&self, count: usize) -> Result<&[Node], IrError>;
}

impl ReaderNode for Node {
    fn op(&self) -> &str {
        if self.args().is_some() {
            self.head()
        } else {
            ""
        }
    }
    fn text(&self) -> Option<&str> {
        self.args().is_none().then(|| self.head())
    }
    fn offset(&self) -> usize {
        self.span().map_or(0, |span| span.start)
    }
    fn as_atom(&self) -> Result<&str, IrError> {
        self.text()
            .ok_or_else(|| error(self.offset(), "expected atom"))
    }
    fn expect_args(&self, count: usize) -> Result<&[Node], IrError> {
        self.args()
            .filter(|args| args.len() == count)
            .ok_or_else(|| {
                error(
                    self.offset(),
                    format!("{} expects {count} arguments", self.op()),
                )
            })
    }
}

/// Read an explicit program, preserving its schedule and inferring value storage
/// through the same common analysis as `PhysicalPlanBuilder::from_scheduled`.
pub fn lower_ir(text: &str, config: &IrConfig) -> Result<Vec<PhysicalPlan>, IrError> {
    let ir = crate::analysis::analyze(parse(text)?).map_err(|e| {
        let offset = match &e {
            crate::analysis::AnalysisError::InvalidIr { span, .. } => {
                span.as_ref().map_or(0, |s| s.start)
            }
            _ => 0,
        };
        error(offset, e.to_string())
    })?;
    let node = ir.ir().expect("analysis retains the parsed source");
    let mut bindings = crate::analysis::Bindings {
        shapes: BTreeMap::new(),
        symbols: config.symbols.clone(),
    };
    let storage = crate::analysis::storage::infer_source(&ir, &mut bindings)
        .map_err(|e| error(0, e.to_string()))?;

    let mut reader = Reader {
        config,
        builder: PhysicalPlanBuilder::new(config.target, config.world_size),
        tensors: BTreeMap::new(),
        output: None,
        storage: storage
            .values
            .into_iter()
            .map(|(id, class)| (ir.tensor(id).name.clone(), class))
            .collect(),
    };
    reader.collect(node)?;

    let statements = reader.statements(node, &BTreeSet::new(), false)?;

    let (name, output) = reader
        .output
        .ok_or_else(|| error(0, "missing output tensor"))?;

    let plan = reader
        .builder
        .build(statements, name, output)
        .map_err(|e| error(0, e.to_string()))?;

    Ok(vec![plan])
}

fn parse(text: &str) -> Result<Node, IrError> {
    Node::parse(text).map_err(|e| error(e.offset, e.message))
}

struct Reader<'a> {
    config: &'a IrConfig,
    builder: PhysicalPlanBuilder,
    tensors: BTreeMap<String, (ValueInstanceId, String, Vec<usize>)>,
    output: Option<(String, ValueInstanceId)>,
    storage: BTreeMap<String, Storage>,
}

impl PhysicalPlanBuilder {
    /// Resolves the Python Builder's expression notation against registered values.
    pub(crate) fn parse_expression(&self, text: &str) -> Result<Expression, IrError> {
        Decoder {
            builder: self,
            symbols: &BTreeMap::new(),
            scope: None,
            text_ir: false,
        }
        .expression(&parse(text)?)
    }
}

/// Python loop bounds use the same syntax reader, without passing through computation IR.
pub(crate) fn parse_index(text: &str) -> Result<IndexExpr, IrError> {
    index_node(&parse(text)?)
}

fn index_node(node: &Node) -> Result<IndexExpr, IrError> {
    if let Some(atom) = node.text() {
        return Ok(atom
            .parse()
            .map(IndexExpr::Constant)
            .unwrap_or_else(|_| IndexExpr::Variable(atom.to_owned())));
    }
    let args = node.expect_args(2)?;
    let left = Box::new(index_node(&args[0])?);
    let right = Box::new(index_node(&args[1])?);
    Ok(match node.op() {
        "+" => IndexExpr::Add(left, right),
        "-" => IndexExpr::Sub(left, right),
        "*" => IndexExpr::Mul(left, right),
        "/" => IndexExpr::Div(left, right),
        _ => return Err(error(node.offset(), "unsupported index expression")),
    })
}

/// Input-only decoding shared by the text Reader and Python Builder.
struct Decoder<'a> {
    builder: &'a PhysicalPlanBuilder,
    symbols: &'a BTreeMap<String, i64>,
    scope: Option<&'a BTreeSet<String>>,
    text_ir: bool,
}

impl Decoder<'_> {
    fn integer(&self, node: &Node) -> Result<i64, IrError> {
        let name = node.as_atom()?;
        name.parse()
            .ok()
            .or_else(|| self.symbols.get(name).copied())
            .ok_or_else(|| {
                error(
                    node.offset(),
                    format!("unresolved integer or symbol {name}"),
                )
            })
    }

    fn axis(&self, node: &Node) -> Result<usize, IrError> {
        usize::try_from(self.integer(node)?)
            .map_err(|_| error(node.offset(), "axis must be nonnegative"))
    }

    fn extent(&self, node: &Node) -> Result<usize, IrError> {
        let extent = self.integer(node)?;
        if extent <= 0 {
            return Err(error(node.offset(), "extent must be positive"));
        }
        usize::try_from(extent).map_err(|_| error(node.offset(), "extent exceeds usize"))
    }

    fn expression(&self, node: &Node) -> Result<Expression, IrError> {
        if let Some(atom) = node.text() {
            let constant = if let Ok(value) = self.integer(node) {
                Constant::Integer(value)
            } else if !self.text_ir {
                Constant::Float32(
                    atom.parse::<f32>()
                        .map_err(|_| error(node.offset(), "invalid scalar constant"))?
                        .to_bits(),
                )
            } else {
                return Err(error(node.offset(), format!("unresolved symbol {atom}")));
            };
            return Ok(Expression::Constant(constant));
        }
        let op = node.op();
        Ok(match op {
            "load" => {
                let args = node.expect_args(2)?;
                Expression::Load(self.access(&args[0], &args[1])?)
            }
            "store" => {
                let args = node.expect_args(3)?;
                Expression::Store {
                    destination: self.access(&args[0], &args[2])?,
                    value: Box::new(self.expression(&args[1])?),
                }
            }
            "+" | "-" | "*" | "/" | "@" => {
                let args = node.expect_args(2)?;
                let operands = Box::new([self.expression(&args[0])?, self.expression(&args[1])?]);
                match op {
                    "+" => Expression::Add(operands),
                    "-" => Expression::Sub(operands),
                    "*" => Expression::Mul(operands),
                    "/" => Expression::Div(operands),
                    _ => Expression::Matmul(operands),
                }
            }
            "sqr" | "sqrt" | "sigmoid" | "relu" if op != "relu" || !self.text_ir => {
                let args = node.expect_args(1)?;
                let value = Box::new(self.expression(&args[0])?);
                match op {
                    "sqr" => Expression::Sqr(value),
                    "sqrt" => Expression::Sqrt(value),
                    "sigmoid" => Expression::Sigmoid(value),
                    _ => Expression::Relu(value),
                }
            }
            "rsum" | "bcast" | "unsqueeze" => {
                let args = node.expect_args(2)?;
                let value = Box::new(self.expression(&args[0])?);
                let axis = self.axis(&args[1])?;
                match op {
                    "rsum" => Expression::ReduceSum { value, axis },
                    "bcast" => Expression::Broadcast { value, axis },
                    _ => Expression::Unsqueeze { value, axis },
                }
            }
            "float_bits" if !self.text_ir => {
                let args = node.expect_args(1)?;
                let bits = u32::try_from(self.integer(&args[0])?)
                    .map_err(|_| error(node.offset(), "invalid FP32 bits"))?;
                Expression::Constant(Constant::Float32(bits))
            }
            "all_gather" if !self.text_ir => {
                let args = node.expect_args(5)?;
                Expression::AllGather {
                    source: self.access(&args[0], &args[1])?,
                    destination: self.access(&args[2], &args[3])?,
                    axis: self.axis(&args[4])?,
                }
            }
            _ => return Err(error(node.offset(), format!("unsupported operation {op}"))),
        })
    }

    fn access(&self, view: &Node, index: &Node) -> Result<TensorAccess, IrError> {
        if view.op() != "view" || index.op() != "keyed_index" {
            return Err(error(view.offset(), "expected view and keyed_index"));
        }
        let view = view.expect_args(2)?;
        if !matches!(view[0].op(), "input" | "output" | "tensor") || view[1].op() != "layout" {
            return Err(error(view[0].offset(), "invalid tensor view"));
        }
        let base = view[0].expect_args(1)?;
        let name = base[0].as_atom()?;
        let mut values = self
            .builder
            .values
            .iter()
            .filter(|(_, value)| value.name() == Some(name));
        let (id, value) = values
            .next()
            .ok_or_else(|| error(base[0].offset(), format!("unknown tensor {name}")))?;
        if values.next().is_some() {
            return Err(error(base[0].offset(), format!("ambiguous tensor {name}")));
        }
        let layout = view[1].args().unwrap_or_default();
        let mut slots = BTreeMap::new();
        for slot in index.args().unwrap_or_default() {
            if slot.op() != "slot" {
                return Err(error(slot.offset(), "expected index slot"));
            }
            let args = slot.expect_args(2)?;
            if slots.insert(args[0].as_atom()?, &args[1]).is_some() {
                return Err(error(slot.offset(), "duplicate index slot"));
            }
        }
        let mut axes = BTreeSet::new();
        let mut indices = Vec::new();
        let mut shape = Vec::new();
        for axis in layout {
            if axis.op() != "axis" {
                return Err(error(axis.offset(), "expected layout axis"));
            }
            let args = axis.expect_args(2)?;
            let label = args[0].as_atom()?;
            if !axes.insert(label) {
                return Err(error(axis.offset(), "duplicate layout axis"));
            }
            shape.push(self.extent(&args[1])?);
            indices.push(match slots.remove(label) {
                None => AccessIndex::FullTile,
                Some(index) if index.text() == Some("fulltile") => AccessIndex::FullTile,
                Some(index) => self.access_index(index)?,
            });
        }
        if !slots.is_empty() {
            return Err(error(index.offset(), "index slot absent from view"));
        }
        let mut access = TensorAccess::new(id, indices);
        if shape != value.shape() {
            access = access.with_view_shape(shape);
        }
        access
            .validate_view(value.shape())
            .map_err(|e| error(view[1].offset(), e))?;
        Ok(access)
    }

    fn tile_width(&self, node: &Node) -> Result<TileWidth, IrError> {
        let atom = node.as_atom()?;
        if super::expression::is_symbol(atom) && !self.symbols.contains_key(atom) {
            Ok(TileWidth::Symbol(atom.into()))
        } else {
            self.extent(node).map(TileWidth::Constant)
        }
    }

    fn access_index(&self, node: &Node) -> Result<AccessIndex, IrError> {
        let count = match node.op() {
            "tile" => 2,
            "clipped_tile" if !self.text_ir => 2,
            "elem" => 1,
            _ => return Err(error(node.offset(), "unsupported access index")),
        };
        let args = node.expect_args(count)?;
        let variable = args[0].as_atom()?.to_owned();
        if self.scope.is_some_and(|scope| !scope.contains(&variable)) {
            return Err(error(node.offset(), format!("unbound index {variable}")));
        }
        Ok(match node.op() {
            "tile" => AccessIndex::Tile {
                variable,
                width: self.tile_width(&args[1])?,
            },
            "clipped_tile" => AccessIndex::ClippedTile {
                variable,
                width: self.tile_width(&args[1])?,
            },
            _ => AccessIndex::Elem(variable),
        })
    }
}

impl Reader<'_> {
    fn number(&self, n: &Node) -> Result<i64, IrError> {
        let atom = n.as_atom()?;
        atom.parse()
            .ok()
            .or_else(|| self.config.symbols.get(atom).copied())
            .ok_or_else(|| error(n.offset(), format!("unresolved symbol {atom}")))
    }
    fn index(&self, n: &Node, scope: &BTreeSet<String>) -> Result<IndexExpr, IrError> {
        if let Some(a) = n.text() {
            return if scope.contains(a)
                || (super::expression::is_symbol(a) && !self.config.symbols.contains_key(a))
            {
                Ok(IndexExpr::Variable(a.into()))
            } else {
                self.number(n).map(IndexExpr::Constant)
            };
        }
        let args = n.expect_args(2)?;
        let a = Box::new(self.index(&args[0], scope)?);
        let b = Box::new(self.index(&args[1], scope)?);
        match n.op() {
            "+" => Ok(IndexExpr::Add(a, b)),
            "-" => Ok(IndexExpr::Sub(a, b)),
            "*" => Ok(IndexExpr::Mul(a, b)),
            "/" => Ok(IndexExpr::Div(a, b)),
            _ => Err(error(n.offset(), "unsupported range expression")),
        }
    }
    fn collect(&mut self, n: &Node) -> Result<(), IrError> {
        if n.op() == "view" {
            let args = n.expect_args(2)?;
            let base = args[0].expect_args(1)?;
            let role = args[0].op();
            if !matches!(role, "input" | "tensor" | "output") {
                return Err(error(n.offset(), "unsupported view base"));
            }
            let name = base[0].as_atom()?.to_owned();
            if args[1].op() != "layout" {
                return Err(error(n.offset(), "expected layout"));
            }
            let mut shape = Vec::new();
            let mut axes = BTreeSet::new();
            for axis in args[1].args().unwrap_or_default() {
                if axis.op() != "axis" {
                    return Err(error(axis.offset(), "expected axis"));
                }
                let a = axis.expect_args(2)?;
                if !axes.insert(a[0].as_atom()?) {
                    return Err(error(axis.offset(), "duplicate layout axis"));
                }
                let size = self.number(&a[1])?;
                if size <= 0 {
                    return Err(error(axis.offset(), "nonpositive tensor extent"));
                }
                shape.push(size as usize);
            }
            if !(1..=3).contains(&shape.len()) {
                return Err(error(n.offset(), "supported tensor ranks are 1, 2, 3"));
            }
            if let Some((_, old_role, old_shape)) = self.tensors.get(&name) {
                let elements =
                    super::expression::element_count(&shape).map_err(|e| error(n.offset(), e))?;
                let old_elements = super::expression::element_count(old_shape)
                    .map_err(|e| error(n.offset(), e))?;
                if old_role != role || old_elements != elements {
                    return Err(error(n.offset(), format!("inconsistent view for {name}")));
                }
            } else {
                let dtype = *self
                    .config
                    .dtypes
                    .get(&name)
                    .ok_or_else(|| error(n.offset(), format!("missing dtype for {name}")))?;
                let storage = self.storage[&name];
                let id = self
                    .builder
                    .add_named_value(&name, dtype, shape.iter().copied(), storage);
                if role == "input" {
                    self.builder.bind_input(&name, id);
                }
                if role == "output" {
                    if self.output.is_some() {
                        return Err(error(n.offset(), "one output tensor is supported"));
                    }
                    self.output = Some((name.clone(), id));
                }
                self.tensors.insert(name, (id, role.into(), shape));
            }
        }
        for child in n.args().unwrap_or_default() {
            self.collect(child)?;
        }
        Ok(())
    }
    fn notation(&self, n: &Node, scope: &BTreeSet<String>) -> Result<Expression, IrError> {
        Decoder {
            builder: &self.builder,
            symbols: &self.config.symbols,
            scope: Some(scope),
            text_ir: true,
        }
        .expression(n)
    }
    fn statements(
        &mut self,
        n: &Node,
        scope: &BTreeSet<String>,
        serial: bool,
    ) -> Result<Vec<Statement>, IrError> {
        match n.op() {
            "seq" => {
                let mut out = Vec::new();
                for child in n.expect_args(2)? {
                    out.extend(self.statements(child, scope, serial)?);
                }
                Ok(out)
            }
            "ploop" | "sloop" | "mloop" => {
                let split = n.op() == "mloop";
                let args = n.expect_args(if split { 7 } else { 5 })?;
                let parallel = n.op() != "sloop";
                if serial && parallel {
                    return Err(error(
                        n.offset(),
                        "parallel work inside a sequential loop is unsupported",
                    ));
                }
                let variable = args[3].as_atom()?.to_owned();
                let domain = LoopDomain {
                    variable: variable.clone(),
                    start: self.index(&args[0], scope)?,
                    stop: self.index(&args[1], scope)?,
                    step: self.index(&args[2], scope)?,
                };
                let mut child_scope = scope.clone();
                child_scope.insert(variable);
                if split {
                    let split_var = args[4].as_atom()?.to_owned();
                    if child_scope.contains(&split_var) {
                        return Err(error(
                            args[4].offset(),
                            "split variable shadows an active loop",
                        ));
                    }
                    let count = self.number(&args[5])?;
                    let empty = BTreeMap::new();
                    let start = domain
                        .start
                        .evaluate(&empty)
                        .map_err(|e| error(n.offset(), e))?;
                    let stop = domain
                        .stop
                        .evaluate(&empty)
                        .map_err(|e| error(n.offset(), e))?;
                    let mut step_symbols = BTreeSet::new();
                    super::bindings::free_symbols(&domain.step, scope, &mut step_symbols);
                    let step = if step_symbols.is_empty() {
                        Some(
                            domain
                                .step
                                .evaluate(&empty)
                                .map_err(|e| error(n.offset(), e))?,
                        )
                    } else {
                        None
                    };
                    if count <= 0
                        || stop <= start
                        || (stop - start) % count != 0
                        || step
                            .is_some_and(|step| step <= 0 || ((stop - start) / count) % step != 0)
                    {
                        return Err(error(
                            n.offset(),
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
                let args = n.expect_args(3)?;
                let name = args[0].expect_args(2)?[0].expect_args(1)?[0].as_atom()?;
                let &(output, ref role, _) = self
                    .tensors
                    .get(name)
                    .ok_or_else(|| error(n.offset(), "unknown store tensor"))?;
                if role == "input" {
                    return Err(error(n.offset(), "cannot store to input"));
                }
                let inflows = expression.reads();
                let id = self.builder.add_operation(inflows, [output], expression);
                Ok(vec![Statement::Operation(id)])
            }
            other => Err(error(
                n.offset(),
                format!("unsupported program node {other}"),
            )),
        }
    }
}
