use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AnalysisError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("invalid {op} at {span:?}: {message}")]
    InvalidIr {
        op: String,
        span: Option<SourceSpan>,
        message: String,
    },
}

fn invalid(node: &IrNode, message: impl Into<String>) -> AnalysisError {
    AnalysisError::InvalidIr {
        op: node.head().to_owned(),
        span: node.span(),
        message: message.into(),
    }
}

fn args(node: &IrNode, count: usize) -> Result<&[IrNode], AnalysisError> {
    node.args()
        .filter(|args| args.len() == count)
        .ok_or_else(|| invalid(node, format!("expected {count} arguments")))
}

fn symbol(node: &IrNode) -> Result<&str, AnalysisError> {
    if node.args().is_some() || node.head().is_empty() {
        return Err(invalid(node, "expected a symbol"));
    }
    Ok(node.head())
}

pub fn analyze_text(text: &str) -> Result<ScheduledIr, AnalysisError> {
    analyze(IrNode::parse(text)?)
}

/// Collect one extracted program. The returned snapshot owns the unmodified IR.
pub fn analyze(root: IrNode) -> Result<ScheduledIr, AnalysisError> {
    let mut collector = Collector::default();
    collector.program(&root)?;
    Ok(ScheduledIr {
        ir: Some(root),
        tensors: collector.tensors,
        kernels: collector.kernels,
        scopes: collector.scopes,
        statements: collector.statements,
        accesses: collector.accesses,
    })
}

#[derive(Default)]
struct Collector {
    tensors: Vec<TensorInfo>,
    names: BTreeMap<String, TensorId>,
    kernels: Vec<KernelInfo>,
    scopes: Vec<ScopeInfo>,
    statements: Vec<StatementInfo>,
    accesses: Vec<AccessInfo>,
    bindings: Vec<(String, ScopeId)>,
}

impl Collector {
    fn program(&mut self, node: &IrNode) -> Result<(), AnalysisError> {
        if node.head() == "seq" {
            for child in args(node, 2)? {
                self.program(child)?;
            }
        } else if !Self::only_dummy(node)? {
            let kernel = KernelId(self.kernels.len());
            let root_scope = ScopeId(self.scopes.len());
            self.scopes.push(ScopeInfo {
                kernel,
                parent: None,
                kind: ScopeKind::Kernel,
                loop_info: None,
                source_span: node.span(),
                children: Vec::new(),
                accesses: Vec::new(),
                read_writes: ReadWrites::default(),
            });
            self.kernels.push(KernelInfo {
                root_scope,
                accesses: Vec::new(),
                read_writes: ReadWrites::default(),
            });
            self.statement(node, root_scope)?;
        }
        Ok(())
    }

    fn only_dummy(node: &IrNode) -> Result<bool, AnalysisError> {
        match node.head() {
            "dummy" if node.args().is_none() => Ok(true),
            "seq" => {
                let children = args(node, 2)?;
                Ok(Self::only_dummy(&children[0])? && Self::only_dummy(&children[1])?)
            }
            "sloop" | "ploop" => Self::only_dummy(&args(node, 5)?[4]),
            "mloop" => Self::only_dummy(&args(node, 7)?[6]),
            _ => Ok(false),
        }
    }

    fn statement(&mut self, node: &IrNode, scope: ScopeId) -> Result<(), AnalysisError> {
        match node.head() {
            "dummy" if node.args().is_none() => Ok(()),
            "seq" => {
                for child in args(node, 2)? {
                    self.statement(child, scope)?;
                }
                Ok(())
            }
            "mloop" => self.mloop(node, scope),
            "ploop" | "sloop" => {
                let children = args(node, 5)?;
                let variable = symbol(&children[3])?.to_owned();
                let loop_info = LoopInfo {
                    variable: variable.clone(),
                    start: self.index_expr(&children[0]),
                    end: self.index_expr(&children[1]),
                    step: self.index_expr(&children[2]),
                };
                if matches!(loop_info.step, IndexExpr::Integer(step) if step <= 0) {
                    return Err(invalid(node, "loop step must be positive"));
                }
                let id = ScopeId(self.scopes.len());
                self.scopes.push(ScopeInfo {
                    kernel: self.scopes[scope.0].kernel,
                    parent: Some(scope),
                    kind: if node.head() == "ploop" {
                        ScopeKind::ParallelLoop
                    } else {
                        ScopeKind::SequentialLoop
                    },
                    loop_info: Some(loop_info),
                    source_span: node.span(),
                    children: Vec::new(),
                    accesses: Vec::new(),
                    read_writes: ReadWrites::default(),
                });
                self.scopes[scope.0].children.push(ScopeItem::Scope(id));
                self.bindings.push((variable, id));
                self.statement(&children[4], id)?;
                self.bindings.pop();
                Ok(())
            }
            "store" => self.store(node, scope),
            _ => Err(invalid(
                node,
                format!(
                    "unsupported program node {}; expected seq, ploop, sloop, mloop, store or dummy",
                    node.head()
                ),
            )),
        }
    }

    fn mloop(&mut self, node: &IrNode, parent: ScopeId) -> Result<(), AnalysisError> {
        let c = args(node, 7)?;
        let start = self.index_expr(&c[0]);
        let stop = self.index_expr(&c[1]);
        let step = self.index_expr(&c[2]);
        let count = self.index_expr(&c[5]);
        let serial_name = symbol(&c[3])?.to_owned();
        let split_name = symbol(&c[4])?.to_owned();
        if serial_name == split_name {
            return Err(invalid(node, "mloop bindings must have distinct names"));
        }
        let split = ScopeId(self.scopes.len());
        let serial = ScopeId(split.0 + 1);
        let apply = |op: &str, a, b| IndexExpr::Apply(op.into(), vec![a, b]);
        let chunk = apply("//", apply("-", stop, start.clone()), count.clone());
        let inner_start = apply(
            "+",
            start.clone(),
            apply("*", IndexExpr::LoopVar(split), chunk.clone()),
        );
        let inner_end = apply(
            "+",
            start,
            apply(
                "*",
                apply("+", IndexExpr::LoopVar(split), IndexExpr::Integer(1)),
                chunk,
            ),
        );
        // The original atomic mloop stays in ScheduledIr.ir. Two lexical
        // scopes give its two bindings distinct IDs without inventing a schedule.
        for (id, parent, kind, info) in [
            (
                split,
                parent,
                ScopeKind::SplitLoop,
                LoopInfo {
                    variable: split_name.clone(),
                    start: IndexExpr::Integer(0),
                    end: count,
                    step: IndexExpr::Integer(1),
                },
            ),
            (
                serial,
                split,
                ScopeKind::SequentialLoop,
                LoopInfo {
                    variable: serial_name.clone(),
                    start: inner_start,
                    end: inner_end,
                    step,
                },
            ),
        ] {
            self.scopes.push(ScopeInfo {
                kernel: self.scopes[parent.0].kernel,
                parent: Some(parent),
                kind,
                loop_info: Some(info),
                source_span: node.span(),
                children: Vec::new(),
                accesses: Vec::new(),
                read_writes: ReadWrites::default(),
            });
            self.scopes[parent.0].children.push(ScopeItem::Scope(id));
        }
        self.bindings.push((split_name, split));
        self.bindings.push((serial_name, serial));
        self.statement(&c[6], serial)?;
        self.bindings.truncate(self.bindings.len() - 2);
        Ok(())
    }

    fn view_axes(base: &IrNode) -> Option<Vec<String>> {
        (base.head() == "view").then(|| {
            base.args().unwrap()[1]
                .args()
                .unwrap()
                .iter()
                .map(|axis| axis.args().unwrap()[0].head().to_owned())
                .collect()
        })
    }

    fn tensor_group(node: &IrNode) -> Result<(TensorKind, Vec<String>), AnalysisError> {
        if node.head() == "view" {
            return Self::tensor_group(&args(node, 2)?[0]);
        }
        let kind = match node.head() {
            "input" => TensorKind::Input,
            "output" => TensorKind::Output,
            "tensor" => TensorKind::Intermediate,
            _ => return Err(invalid(node, "expected input/output/tensor reference")),
        };
        let mut names = Vec::new();
        for child in node
            .args()
            .ok_or_else(|| invalid(node, "missing tensor name"))?
        {
            for name in symbol(child)?.split(',') {
                if name.is_empty() {
                    return Err(invalid(node, "empty tensor name"));
                }
                names.push(name.to_owned());
            }
        }
        if names.is_empty() {
            return Err(invalid(node, "missing tensor name"));
        }
        Ok((kind, names))
    }

    fn tensor(&mut self, name: &str, kind: TensorKind) -> TensorId {
        let id = *self.names.entry(name.to_owned()).or_insert_with(|| {
            let id = TensorId(self.tensors.len());
            self.tensors.push(TensorInfo {
                name: name.to_owned(),
                declarations: BTreeSet::new(),
            });
            id
        });
        self.tensors[id.0].declarations.insert(kind);
        id
    }

    fn store(&mut self, node: &IrNode, scope: ScopeId) -> Result<(), AnalysisError> {
        let children = args(node, 3)?;
        let (kind, names) = Self::tensor_group(&children[0])?;
        let (index, view_shape) = self.access_index(&children[0], &children[2])?;
        for (group_index, name) in names.iter().enumerate() {
            let statement = StatementId(self.statements.len());
            self.statements.push(StatementInfo {
                scope,
                source_span: node.span(),
                group_index,
                expression: ValueExpr::Literal("0".into()),
                accesses: Vec::new(),
            });
            self.scopes[scope.0]
                .children
                .push(ScopeItem::Statement(statement));
            // Read the old value before registering the write, including self-loads.
            self.statements[statement.0].expression =
                self.expression(&children[1], statement, group_index, names.len())?;
            let tensor = self.tensor(name, kind);
            self.record(AccessInfo {
                tensor,
                kind: AccessKind::Write,
                statement,
                scope,
                index: index.clone(),
                view_shape: view_shape.clone(),
                view_axes: Self::view_axes(&children[0]),
                source_span: node.span(),
            });
        }
        Ok(())
    }

    fn expression(
        &mut self,
        node: &IrNode,
        statement: StatementId,
        component: usize,
        group_size: usize,
    ) -> Result<ValueExpr, AnalysisError> {
        let Some(children) = node.args() else {
            return Ok(if node.head().parse::<f64>().is_ok() {
                ValueExpr::Literal(node.head().to_owned())
            } else {
                ValueExpr::Index(self.index_expr(node))
            });
        };
        if node.head() == "load" {
            let children = args(node, 2)?;
            let (kind, names) = Self::tensor_group(&children[0])?;
            let selected = if names.len() == 1 {
                0
            } else if names.len() == group_size {
                component
            } else {
                return Err(invalid(node, "load group does not match store group"));
            };
            let (index, view_shape) = self.access_index(&children[0], &children[1])?;
            let tensor = self.tensor(&names[selected], kind);
            let access = self.record(AccessInfo {
                tensor,
                kind: AccessKind::Read,
                statement,
                scope: self.statements[statement.0].scope,
                index,
                view_shape,
                view_axes: Self::view_axes(&children[0]),
                source_span: node.span(),
            });
            Ok(ValueExpr::Load(access))
        } else {
            if matches!(
                node.head(),
                "store" | "seq" | "ploop" | "sloop" | "mloop" | "input" | "output" | "tensor"
            ) {
                return Err(invalid(
                    node,
                    "statement or tensor reference in a value expression",
                ));
            }
            Ok(ValueExpr::Apply(
                node.head().to_owned(),
                children
                    .iter()
                    .map(|child| self.expression(child, statement, component, group_size))
                    .collect::<Result<_, _>>()?,
            ))
        }
    }

    fn record(&mut self, access: AccessInfo) -> AccessId {
        let id = AccessId(self.accesses.len());
        let scope = &mut self.scopes[access.scope.0];
        let kernel = &mut self.kernels[scope.kernel.0];
        for summary in [&mut scope.read_writes, &mut kernel.read_writes] {
            match access.kind {
                AccessKind::Read => summary.reads.insert(access.tensor),
                AccessKind::Write => summary.writes.insert(access.tensor),
            };
        }
        scope.accesses.push(id);
        kernel.accesses.push(id);
        self.statements[access.statement.0].accesses.push(id);
        self.accesses.push(access);
        id
    }

    fn index_expr(&self, node: &IrNode) -> IndexExpr {
        if let Some(args) = node.args() {
            IndexExpr::Apply(
                node.head().to_owned(),
                args.iter().map(|n| self.index_expr(n)).collect(),
            )
        } else if let Ok(value) = node.head().parse() {
            IndexExpr::Integer(value)
        } else if let Some((_, scope)) = self
            .bindings
            .iter()
            .rev()
            .find(|(name, _)| name == node.head())
        {
            IndexExpr::LoopVar(*scope)
        } else {
            IndexExpr::Symbol(node.head().to_owned())
        }
    }

    fn access_index(
        &self,
        base: &IrNode,
        index: &IrNode,
    ) -> Result<(Vec<IndexDim>, Option<Vec<IndexExpr>>), AnalysisError> {
        let mut axes = Vec::new();
        let view_shape = if base.head() == "view" {
            let layout = &args(base, 2)?[1];
            if layout.head() != "layout" {
                return Err(invalid(layout, "expected layout"));
            }
            let mut shape = Vec::new();
            for axis in layout
                .args()
                .ok_or_else(|| invalid(layout, "missing axes"))?
            {
                if axis.head() != "axis" {
                    return Err(invalid(axis, "expected axis"));
                }
                let children = args(axis, 2)?;
                let name = symbol(&children[0])?.to_owned();
                if axes.contains(&name) {
                    return Err(invalid(axis, "duplicate layout axis"));
                }
                axes.push(name);
                shape.push(self.index_expr(&children[1]));
            }
            if shape.is_empty() {
                return Err(invalid(layout, "empty layout"));
            }
            Some(shape)
        } else {
            None
        };
        if index.head() != "keyed_index" {
            return Ok((self.index(index)?, view_shape));
        }
        let mut slots = BTreeMap::new();
        for slot in index
            .args()
            .ok_or_else(|| invalid(index, "missing slots"))?
        {
            if slot.head() != "slot" {
                return Err(invalid(slot, "expected slot"));
            }
            let children = args(slot, 2)?;
            let name = symbol(&children[0])?.to_owned();
            if slots.insert(name, children[1].clone()).is_some() {
                return Err(invalid(slot, "duplicate keyed index slot"));
            }
        }
        if view_shape.is_none() {
            // Bare tensors have no named layout. Only the explicit positional
            // a_0, a_1, ... convention supplies an unambiguous axis mapping.
            axes = (0..slots.len()).map(|i| format!("a_{i}")).collect();
            if axes.iter().any(|name| !slots.contains_key(name)) {
                return Err(invalid(
                    index,
                    "named slots require a view layout (or contiguous a_0, a_1, ... slots)",
                ));
            }
        }
        let dimensions: Vec<_> = axes
            .iter()
            .map(|name| {
                slots
                    .remove(name)
                    .unwrap_or_else(|| IrNode::atom("fulltile"))
            })
            .collect();
        if !slots.is_empty() {
            return Err(invalid(index, "slot is absent from view layout"));
        }
        Ok((self.index(&IrNode::call("index", dimensions))?, view_shape))
    }

    fn index(&self, node: &IrNode) -> Result<Vec<IndexDim>, AnalysisError> {
        let dimensions = if node.head() == "fulltile" && node.args().is_none() {
            std::slice::from_ref(node)
        } else if node.head() == "index" {
            node.args()
                .filter(|a| !a.is_empty())
                .ok_or_else(|| invalid(node, "empty index"))?
        } else {
            return Err(invalid(node, "expected index or fulltile"));
        };
        dimensions.iter().map(|dim| {
            match dim.head() {
                "fulltile" if dim.args().is_none() => Ok(IndexDim::FullTile),
                "elem" => Ok(IndexDim::Elem(self.index_expr(&args(dim, 1)?[0]))),
                "const_tile" => {
                    let args = args(dim, 2)?;
                    Ok(IndexDim::ConstTile { start: self.index_expr(&args[0]), width: self.index_expr(&args[1]) })
                }
                "tile" => {
                    let children = dim.args().ok_or_else(|| invalid(dim, "missing tile arguments"))?;
                    let (start, width) = match children {
                        [start, width] => (self.index_expr(start), self.index_expr(width)),
                        [start] => {
                            let start = self.index_expr(start);
                            let IndexExpr::LoopVar(scope) = start else {
                                return Err(invalid(dim, "legacy tile requires an in-scope loop variable; use explicit width otherwise"));
                            };
                            (IndexExpr::LoopVar(scope), self.scopes[scope.0].loop_info.as_ref().unwrap().step.clone())
                        }
                        _ => return Err(invalid(dim, "expected tile start and optional width")),
                    };
                    if matches!(width, IndexExpr::Integer(value) if value <= 0) {
                        return Err(invalid(dim, "tile width must be positive"));
                    }
                    Ok(IndexDim::Tile { start, width })
                }
                _ => Err(invalid(dim, "unsupported index dimension")),
            }
        }).collect()
    }
}
