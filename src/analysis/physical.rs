//! Direct projection of a PhysicalPlan into shared occurrence/scope tables.
//! No text serialization, parsing, pattern-based schedule recovery, or IR rewrite.
use super::*;
use crate::{
    AccessIndex as A, Expression as E, IndexExpr as I, PhysicalPlan, Statement, TileWidth, ValueOp,
};
use std::collections::{BTreeMap, BTreeSet};

pub fn from_physical(plan: &PhysicalPlan) -> Result<ScheduledIr, ResolveError> {
    let mut names = BTreeSet::new();
    for (_, value) in plan.value_instances() {
        if !names.insert(
            value
                .name()
                .ok_or_else(|| invalid("physical value has no normalized name"))?,
        ) {
            return Err(invalid(
                "physical value names must be unique for the named tensor projection",
            ));
        }
    }
    let tensors = plan
        .value_instances()
        .map(|(id, v)| {
            let mut declarations = BTreeSet::new();
            if plan.inputs().iter().any(|b| b.value() == id) {
                declarations.insert(TensorKind::Input);
            }
            if plan.outputs().iter().any(|b| b.value() == id) {
                declarations.insert(TensorKind::Output);
            }
            if declarations.is_empty() {
                declarations.insert(TensorKind::Intermediate);
            }
            TensorInfo {
                name: v.name().unwrap().to_owned(),
                declarations,
            }
        })
        .collect();
    let mut projection = Projection {
        plan,
        ir: ScheduledIr {
            ir: None,
            tensors,
            kernels: vec![],
            scopes: vec![],
            statements: vec![],
            accesses: vec![],
        },
        loops: BTreeMap::new(),
    };
    for statement in plan.statements() {
        let ki = KernelId(projection.ir.kernels.len());
        let root = projection.scope(ki, None, ScopeKind::Kernel, None);
        projection.ir.kernels.push(KernelInfo {
            root_scope: root,
            accesses: vec![],
            read_writes: ReadWrites::default(),
        });
        projection.statement(statement, root)?;
    }
    Ok(projection.ir)
}

struct Projection<'a> {
    plan: &'a PhysicalPlan,
    ir: ScheduledIr,
    loops: BTreeMap<String, ScopeId>,
}
impl Projection<'_> {
    fn index(&self, i: &I) -> IndexExpr {
        match i {
            I::Constant(n) => IndexExpr::Integer(*n),
            I::Variable(s) => self
                .loops
                .get(s)
                .map(|id| IndexExpr::LoopVar(*id))
                .unwrap_or_else(|| IndexExpr::Symbol(s.clone())),
            I::Add(a, b) | I::Sub(a, b) | I::Mul(a, b) | I::Div(a, b) => IndexExpr::Apply(
                match i {
                    I::Add(..) => "+",
                    I::Sub(..) => "-",
                    I::Mul(..) => "*",
                    _ => "//",
                }
                .into(),
                vec![self.index(a), self.index(b)],
            ),
        }
    }
    fn width(w: &TileWidth) -> IndexExpr {
        match w {
            TileWidth::Constant(n) => IndexExpr::Integer(*n as i64),
            TileWidth::Symbol(s) => IndexExpr::Symbol(s.clone()),
        }
    }
    fn scope(
        &mut self,
        kernel: KernelId,
        parent: Option<ScopeId>,
        kind: ScopeKind,
        loop_info: Option<LoopInfo>,
    ) -> ScopeId {
        let id = ScopeId(self.ir.scopes.len());
        self.ir.scopes.push(ScopeInfo {
            kernel,
            parent,
            kind,
            loop_info,
            source_span: None,
            children: vec![],
            accesses: vec![],
            read_writes: ReadWrites::default(),
        });
        if let Some(p) = parent {
            self.ir.scopes[p.0].children.push(ScopeItem::Scope(id));
        }
        id
    }
    fn statement(&mut self, s: &Statement, parent: ScopeId) -> Result<(), ResolveError> {
        match s {
            Statement::Region(body) => {
                for s in body {
                    self.statement(s, parent)?;
                }
            }
            Statement::Loop(l) => {
                let kind = match l.kind {
                    crate::LoopKind::Parallel => ScopeKind::ParallelLoop,
                    crate::LoopKind::Sequential => ScopeKind::SequentialLoop,
                    crate::LoopKind::Split => ScopeKind::SplitLoop,
                };
                let info = LoopInfo {
                    variable: l.domain.variable.clone(),
                    start: self.index(&l.domain.start),
                    end: self.index(&l.domain.stop),
                    step: self.index(&l.domain.step),
                };
                let id = self.scope(self.ir.scope(parent).kernel, Some(parent), kind, Some(info));
                let previous = self.loops.insert(l.domain.variable.clone(), id);
                for child in &l.body {
                    self.statement(child, id)?;
                }
                if let Some(old) = previous {
                    self.loops.insert(l.domain.variable.clone(), old);
                } else {
                    self.loops.remove(&l.domain.variable);
                }
            }
            Statement::Operation(id) => {
                let E::Store { destination, value } =
                    self.plan.operation(*id).unwrap().expression()
                else {
                    return Err(invalid(
                        "shared computation analysis does not implement communication",
                    ));
                };
                let sid = StatementId(self.ir.statements.len());
                self.ir.statements.push(StatementInfo {
                    scope: parent,
                    source_span: None,
                    group_index: 0,
                    expression: ValueExpr::Literal("0".into()),
                    accesses: vec![],
                });
                self.ir.scopes[parent.0]
                    .children
                    .push(ScopeItem::Statement(sid));
                let rhs = self.expression(value, sid)?;
                self.ir.statements[sid.0].expression = rhs;
                self.access(destination, sid, AccessKind::Write)?;
            }
        }
        Ok(())
    }
    fn access(
        &mut self,
        a: &crate::TensorAccess,
        s: StatementId,
        kind: AccessKind,
    ) -> Result<AccessId, ResolveError> {
        let scope = self.ir.statement(s).scope;
        let tid = TensorId(a.value.index());
        let value = self.plan.value_instance(a.value).unwrap();
        let shape = a.view_dimensions.as_deref().or_else(|| {
            if a.view_shape.is_none() {
                value.dimensions()
            } else {
                None
            }
        });
        let shape = shape
            .map(|dims| dims.iter().map(|i| self.index(i)).collect())
            .unwrap_or_else(|| {
                a.shape(value.shape())
                    .iter()
                    .map(|n| IndexExpr::Integer(*n as i64))
                    .collect::<Vec<_>>()
            });
        let indices = a
            .indices
            .iter()
            .map(|axis| {
                Ok(match axis {
                    A::FullTile => IndexDim::FullTile,
                    A::Tile { variable, width } | A::ClippedTile { variable, width } => {
                        IndexDim::Tile {
                            start: self.index(&I::Variable(variable.clone())),
                            width: Self::width(width),
                        }
                    }
                    A::Elem(v) => IndexDim::Elem(self.index(&I::Variable(v.clone()))),
                    A::Element(i) => IndexDim::Elem(self.index(i)),
                    A::Slice { start, width } => IndexDim::ConstTile {
                        start: self.index(start),
                        width: Self::width(width),
                    },
                })
            })
            .collect::<Result<Vec<_>, ResolveError>>()?;
        let id = AccessId(self.ir.accesses.len());
        self.ir.accesses.push(AccessInfo {
            tensor: tid,
            kind,
            statement: s,
            scope,
            index: indices,
            view_axes: Some((0..shape.len()).map(|i| format!("d{i}")).collect()),
            view_shape: Some(shape),
            source_span: None,
        });
        self.ir.statements[s.0].accesses.push(id);
        let scope = &mut self.ir.scopes[scope.0];
        let kernel = &mut self.ir.kernels[scope.kernel.0];
        scope.accesses.push(id);
        kernel.accesses.push(id);
        for rw in [&mut scope.read_writes, &mut kernel.read_writes] {
            match kind {
                AccessKind::Read => {
                    rw.reads.insert(tid);
                }
                AccessKind::Write => {
                    rw.writes.insert(tid);
                }
            }
        }
        Ok(id)
    }
    fn expression(&mut self, e: &E, s: StatementId) -> Result<ValueExpr, ResolveError> {
        let apply = |op: &str, args| ValueExpr::Apply(op.into(), args);
        Ok(match e {
            E::Load(a) => ValueExpr::Load(self.access(a, s, AccessKind::Read)?),
            E::Index(i) => ValueExpr::Index(self.index(i)),
            E::Constant(c) => ValueExpr::Literal(match c {
                crate::Constant::Integer(n) => n.to_string(),
                crate::Constant::Float32(bits) => format!("{:?}", f32::from_bits(*bits)),
                crate::Constant::Float64(bits) => format!("{:?}", f64::from_bits(*bits)),
            }),
            E::Add(a) | E::Sub(a) | E::Mul(a) | E::Div(a) | E::Matmul(a) => apply(
                match e {
                    E::Add(_) => "+",
                    E::Sub(_) => "-",
                    E::Mul(_) => "*",
                    E::Div(_) => "/",
                    _ => "@",
                },
                vec![self.expression(&a[0], s)?, self.expression(&a[1], s)?],
            ),
            E::Sqr(v) | E::Sqrt(v) | E::Sigmoid(v) => apply(
                match e {
                    E::Sqr(_) => "sqr",
                    E::Sqrt(_) => "sqrt",
                    _ => "sigmoid",
                },
                vec![self.expression(v, s)?],
            ),
            E::Relu(v) => apply(
                "max",
                vec![self.expression(v, s)?, ValueExpr::Literal("0".into())],
            ),
            E::ReduceSum { value, axis }
            | E::Broadcast { value, axis }
            | E::Unsqueeze { value, axis } => apply(
                match e {
                    E::ReduceSum { .. } => "rsum",
                    E::Broadcast { .. } => "bcast",
                    _ => "unsqueeze",
                },
                vec![
                    self.expression(value, s)?,
                    ValueExpr::Literal(axis.to_string()),
                ],
            ),
            E::Apply { op, args } => {
                let mut args = args
                    .iter()
                    .map(|e| self.expression(e, s))
                    .collect::<Result<Vec<_>, _>>()?;
                let op = match op {
                    ValueOp::Exp => "exp",
                    ValueOp::Erf => "erf",
                    ValueOp::Abs => "abs",
                    ValueOp::Transpose => "transpose",
                    ValueOp::Permute => "permute",
                    ValueOp::ReduceMax => "rmax",
                    ValueOp::ReduceMin => "rmin",
                    ValueOp::ReduceSum => "rsum",
                    ValueOp::Broadcast => "bcast",
                    ValueOp::Unsqueeze => "unsqueeze",
                    ValueOp::Squeeze => "squeeze",
                    ValueOp::Concat => "concat",
                    ValueOp::LessEqual => "<=",
                    ValueOp::Maximum => "max",
                    ValueOp::Minimum => "min",
                    ValueOp::Cast(dtype) => {
                        args.insert(0, ValueExpr::Index(IndexExpr::Symbol(dtype.clone())));
                        "cast"
                    }
                };
                apply(op, args)
            }
            _ => return Err(invalid("memory effect nested in computation")),
        })
    }
}
