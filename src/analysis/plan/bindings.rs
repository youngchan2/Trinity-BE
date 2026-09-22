//! Bind configuration symbols without changing the scheduled computation.
use super::{
    AccessIndex, Expression, IndexExpr, PhysicalInvariantError, PhysicalPlan, Statement,
    TensorAccess, TileWidth,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn free_symbols(expr: &IndexExpr, scope: &BTreeSet<String>, out: &mut BTreeSet<String>) {
    match expr {
        IndexExpr::Variable(name) if !scope.contains(name) => {
            out.insert(name.clone());
        }
        IndexExpr::Add(a, b)
        | IndexExpr::Sub(a, b)
        | IndexExpr::Mul(a, b)
        | IndexExpr::Div(a, b) => {
            free_symbols(a, scope, out);
            free_symbols(b, scope, out);
        }
        _ => {}
    }
}

fn bind_index(
    expr: &mut IndexExpr,
    scope: &BTreeSet<String>,
    bindings: &BTreeMap<String, i64>,
) -> Result<(), String> {
    match expr {
        IndexExpr::Variable(name) if !scope.contains(name) => {
            *expr = IndexExpr::Constant(
                *bindings
                    .get(name)
                    .ok_or_else(|| format!("unbound configuration symbol {name}"))?,
            );
        }
        IndexExpr::Add(a, b)
        | IndexExpr::Sub(a, b)
        | IndexExpr::Mul(a, b)
        | IndexExpr::Div(a, b) => {
            bind_index(a, scope, bindings)?;
            bind_index(b, scope, bindings)?;
        }
        _ => {}
    }
    // Keep expressions containing induction variables; fold constant arithmetic.
    let mut names = BTreeSet::new();
    free_symbols(expr, &BTreeSet::new(), &mut names);
    if names.is_empty() {
        *expr = IndexExpr::Constant(expr.evaluate(&BTreeMap::new())?);
    }
    Ok(())
}

fn bind_access(
    access: &mut TensorAccess,
    scope: &BTreeSet<String>,
    bindings: &BTreeMap<String, i64>,
) -> Result<(), String> {
    if let Some(dims) = &access.view_dimensions {
        access.view_shape = Some(
            dims.iter()
                .map(|d| {
                    usize::try_from(d.evaluate(bindings)?)
                        .ok()
                        .filter(|n| *n > 0)
                        .ok_or_else(|| "invalid view dimension".into())
                })
                .collect::<Result<_, String>>()?,
        );
        access.view_dimensions = None;
    }
    for index in &mut access.indices {
        if let AccessIndex::Slice { start, .. } | AccessIndex::Element(start) = index {
            bind_index(start, scope, bindings)?;
        }
        if let AccessIndex::Slice { width, .. }
        | AccessIndex::Tile { width, .. }
        | AccessIndex::ClippedTile { width, .. } = index
        {
            *width = TileWidth::Constant(width.resolve(bindings)?);
        }
    }
    Ok(())
}

fn bind_expression(
    expr: &mut Expression,
    scope: &BTreeSet<String>,
    bindings: &BTreeMap<String, i64>,
) -> Result<(), String> {
    match expr {
        Expression::Load(access)
        | Expression::Store {
            destination: access,
            ..
        } => bind_access(access, scope, bindings)?,
        Expression::Index(i) => bind_index(i, scope, bindings)?,
        Expression::AllGather {
            source,
            destination,
            ..
        } => {
            bind_access(source, scope, bindings)?;
            bind_access(destination, scope, bindings)?;
        }
        _ => {}
    }
    for child in expr.children_mut() {
        bind_expression(child, scope, bindings)?;
    }
    Ok(())
}

impl PhysicalPlan {
    /// Symbolic configuration parameters, including those with sample bindings.
    /// Loop induction variables are excluded; bind_symbols makes a concrete plan.
    pub fn symbols(&self) -> BTreeSet<String> {
        fn visit(
            plan: &PhysicalPlan,
            nodes: &[Statement],
            scope: &BTreeSet<String>,
            out: &mut BTreeSet<String>,
        ) {
            for node in nodes {
                if let Statement::Region(body) = node {
                    visit(plan, body, scope, out);
                }
                if let Statement::Loop(l) = node {
                    for expr in [&l.domain.start, &l.domain.stop, &l.domain.step] {
                        free_symbols(expr, scope, out);
                    }
                    let mut nested = scope.clone();
                    nested.insert(l.domain.variable.clone());
                    visit(plan, &l.body, &nested, out);
                }
                if let Statement::Operation(id) = node {
                    fn expr(e: &Expression, scope: &BTreeSet<String>, out: &mut BTreeSet<String>) {
                        if let Expression::Index(i) = e {
                            free_symbols(i, scope, out);
                        }
                        for child in e.children() {
                            expr(child, scope, out);
                        }
                    }
                    let e = plan.operation(*id).unwrap().expression();
                    expr(e, scope, out);
                    for access in e.accesses() {
                        if let Some(dims) = &access.view_dimensions {
                            for d in dims {
                                free_symbols(d, scope, out);
                            }
                        }
                        for i in &access.indices {
                            if let AccessIndex::Slice { start, .. } | AccessIndex::Element(start) =
                                i
                            {
                                free_symbols(start, scope, out);
                            }
                        }
                    }
                }
            }
        }
        let mut out = BTreeSet::new();
        visit(self, &self.statements, &BTreeSet::new(), &mut out);
        for (_, v) in self.value_instances() {
            if let Some(dims) = v.dimensions() {
                for d in dims {
                    free_symbols(d, &BTreeSet::new(), &mut out);
                }
            }
        }
        for (_, op) in self.operations() {
            for access in op.expression.accesses() {
                for index in &access.indices {
                    if let AccessIndex::Slice {
                        width: TileWidth::Symbol(name),
                        ..
                    }
                    | AccessIndex::Tile {
                        width: TileWidth::Symbol(name),
                        ..
                    }
                    | AccessIndex::ClippedTile {
                        width: TileWidth::Symbol(name),
                        ..
                    } = index
                    {
                        out.insert(name.clone());
                    }
                }
            }
        }
        out
    }

    /// Return a concrete candidate plan. The original plan retains its symbols.
    /// Bind the same parameter in tile widths and loop ranges together. This does
    /// not tune, retile fixed widths, change tensor shapes, or reorder statements.
    pub fn bind_symbols(
        &self,
        bindings: &BTreeMap<String, i64>,
    ) -> Result<Self, PhysicalInvariantError> {
        fn visit(
            nodes: &mut [Statement],
            ops: &mut [super::Operation],
            scope: &BTreeSet<String>,
            bindings: &BTreeMap<String, i64>,
        ) -> Result<(), String> {
            for node in nodes {
                if let Statement::Region(body) = node {
                    visit(body, ops, scope, bindings)?;
                }
                if let Statement::Operation(id) = node {
                    bind_expression(&mut ops[id.index()].expression, scope, bindings)?;
                }
                if let Statement::Loop(l) = node {
                    for expr in [&mut l.domain.start, &mut l.domain.stop, &mut l.domain.step] {
                        bind_index(expr, scope, bindings)?;
                    }
                    let mut nested = scope.clone();
                    nested.insert(l.domain.variable.clone());
                    visit(&mut l.body, ops, &nested, bindings)?;
                }
            }
            Ok(())
        }
        let mut plan = self.clone();
        let mut all = self.bindings.clone();
        all.extend(bindings.clone());
        for v in &mut plan.value_instances.values {
            if let Some(dims) = &v.dimensions {
                v.shape = dims
                    .iter()
                    .map(|d| {
                        d.evaluate(&all).and_then(|n| {
                            usize::try_from(n)
                                .ok()
                                .filter(|n| *n > 0)
                                .ok_or_else(|| "invalid tensor dimension".into())
                        })
                    })
                    .collect::<Result<_, _>>()
                    .map_err(PhysicalInvariantError::InvalidProgram)?;
                v.dimensions = None;
            }
        }
        visit(
            &mut plan.statements,
            &mut plan.operations.values,
            &BTreeSet::new(),
            &all,
        )
        .map_err(PhysicalInvariantError::InvalidProgram)?;
        for op in &mut plan.operations.values {
            for access in op.expression.accesses() {
                access
                    .validate_view(plan.value_instances.values[access.value.index()].shape())
                    .map_err(PhysicalInvariantError::InvalidProgram)?;
            }
            op.expression.map_accesses(&mut |access| {
                if access.view_shape.as_deref()
                    == Some(plan.value_instances.values[access.value.index()].shape())
                {
                    access.view_shape = None;
                }
            });
        }
        plan.bindings.clear();
        Ok(super::normalize::canonicalize(plan))
    }
}
