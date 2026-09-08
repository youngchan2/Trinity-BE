use std::collections::{BTreeMap, BTreeSet};

use super::shape::{constant, loop_range, product, tile};
use super::{Error, Options, TileAccess, invalid};
use crate::analyzer::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Register,
    Global,
    /// Existing backend's cross-sloop tensor, supplied by the caller in fp16.
    Materialized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialValue {
    Zero,
    Global,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Initialization {
    pub scope: ScopeId,
    pub access: AccessId,
    pub value: InitialValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlan {
    pub storage: Storage,
    pub representative: AccessId,
    pub initialization: Option<Initialization>,
    /// A register's final store executes at the end of this scope.
    pub export_scope: Option<ScopeId>,
    /// Value is visible outside this kernel.
    pub publish: bool,
    /// Additive recurrences, separately from ordinary assignments/epilogues.
    pub accumulators: BTreeSet<StatementId>,
}

impl TensorPlan {
    pub(crate) fn has_global(&self) -> bool {
        self.storage != Storage::Register
            || self.publish
            || self
                .initialization
                .as_ref()
                .is_some_and(|i| i.value == InitialValue::Global)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    pub root_scope: ScopeId,
    pub parallel_loops: Vec<ScopeId>,
    pub grid: Vec<usize>,
    pub tensors: BTreeMap<TensorId, TensorPlan>,
    /// Per-occurrence binding: an accumulator can be local while later sibling
    /// loops read the same tensor through global memory.
    pub register_accesses: BTreeSet<AccessId>,
}

#[derive(Debug, Clone)]
pub struct TritonPlan {
    pub(crate) analysis: ProgramAnalysis,
    pub(crate) options: Options,
    pub(crate) kernels: Vec<KernelPlan>,
    pub(crate) accesses: Vec<TileAccess>,
    pub(crate) expressions: Vec<Expr>,
    pub(crate) globals: BTreeSet<TensorId>,
}

impl TritonPlan {
    pub fn analysis(&self) -> &ProgramAnalysis {
        &self.analysis
    }
    pub fn kernels(&self) -> &[KernelPlan] {
        &self.kernels
    }
    pub fn accesses(&self) -> &[TileAccess] {
        &self.accesses
    }
    pub fn options(&self) -> &Options {
        &self.options
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Expr {
    pub kind: ExprKind,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) enum ExprKind {
    Scalar(String),
    Index(IndexExpr),
    Load(AccessId),
    Unary(String, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Reduce(String, usize, Box<Expr>),
    Permute(Vec<usize>, Box<Expr>),
    Transform(String, usize, Box<Expr>),
    Dot(Box<Expr>, Box<Expr>),
}

/// Resolve shapes, reaching definitions, storage and initialization without
/// changing the optimizer's schedule. The returned plan is immutable to emitters.
pub fn lower(analysis: ProgramAnalysis, options: Options) -> Result<TritonPlan, Error> {
    for tensor in analysis.tensors() {
        if !tensor
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(invalid(
                "tensor names must contain only ASCII letters, digits and underscores",
            ));
        }
    }
    for shape in options.shapes.values() {
        product(shape)?;
    }
    let accesses = analysis
        .accesses()
        .iter()
        .map(|a| tile(a, &analysis, &options))
        .collect::<Result<Vec<_>, _>>()?;
    let expressions = analysis
        .statements()
        .iter()
        .map(|s| expression(&s.expression, &accesses, &options))
        .collect::<Result<Vec<_>, _>>()?;
    for (id, statement) in analysis.statements().iter().enumerate() {
        let write = statement.accesses.last().unwrap();
        let expected = &accesses[write.index()].shape;
        if broadcast(&expressions[id].shape, expected)? != *expected {
            return Err(invalid(format!(
                "statement {id}: result {:?} cannot store into {expected:?}",
                expressions[id].shape
            )));
        }
    }
    let mut plan = TritonPlan {
        analysis,
        options,
        kernels: Vec::new(),
        accesses,
        expressions,
        globals: BTreeSet::new(),
    };
    let ir = &plan.analysis;
    let mut previously_written = BTreeSet::new();
    let mut previous_definitions: Vec<AccessId> = Vec::new();
    for (ki, kernel) in ir.kernels().iter().enumerate() {
        let mut parallel = Vec::new();
        for (si, scope) in ir
            .scopes()
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kernel.index() == ki && s.loop_info.is_some())
        {
            loop_range(ir, ScopeId(si), &plan.options)?;
            if scope.kind == ScopeKind::ParallelLoop {
                parallel.push(ScopeId(si));
            }
        }
        // Parallel loops must be one enclosing chain. Sibling parallel regions
        // need a kernel split, which belongs to the optimizer/adapter.
        for pair in parallel.windows(2) {
            if !ir.is_within(pair[1], pair[0]) {
                return Err(invalid("sibling ploop regions in one kernel"));
            }
        }
        if parallel.len() > 3 {
            return Err(invalid("Triton supports at most three grid axes"));
        }
        if let Some(last) = parallel.last() {
            for access in &kernel.accesses {
                if !ir.is_within(ir.access(*access).scope, *last) {
                    return Err(invalid(
                        "all statements must be inside the kernel's ploop chain",
                    ));
                }
            }
            let mut current = ir.scope(*last).parent;
            while let Some(id) = current {
                if ir.scope(id).kind == ScopeKind::SequentialLoop {
                    return Err(invalid(
                        "ploop nested in sloop needs an explicit kernel split",
                    ));
                }
                current = ir.scope(id).parent;
            }
        }
        let grid = parallel
            .iter()
            .map(|id| {
                let (start, end, step) = loop_range(ir, *id, &plan.options)?;
                Ok(((end - start) as usize).div_ceil(step))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        product(&grid)?;
        let dependencies = value_dependencies(ir, kernel);
        let mut tensors = BTreeMap::new();
        let names: BTreeSet<_> = kernel
            .read_writes
            .reads
            .union(&kernel.read_writes.writes)
            .copied()
            .collect();
        for tensor in names {
            let uses: Vec<_> = kernel
                .accesses
                .iter()
                .copied()
                .filter(|a| ir.access(*a).tensor == tensor)
                .collect();
            let writes: Vec<_> = uses
                .iter()
                .copied()
                .filter(|a| ir.access(*a).kind == AccessKind::Write)
                .collect();
            let input = ir.tensor(tensor).declarations.contains(&TensorKind::Input);
            let publish = input
                || ir.tensor(tensor).declarations.contains(&TensorKind::Output)
                || ir.kernels()[ki + 1..]
                    .iter()
                    .any(|k| k.read_writes.reads.contains(&tensor));
            let representative = writes.first().copied().unwrap_or(uses[0]);
            let all_same = uses
                .iter()
                .all(|a| same_region(ir.access(*a), ir.access(representative)));
            let storage = if input || writes.is_empty() {
                Storage::Global
            } else if all_same {
                Storage::Register
            } else {
                Storage::Materialized
            };
            if storage != Storage::Register || publish {
                plan.globals.insert(tensor);
            }
            if writes.is_empty() && !input && !previously_written.contains(&tensor) {
                return Err(invalid(format!(
                    "{}: read before any producer",
                    ir.tensor(tensor).name
                )));
            }
            let first = ir.access(uses[0]);
            let mut initialization = None;
            if !writes.is_empty() && !input && first.kind == AccessKind::Read {
                let value = if previously_written.contains(&tensor) {
                    InitialValue::Global
                } else if additive_update(ir, first.statement, representative) {
                    InitialValue::Zero
                } else {
                    return Err(invalid(format!(
                        "{}: first read is not a defined value or additive accumulator",
                        ir.tensor(tensor).name
                    )));
                };
                let family: Vec<_> = uses
                    .iter()
                    .copied()
                    .filter(|a| same_region(ir.access(*a), ir.access(representative)))
                    .collect();
                let mut scope = common_scope(ir, &family);
                let first_write = ir.access(representative);
                if scope == first_write.scope {
                    let deps: BTreeSet<_> = first_write
                        .index
                        .iter()
                        .flat_map(IndexDim::loop_dependencies)
                        .collect();
                    while ir.scope(scope).kind == ScopeKind::SequentialLoop
                        && !deps.contains(&scope)
                    {
                        scope = ir.scope(scope).parent.unwrap();
                    }
                }
                initialization = Some(Initialization {
                    scope,
                    access: representative,
                    value,
                });
                if value == InitialValue::Global {
                    plan.globals.insert(tensor);
                }
            }
            if initialization.is_none() && storage == Storage::Register {
                let common = common_scope(ir, &uses);
                if common != ir.access(representative).scope {
                    // Triton needs an incoming SSA value when a value defined in
                    // a loop is used after it, even for a statically nonempty loop.
                    initialization = Some(Initialization {
                        scope: common,
                        access: representative,
                        value: InitialValue::Zero,
                    });
                }
            }
            if storage == Storage::Materialized {
                validate_materialized_reads(
                    ir,
                    &uses,
                    representative,
                    initialization.as_ref(),
                    &plan.options,
                )?;
            }
            if !input {
                let global_reads: Vec<_> = if storage == Storage::Global {
                    uses.iter()
                        .copied()
                        .filter(|a| ir.access(*a).kind == AccessKind::Read)
                        .collect()
                } else if initialization
                    .as_ref()
                    .is_some_and(|i| i.value == InitialValue::Global)
                {
                    vec![representative]
                } else {
                    Vec::new()
                };
                for read in global_reads {
                    if !previous_definitions.iter().any(|id| {
                        ir.access(*id).tensor == tensor
                            && covers(ir, ir.access(*id), ir.access(read), &plan.options)
                    }) {
                        return Err(invalid(format!(
                            "{}: global read is not covered by an earlier kernel's writes",
                            ir.tensor(tensor).name
                        )));
                    }
                }
            }
            if input && !writes.is_empty() {
                for read in uses
                    .iter()
                    .filter(|a| ir.access(**a).kind == AccessKind::Read)
                {
                    if !unowned_axes(ir, ir.access(*read), &parallel, &plan.options)?.is_empty() {
                        return Err(invalid(
                            "a mutated input is read across program ownership boundaries",
                        ));
                    }
                }
            }
            let export_scope = if (storage == Storage::Register && publish)
                || (storage == Storage::Materialized && initialization.is_some())
            {
                Some(
                    initialization
                        .as_ref()
                        .map(|i| i.scope)
                        .unwrap_or_else(|| common_scope(ir, &uses)),
                )
            } else {
                None
            };
            // A promoted accumulator cannot outlive its tile coordinate.
            if let Some(scope) = export_scope {
                for dep in ir
                    .access(representative)
                    .index
                    .iter()
                    .flat_map(IndexDim::loop_dependencies)
                {
                    if !ir.is_within(scope, dep) {
                        return Err(invalid("register export escapes its tile coordinate"));
                    }
                }
            }
            if publish || storage == Storage::Materialized {
                for write in &writes {
                    for axis in unowned_axes(ir, ir.access(*write), &parallel, &plan.options)? {
                        if input || dependencies[&tensor].contains(&axis) {
                            return Err(invalid(format!(
                                "{}: global write is not proven disjoint or invariant across ploop {}",
                                ir.tensor(tensor).name,
                                ir.scope(axis).loop_info.as_ref().unwrap().variable
                            )));
                        }
                    }
                }
            }
            let accumulators = writes
                .iter()
                .filter_map(|w| {
                    let s = ir.access(*w).statement;
                    additive_update(ir, s, *w).then_some(s)
                })
                .collect();
            tensors.insert(
                tensor,
                TensorPlan {
                    storage,
                    representative,
                    initialization,
                    export_scope,
                    publish,
                    accumulators,
                },
            );
        }
        previously_written.extend(kernel.read_writes.writes.iter().copied());
        previous_definitions.extend(
            kernel
                .accesses
                .iter()
                .copied()
                .filter(|a| ir.access(*a).kind == AccessKind::Write),
        );
        let mut register_accesses = BTreeSet::new();
        for (tensor, tp) in &mut tensors {
            let uses: Vec<_> = kernel
                .accesses
                .iter()
                .copied()
                .filter(|a| ir.access(*a).tensor == *tensor)
                .collect();
            // A plain output store does not need an invented local variable.
            let direct_store = tp.storage == Storage::Register
                && tp.publish
                && tp.initialization.is_none()
                && !uses.iter().any(|a| ir.access(*a).kind == AccessKind::Read);
            if direct_store {
                tp.export_scope = None;
            }
            for access in uses {
                if (tp.storage == Storage::Register && !direct_store)
                    || (tp.storage == Storage::Materialized
                        && tp.initialization.is_some()
                        && same_region(ir.access(access), ir.access(tp.representative)))
                {
                    register_accesses.insert(access);
                }
            }
        }
        plan.kernels.push(KernelPlan {
            root_scope: kernel.root_scope,
            parallel_loops: parallel,
            grid,
            tensors,
            register_accesses,
        });
    }
    Ok(plan)
}

fn common_scope(ir: &ProgramAnalysis, uses: &[AccessId]) -> ScopeId {
    let mut scope = ir.access(uses[0]).scope;
    for access in &uses[1..] {
        while !ir.is_within(ir.access(*access).scope, scope) {
            scope = ir.scope(scope).parent.unwrap();
        }
    }
    scope
}

fn additive_update(ir: &ProgramAnalysis, statement: StatementId, write: AccessId) -> bool {
    let target = ir.access(write);
    fn self_term(expr: &ValueExpr, ir: &ProgramAnalysis, target: &AccessInfo) -> bool {
        match expr {
            ValueExpr::Load(id) => {
                let a = ir.access(*id);
                a.tensor == target.tensor && same_region(a, target)
            }
            ValueExpr::Apply(op, args) if op == "*" && args.len() == 2 => {
                (matches!(&args[0], ValueExpr::Literal(v) if v.parse::<f64>().is_ok_and(f64::is_finite))
                    && self_term(&args[1], ir, target))
                    || (matches!(&args[1], ValueExpr::Literal(v) if v.parse::<f64>().is_ok_and(f64::is_finite))
                        && self_term(&args[0], ir, target))
            }
            _ => false,
        }
    }
    let s = ir.statement(statement);
    if s.accesses.last() != Some(&write) {
        return false;
    }
    match &s.expression {
        ValueExpr::Apply(op, args) if op == "+" && args.len() == 2 => {
            self_term(&args[0], ir, target) || self_term(&args[1], ir, target)
        }
        _ => false,
    }
}

fn unowned_axes(
    ir: &ProgramAnalysis,
    write: &AccessInfo,
    parallel: &[ScopeId],
    options: &Options,
) -> Result<BTreeSet<ScopeId>, Error> {
    let mut result = BTreeSet::new();
    for id in parallel {
        let (_, _, step) = loop_range(ir, *id, options)?;
        let owns = write.index.iter().any(|d| match d {
            IndexDim::Tile {
                start: IndexExpr::LoopVar(s),
                width,
            } if s == id => constant(width, options).is_ok_and(|w| w > 0 && w as usize <= step),
            IndexDim::Elem(IndexExpr::LoopVar(s)) => s == id,
            _ => false,
        });
        if !owns {
            result.insert(*id);
        }
    }
    Ok(result)
}

fn value_dependencies(
    ir: &ProgramAnalysis,
    kernel: &KernelInfo,
) -> BTreeMap<TensorId, BTreeSet<ScopeId>> {
    fn expr_deps(
        expr: &ValueExpr,
        ir: &ProgramAnalysis,
        deps: &BTreeMap<TensorId, BTreeSet<ScopeId>>,
    ) -> BTreeSet<ScopeId> {
        match expr {
            ValueExpr::Literal(_) => BTreeSet::new(),
            ValueExpr::Index(i) => i.loop_dependencies(),
            ValueExpr::Apply(_, args) => args.iter().flat_map(|a| expr_deps(a, ir, deps)).collect(),
            ValueExpr::Load(id) => {
                let a = ir.access(*id);
                a.index
                    .iter()
                    .flat_map(IndexDim::loop_dependencies)
                    .chain(deps[&a.tensor].iter().copied())
                    .collect()
            }
        }
    }
    let mut deps: BTreeMap<_, BTreeSet<_>> = kernel
        .read_writes
        .reads
        .union(&kernel.read_writes.writes)
        .map(|t| (*t, BTreeSet::new()))
        .collect();
    loop {
        let mut changed = false;
        for id in &kernel.accesses {
            let a = ir.access(*id);
            if a.kind == AccessKind::Write {
                let incoming = expr_deps(&ir.statement(a.statement).expression, ir, &deps);
                let target = deps.get_mut(&a.tensor).unwrap();
                let old = target.len();
                target.extend(incoming);
                changed |= target.len() != old;
            }
        }
        if !changed {
            return deps;
        }
    }
}

// Prove coverage for the common dense producer-loop / consumer-loop case. Do
// not equate loop variables by their spelling, nor assume sparse tiles cover a
// full tensor. Unknown region relations are rejected rather than zero-filled.
fn validate_materialized_reads(
    ir: &ProgramAnalysis,
    uses: &[AccessId],
    representative: AccessId,
    init: Option<&Initialization>,
    options: &Options,
) -> Result<(), Error> {
    let mut definitions = Vec::new();
    if init.is_some() {
        definitions.push(representative);
    }
    for id in uses {
        let a = ir.access(*id);
        if a.kind == AccessKind::Write {
            definitions.push(*id);
            continue;
        }
        if !definitions
            .iter()
            .any(|w| covers(ir, ir.access(*w), a, options))
        {
            return Err(invalid(format!(
                "{}: materialized read {:?} has no covering earlier definition",
                ir.tensor(a.tensor).name,
                a.source_span
            )));
        }
    }
    Ok(())
}

fn covers(ir: &ProgramAnalysis, write: &AccessInfo, read: &AccessInfo, options: &Options) -> bool {
    if write.index.len() != read.index.len() || write.view_shape != read.view_shape {
        return false;
    }
    let mut mapping = BTreeMap::new();
    let mut inverse = BTreeMap::new();
    let cross_kernel = ir.scope(write.scope).kernel != ir.scope(read.scope).kernel;
    write
        .index
        .iter()
        .zip(&read.index)
        .enumerate()
        .all(|(axis, (w, r))| {
            if w == r {
                return true;
            }
            match (w, r) {
                (
                    IndexDim::Tile {
                        start: IndexExpr::LoopVar(ws),
                        width: ww,
                    },
                    IndexDim::Tile {
                        start: IndexExpr::LoopVar(rs),
                        width: rw,
                    },
                ) => {
                    let Ok((w0, w1, step)) = loop_range(ir, *ws, options) else {
                        return false;
                    };
                    let Ok((r0, r1, rstep)) = loop_range(ir, *rs, options) else {
                        return false;
                    };
                    if mapping.insert(*ws, *rs).is_some_and(|old| old != *rs)
                        || inverse.insert(*rs, *ws).is_some_and(|old| old != *ws)
                    {
                        return false;
                    }
                    (cross_kernel
                        || (ir.scope(*ws).kind == ScopeKind::SequentialLoop
                            && ir.scope(*rs).kind == ScopeKind::SequentialLoop))
                        && constant(ww, options).ok() == Some(step as i64)
                        && constant(rw, options).ok() == Some(rstep as i64)
                        && w0 <= r0
                        && w1 >= r1
                        && !ir.is_within(read.scope, *ws)
                }
                (
                    IndexDim::Tile {
                        start: IndexExpr::LoopVar(ws),
                        width,
                    },
                    IndexDim::FullTile,
                ) => {
                    let Ok((start, end, step)) = loop_range(ir, *ws, options) else {
                        return false;
                    };
                    let extent = if let Some(shape) = &write.view_shape {
                        constant(&shape[axis], options).unwrap_or(-1) as usize
                    } else {
                        options.shapes[&ir.tensor(write.tensor).name][axis]
                    };
                    (cross_kernel || ir.scope(*ws).kind == ScopeKind::SequentialLoop)
                        && !ir.is_within(read.scope, *ws)
                        && start == 0
                        && end >= extent as i64
                        && constant(width, options).ok() == Some(step as i64)
                        && write
                            .index
                            .iter()
                            .filter(|d| d.loop_dependencies().contains(ws))
                            .count()
                            == 1
                }
                (IndexDim::FullTile, _) => true,
                (
                    IndexDim::Elem(IndexExpr::LoopVar(ws)),
                    IndexDim::Elem(IndexExpr::LoopVar(rs)),
                ) if cross_kernel => {
                    loop_range(ir, *ws, options).ok() == loop_range(ir, *rs, options).ok()
                }
                _ => false,
            }
        })
}

fn same_region(a: &AccessInfo, b: &AccessInfo) -> bool {
    a.index == b.index && a.view_shape == b.view_shape
}

pub(crate) fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, Error> {
    let rank = a.len().max(b.len());
    let mut result = vec![1; rank];
    for (i, d) in result.iter_mut().enumerate() {
        let x = a.get(i.wrapping_sub(rank - a.len())).copied().unwrap_or(1);
        let y = b.get(i.wrapping_sub(rank - b.len())).copied().unwrap_or(1);
        if x != y && x != 1 && y != 1 {
            return Err(invalid(format!("incompatible broadcast {a:?} and {b:?}")));
        }
        *d = x.max(y);
    }
    Ok(result)
}

fn axis(expr: &ValueExpr, rank: usize) -> Result<usize, Error> {
    let ValueExpr::Literal(n) = expr else {
        return Err(invalid("axis must be an integer literal"));
    };
    let mut n = n
        .parse::<i64>()
        .map_err(|_| invalid("axis must be an integer literal"))?;
    if n < 0 {
        n += rank as i64;
    }
    if n < 0 || n as usize >= rank {
        return Err(invalid("axis out of range"));
    }
    Ok(n as usize)
}

fn expression(expr: &ValueExpr, accesses: &[TileAccess], options: &Options) -> Result<Expr, Error> {
    let ValueExpr::Apply(op, args) = expr else {
        return Ok(match expr {
            ValueExpr::Literal(n) => {
                if !n.parse::<f64>().is_ok_and(f64::is_finite) {
                    return Err(invalid("nonfinite scalar literal"));
                }
                Expr {
                    kind: ExprKind::Scalar(format!("{:?}", n.parse::<f64>().unwrap())),
                    shape: vec![],
                }
            }
            ValueExpr::Index(i) => {
                if i.loop_dependencies().is_empty() {
                    constant(i, options)?;
                }
                Expr {
                    kind: ExprKind::Index(i.clone()),
                    shape: vec![],
                }
            }
            ValueExpr::Load(id) => Expr {
                kind: ExprKind::Load(*id),
                shape: accesses[id.index()].shape.clone(),
            },
            _ => unreachable!(),
        });
    };
    let arity = match op.as_str() {
        "sqr" | "exp" | "sqrt" | "sigmoid" | "erf" | "abs" | "transpose" => 1,
        "+" | "-" | "*" | "/" | "<=" | "max" | "min" | "@" | "rsum" | "rmax" | "rmin" | "bcast"
        | "unsqueeze" | "squeeze" => 2,
        "permute" | "permute3" | "permute4" => args.len(),
        _ => return Err(invalid(format!("unsupported value operator {op}"))),
    };
    if args.len() != arity || args.is_empty() {
        return Err(invalid(format!("{op}: wrong argument count")));
    }
    let left = expression(&args[0], accesses, options)?;
    let mut shape = left.shape.clone();
    let kind = match op.as_str() {
        "sqr" | "exp" | "sqrt" | "sigmoid" | "erf" | "abs" => {
            ExprKind::Unary(op.clone(), Box::new(left))
        }
        "transpose" | "permute" | "permute3" | "permute4" => {
            let order = if op == "transpose" {
                if shape.len() < 2 {
                    return Err(invalid("transpose requires rank >= 2"));
                }
                let mut order: Vec<_> = (0..shape.len()).collect();
                let n = order.len();
                order.swap(n - 1, n - 2);
                order
            } else {
                args[1..]
                    .iter()
                    .map(|a| axis(a, shape.len()))
                    .collect::<Result<Vec<_>, _>>()?
            };
            if order.len() != shape.len()
                || order.iter().copied().collect::<BTreeSet<_>>().len() != shape.len()
            {
                return Err(invalid("invalid permutation"));
            }
            shape = order.iter().map(|i| shape[*i]).collect();
            ExprKind::Permute(order, Box::new(left))
        }
        "bcast" | "unsqueeze" => {
            let ax = axis(&args[1], shape.len() + 1)?;
            shape.insert(ax, 1);
            ExprKind::Transform(op.clone(), ax, Box::new(left))
        }
        "squeeze" => {
            let ax = axis(&args[1], shape.len())?;
            if shape[ax] != 1 {
                return Err(invalid("squeeze can only remove a singleton tile axis"));
            }
            shape.remove(ax);
            ExprKind::Transform(op.clone(), ax, Box::new(left))
        }
        "rsum" | "rmax" | "rmin" => {
            let ax = axis(&args[1], shape.len())?;
            shape.remove(ax);
            ExprKind::Reduce(op.clone(), ax, Box::new(left))
        }
        _ => {
            let right = expression(&args[1], accesses, options)?;
            if op == "@" {
                let n = left.shape.len();
                if !(2..=3).contains(&n)
                    || right.shape.len() != n
                    || left.shape[n - 1] != right.shape[n - 2]
                {
                    return Err(invalid(format!(
                        "unsupported dot shapes {:?}, {:?}",
                        left.shape, right.shape
                    )));
                }
                shape = broadcast(&left.shape[..n - 2], &right.shape[..n - 2])?;
                shape.extend([left.shape[n - 2], right.shape[n - 1]]);
                if left.shape[n - 2] < 16 || left.shape[n - 1] < 16 || right.shape[n - 1] < 16 {
                    let mut expanded = shape.clone();
                    expanded.push(left.shape[n - 1]);
                    if product(&expanded)? > 1_048_576 {
                        return Err(invalid("small-dot fallback exceeds Triton's element limit"));
                    }
                }
                ExprKind::Dot(Box::new(left), Box::new(right))
            } else {
                shape = broadcast(&left.shape, &right.shape)?;
                ExprKind::Binary(op.clone(), Box::new(left), Box::new(right))
            }
        }
    };
    Ok(Expr { kind, shape })
}
