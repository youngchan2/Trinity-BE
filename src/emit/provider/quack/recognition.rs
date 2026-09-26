//! Quack-owned computation recognizers and diagnostics for scheduled kernel regions.
//! Leading ploop chains share one classification. Original GEMM expressions are
//! borrowed; additional whole-domain summaries are cloned without rewriting the
//! plan's expressions, stores or schedule.
//! These summaries are Quack diagnostics, not a prerequisite of common planning
//! or another provider. Identity, storage, scopes and view facts remain shared.

use crate::{
    AccessIndex, Constant, Expression as E, IndexExpr, Loop, LoopKind, OperationId, PhysicalPlan,
    Statement, Storage, TensorAccess, ValueInstanceId, ValueOp,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

mod normalization;
mod region;
pub mod rope;
use crate::analysis::regions::{OperationScope, RegionScope, operation_scopes};
use crate::analysis::views::{TensorView, tensor_view};
pub use normalization::Normalization;
pub use region::{DomainStore, domain_stores, summarize};
pub use region::{RegionComputation, RegionOperation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuackPatternKind {
    SingleGemm,
    GemmEpilogue,
    GatedGemm,
    RmsNorm,
    LayerNorm,
    Softmax,
    /// Not recognized as a supported whole-region pattern; not invalid IR.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmInitialization {
    ZeroRecurrence,
    ExplicitZero(OperationId),
}

/// One whole-region GEMM, possibly accumulated across a K loop and followed by
/// multiple pointwise stores. Store boundaries, casts, accesses and loops remain
/// in the original plan; epilogue_operations lists them instead of inlining them.
#[derive(Debug, Clone)]
pub struct GemmPattern<'plan> {
    kind: QuackPatternKind,
    pub matmul_operation: OperationId,
    pub accumulation_operation: Option<OperationId>,
    pub reduction_loop: Option<&'plan Loop>,
    pub initialization: Option<GemmInitialization>,
    pub accumulator: &'plan TensorAccess,
    pub epilogue_operations: Vec<OperationId>,
    output: &'plan TensorAccess,
    value: &'plan E,
    matmul: &'plan E,
}

impl<'plan> GemmPattern<'plan> {
    pub fn kind(&self) -> QuackPatternKind {
        self.kind
    }
    pub fn output(&self) -> &'plan TensorAccess {
        self.output
    }
    /// Original final store RHS, not an expanded cross-store expression tree.
    pub fn value(&self) -> &'plan E {
        self.value
    }
    pub fn matmul(&self) -> &'plan E {
        self.matmul
    }
    pub fn operands(&self) -> &'plan [E; 2] {
        let E::Matmul(operands) = self.matmul else {
            unreachable!()
        };
        operands
    }
}

#[derive(Debug, Clone)]
pub struct RegionAnalysis<'plan> {
    pub scope: RegionScope,
    pub kind: QuackPatternKind,
    pub gemm: Option<GemmPattern<'plan>>,
    /// Proven whole-region computation; the original plan remains unchanged.
    pub computation: Option<RegionComputation>,
    pub computation_reason: Option<String>,
    /// Why this complete region was left as Other.
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OperationAnalysis {
    pub scope: OperationScope,
}

#[derive(Debug, Clone, Default)]
pub struct QuackPatternAnalysis<'plan> {
    regions: BTreeMap<Vec<usize>, RegionAnalysis<'plan>>,
    operations: BTreeMap<OperationId, OperationAnalysis>,
}

impl<'plan> QuackPatternAnalysis<'plan> {
    pub fn analyze(plan: &'plan PhysicalPlan) -> Self {
        let mut result = Self {
            operations: operation_scopes(plan)
                .into_iter()
                .map(|(id, scope)| (id, OperationAnalysis { scope }))
                .collect(),
            ..Self::default()
        };
        // Scheduled input already has one Region per kernel. Direct Builder
        // input uses each outer loop / standalone operation as its boundary.
        for (index, statement) in plan.statements().iter().enumerate() {
            let path = vec![index];
            let scope = RegionScope {
                statement_path: path.clone(),
                operations: statement.operations(),
            };
            let summarized = region::summarize(plan, statement, &scope);
            let (computation, computation_reason) = match summarized {
                Ok(c) => (Some(c), None),
                Err(e) => (None, Some(e)),
            };
            let classified = recognize_gemm(plan, statement, &scope);
            let (kind, gemm, reason) = match classified {
                Ok(gemm) => (gemm.kind(), Some(gemm), None),
                Err(reason) => match &computation {
                    Some(c) if c.kind() != QuackPatternKind::Other => (c.kind(), None, None),
                    _ => (QuackPatternKind::Other, None, Some(reason)),
                },
            };
            result.regions.insert(
                path,
                RegionAnalysis {
                    scope,
                    kind,
                    gemm,
                    computation,
                    computation_reason,
                    reason,
                },
            );
        }
        result
    }
    pub fn regions(&self) -> impl Iterator<Item = &RegionAnalysis<'plan>> {
        self.regions.values()
    }
    pub fn region(&self, path: &[usize]) -> Option<&RegionAnalysis<'plan>> {
        self.regions.get(path)
    }
    pub fn operation(&self, id: OperationId) -> Option<&OperationAnalysis> {
        self.operations.get(&id)
    }
    pub fn operations(&self) -> impl Iterator<Item = &OperationAnalysis> {
        self.operations.values()
    }
    pub fn region_for_operation(&self, id: OperationId) -> Option<&RegionAnalysis<'plan>> {
        self.region(&self.operation(id)?.scope.region_path)
    }
}

type MatchResult<T> = Result<T, String>;
fn store(plan: &PhysicalPlan, id: OperationId) -> MatchResult<(&TensorAccess, &E)> {
    match plan.operation(id).unwrap().expression() {
        E::Store { destination, value } => Ok((destination, value)),
        _ => Err("region contains communication or a non-store operation".into()),
    }
}
fn matmul_count(e: &E) -> usize {
    usize::from(matches!(e, E::Matmul(_))) + e.children().iter().map(matmul_count).sum::<usize>()
}
fn scalar(e: &E, expected: f64) -> bool {
    match e {
        E::Constant(Constant::Integer(n)) => *n as f64 == expected,
        E::Constant(Constant::Float32(n)) => f32::from_bits(*n) as f64 == expected,
        E::Constant(Constant::Float64(n)) => f64::from_bits(*n) == expected,
        _ => false,
    }
}
fn unscaled(mut e: &E) -> &E {
    while let E::Mul(args) = e {
        if scalar(&args[0], 1.0) {
            e = &args[1];
        } else if scalar(&args[1], 1.0) {
            e = &args[0];
        } else {
            break;
        }
    }
    e
}
fn same_coordinates(plan: &PhysicalPlan, a: &TensorAccess, b: &TensorAccess) -> bool {
    a.indices == b.indices
        && a.shape(plan.value_instance(a.value).unwrap().shape())
            == b.shape(plan.value_instance(b.value).unwrap().shape())
        && a.view_dimensions == b.view_dimensions
}
fn same_access(plan: &PhysicalPlan, a: &TensorAccess, b: &TensorAccess) -> bool {
    a.value == b.value && same_coordinates(plan, a, b)
}
fn self_load(plan: &PhysicalPlan, e: &E, destination: &TensorAccess) -> bool {
    matches!(unscaled(e), E::Load(a) if same_access(plan, a, destination))
}

pub fn recognize_gemm<'p>(
    plan: &'p PhysicalPlan,
    statement: &'p Statement,
    scope: &RegionScope,
) -> MatchResult<GemmPattern<'p>> {
    let count: usize = scope
        .operations
        .iter()
        .map(|id| matmul_count(plan.operation(*id).unwrap().expression()))
        .sum();
    if count != 1 {
        return Err(format!(
            "region contains {count} Matmul expressions; expected exactly one"
        ));
    }
    let mut body = match statement {
        Statement::Region(body) => body.as_slice(),
        _ => std::slice::from_ref(statement),
    };
    while let [Statement::Loop(l)] = body {
        if l.kind != LoopKind::Parallel {
            break;
        }
        body = &l.body;
    }
    let written: BTreeSet<_> = scope
        .operations
        .iter()
        .map(|id| store(plan, *id).map(|(a, _)| a.value))
        .collect::<MatchResult<_>>()?;
    let mut initialization = None;
    if let Some(Statement::Operation(id)) = body.first() {
        let (access, value) = store(plan, *id)?;
        if scalar(value, 0.0) {
            initialization = Some((*id, access));
            body = &body[1..];
        }
    }
    let (first, rest) = body
        .split_first()
        .ok_or("empty computation after initialization")?;
    let mut covered = BTreeSet::new();
    if let Some((id, _)) = initialization {
        covered.insert(id);
    }
    let mut gemm = match first {
        Statement::Operation(id) => {
            if initialization.is_some() {
                return Err("initialization is not consumed by a GEMM recurrence".into());
            }
            let (output, value) = store(plan, *id)?;
            let mut matmul = None;
            if value.reads().contains(&output.value) || !pointwise_tree(value, &mut matmul) {
                return Err("first computation is not a GEMM with a pointwise epilogue".into());
            }
            let matmul = matmul.ok_or("first computation does not contain the region's GEMM")?;
            covered.insert(*id);
            let epi = !std::ptr::eq(value, matmul);
            GemmPattern {
                kind: if epi {
                    QuackPatternKind::GemmEpilogue
                } else {
                    QuackPatternKind::SingleGemm
                },
                matmul_operation: *id,
                accumulation_operation: None,
                reduction_loop: None,
                initialization: None,
                accumulator: output,
                epilogue_operations: if epi { vec![*id] } else { vec![] },
                output,
                value,
                matmul,
            }
        }
        Statement::Loop(l) if l.kind == LoopKind::Sequential => {
            reduction_gemm(plan, l, initialization, &mut covered)?
        }
        _ => return Err("region has additional parallel/split/nested region structure".into()),
    };
    // No operand may hide another computation in this same region.
    if gemm
        .operands()
        .iter()
        .any(|e| e.reads().iter().any(|v| written.contains(v)))
    {
        return Err("GEMM operand is produced inside the region".into());
    }
    let mut definitions = BTreeMap::new();
    definitions.insert(gemm.output.value, (gemm.output, covered.clone()));
    let mut last_dependencies = covered;
    for statement in rest {
        let Statement::Operation(id) = statement else {
            return Err("epilogue contains another loop or region".into());
        };
        let (output, value) = store(plan, *id)?;
        if !same_coordinates(plan, output, gemm.accumulator) {
            return Err("epilogue changes the GEMM output coordinates".into());
        }
        let mut dependencies = BTreeSet::new();
        let mut arithmetic = false;
        let depends = epilogue_tree(
            plan,
            value,
            &definitions,
            &written,
            &mut dependencies,
            &mut arithmetic,
        )?;
        if !depends {
            return Err("region contains an unrelated computation".into());
        }
        if arithmetic {
            gemm.kind = QuackPatternKind::GemmEpilogue;
        }
        dependencies.insert(*id);
        definitions.insert(output.value, (output, dependencies.clone()));
        last_dependencies = dependencies;
        gemm.epilogue_operations.push(*id);
        gemm.output = output;
        gemm.value = value;
    }
    if last_dependencies != scope.operations.iter().copied().collect() {
        return Err("region contains stores not used by its final result".into());
    }
    // A region with externally visible intermediate results is not a single
    // GEMM output contract. Storage and boundary identities come from the plan.
    for value in written.iter().filter(|v| **v != gemm.output.value) {
        if plan.value_instance(*value).unwrap().storage() != Storage::Register
            || plan.outputs().iter().any(|b| b.value() == *value)
            || plan.mutable_inputs().contains(value)
            || plan.operations().any(|(id, op)| {
                !scope.operations.contains(&id) && op.expression().reads().contains(value)
            })
        {
            return Err("region publishes an additional intermediate result".into());
        }
    }
    Ok(gemm)
}

fn reduction_gemm<'p>(
    plan: &'p PhysicalPlan,
    l: &'p Loop,
    initialization: Option<(OperationId, &'p TensorAccess)>,
    covered: &mut BTreeSet<OperationId>,
) -> MatchResult<GemmPattern<'p>> {
    let (partial, update) = match l.body.as_slice() {
        [Statement::Operation(update)] => (None, *update),
        [Statement::Operation(partial), Statement::Operation(update)] => (Some(*partial), *update),
        _ => {
            return Err(
                "K loop must contain a GEMM accumulation, optionally preceded by its product store"
                    .into(),
            );
        }
    };
    let (accumulator, value) = store(plan, update)?;
    let E::Add(args) = value else {
        return Err("K loop does not contain an additive GEMM recurrence".into());
    };
    let product = if self_load(plan, &args[0], accumulator) {
        unscaled(&args[1])
    } else if self_load(plan, &args[1], accumulator) {
        unscaled(&args[0])
    } else {
        return Err("recurrence must read its own exact destination with coefficient one".into());
    };
    let (matmul_operation, matmul) = if let Some(id) = partial {
        let (access, rhs) = store(plan, id)?;
        if !matches!(product, E::Load(a) if same_access(plan, a, access))
            || !same_coordinates(plan, access, accumulator)
            || plan.value_instance(access.value).unwrap().storage() != Storage::Register
        {
            return Err("partial product store does not feed the accumulator directly".into());
        }
        covered.insert(id);
        (id, rhs)
    } else {
        (update, product)
    };
    let E::Matmul(operands) = matmul else {
        return Err("per-iteration transforms or extra addends are not a GEMM epilogue".into());
    };
    if !operands.iter().all(operand_view) || !contracts_k(plan, l, operands, accumulator) {
        return Err("sequential loop is not a complete, matching GEMM K-axis traversal".into());
    }
    let initialization = if let Some((id, access)) = initialization {
        if !same_access(plan, access, accumulator) {
            return Err("zero initialization targets a different accumulator access".into());
        }
        GemmInitialization::ExplicitZero(id)
    } else {
        let op = plan.operation(update).unwrap();
        // Consume the common plan's zero-recurrence contract, including the
        // direct Builder convention whose self load was removed from inflows.
        if !op.zero_init().contains(&accumulator.value) && op.inflows().contains(&accumulator.value)
        {
            return Err("GEMM recurrence has an external/nonzero initial value".into());
        }
        GemmInitialization::ZeroRecurrence
    };
    covered.insert(update);
    Ok(GemmPattern {
        kind: QuackPatternKind::SingleGemm,
        matmul_operation,
        accumulation_operation: Some(update),
        reduction_loop: Some(l),
        initialization: Some(initialization),
        accumulator,
        epilogue_operations: Vec::new(),
        output: accumulator,
        value,
        matmul,
    })
}

type Definitions<'p> = BTreeMap<ValueInstanceId, (&'p TensorAccess, BTreeSet<OperationId>)>;
fn epilogue_tree(
    plan: &PhysicalPlan,
    e: &E,
    definitions: &Definitions<'_>,
    written: &BTreeSet<ValueInstanceId>,
    dependencies: &mut BTreeSet<OperationId>,
    arithmetic: &mut bool,
) -> MatchResult<bool> {
    match e {
        E::Load(access) => {
            if let Some((definition, ops)) = definitions.get(&access.value) {
                if !same_access(plan, access, definition) {
                    return Err("epilogue reads a different view/tile of a local definition".into());
                }
                dependencies.extend(ops);
                Ok(true)
            } else if written.contains(&access.value) {
                Err("epilogue reads an unavailable local definition".into())
            } else {
                Ok(false)
            }
        }
        E::Constant(_) | E::Index(_) => Ok(false),
        _ => {
            let pointwise = is_pointwise(e);
            let view = operand_view(e);
            if !pointwise && !view {
                return Err("region contains a reduction or unsupported epilogue transform".into());
            }
            let mut dependent = false;
            for child in e.children() {
                dependent |=
                    epilogue_tree(plan, child, definitions, written, dependencies, arithmetic)?;
            }
            if view && dependent {
                return Err("epilogue transforms the GEMM result's axes".into());
            }
            *arithmetic |= pointwise;
            Ok(dependent)
        }
    }
}
fn is_pointwise(e: &E) -> bool {
    matches!(
        e,
        E::Add(_)
            | E::Sub(_)
            | E::Mul(_)
            | E::Div(_)
            | E::Sqr(_)
            | E::Sqrt(_)
            | E::Sigmoid(_)
            | E::Relu(_)
            | E::Apply {
                op: ValueOp::Exp
                    | ValueOp::Erf
                    | ValueOp::Abs
                    | ValueOp::LessEqual
                    | ValueOp::Maximum
                    | ValueOp::Minimum
                    | ValueOp::Cast(_),
                ..
            }
    )
}

/// Recover operand axes through the IR's view expressions solely to prove that
/// the serial variable traverses the contraction axes, never M/N/batch axes.
fn operand_axes(e: &E) -> Option<Vec<Option<(&TensorAccess, usize)>>> {
    match e {
        E::Load(a) => Some((0..a.indices.len()).map(|i| Some((a, i))).collect()),
        E::Broadcast { value, axis } | E::Unsqueeze { value, axis } => {
            let mut axes = operand_axes(value)?;
            if *axis > axes.len() {
                return None;
            }
            axes.insert(*axis, None);
            Some(axes)
        }
        E::Apply { op, args } => {
            let mut axes = operand_axes(args.first()?)?;
            let integer = |e: &E| match e {
                E::Constant(Constant::Integer(i)) => usize::try_from(*i).ok(),
                _ => None,
            };
            match op {
                ValueOp::Transpose if axes.len() >= 2 => {
                    let n = axes.len();
                    axes.swap(n - 1, n - 2);
                }
                ValueOp::Permute => {
                    let order: Vec<_> = args[1..].iter().map(integer).collect::<Option<_>>()?;
                    if order.len() != axes.len()
                        || order.iter().copied().collect::<BTreeSet<_>>()
                            != (0..axes.len()).collect()
                    {
                        return None;
                    }
                    axes = order.into_iter().map(|i| axes[i]).collect();
                }
                ValueOp::Squeeze => {
                    let i = integer(args.get(1)?)?;
                    if i >= axes.len() {
                        return None;
                    }
                    axes.remove(i);
                }
                ValueOp::Unsqueeze | ValueOp::Broadcast => {
                    let i = integer(args.get(1)?)?;
                    if i > axes.len() {
                        return None;
                    }
                    axes.insert(i, None);
                }
                _ => return None,
            }
            Some(axes)
        }
        _ => None,
    }
}
fn uses_index(e: &IndexExpr, variable: &str) -> bool {
    match e {
        IndexExpr::Variable(v) => v == variable,
        IndexExpr::Constant(_) => false,
        IndexExpr::Add(a, b)
        | IndexExpr::Sub(a, b)
        | IndexExpr::Mul(a, b)
        | IndexExpr::Div(a, b) => uses_index(a, variable) || uses_index(b, variable),
    }
}
fn uses_axis(a: &AccessIndex, variable: &str) -> bool {
    match a {
        AccessIndex::Tile { variable: v, .. }
        | AccessIndex::ClippedTile { variable: v, .. }
        | AccessIndex::Elem(v) => v == variable,
        AccessIndex::Slice { start, .. } | AccessIndex::Element(start) => {
            uses_index(start, variable)
        }
        AccessIndex::FullTile => false,
    }
}
fn contracts_k(
    plan: &PhysicalPlan,
    l: &Loop,
    operands: &[E; 2],
    accumulator: &TensorAccess,
) -> bool {
    let variable = &l.domain.variable;
    if accumulator.indices.iter().any(|a| uses_axis(a, variable)) {
        return false;
    }
    let matching = |e: &E, from_end: usize| -> bool {
        let Some(axes) = operand_axes(e) else {
            return false;
        };
        if axes.len() < 2 {
            return false;
        }
        let Some((access, k)) = axes[axes.len() - from_end] else {
            return false;
        };
        let extent = access.shape(plan.value_instance(access.value).unwrap().shape())[k];
        let width = match &access.indices[k] {
            AccessIndex::Tile { variable: v, width }
            | AccessIndex::ClippedTile { variable: v, width }
                if v == variable =>
            {
                width
            }
            AccessIndex::Slice {
                start: IndexExpr::Variable(v),
                width,
            } if v == variable => width,
            _ => return false,
        };
        width.resolve(plan.bindings()).ok().map(|w| w as i64)
            == l.domain.step.evaluate(plan.bindings()).ok()
            && l.domain.start.evaluate(plan.bindings()) == Ok(0)
            && l.domain.stop.evaluate(plan.bindings()) == Ok(extent as i64)
            && e.accesses().iter().all(|a| {
                a.indices
                    .iter()
                    .enumerate()
                    .all(|(i, axis)| !uses_axis(axis, variable) || (*a == access && i == k))
            })
    };
    matching(&operands[0], 1) && matching(&operands[1], 2)
}

fn pointwise_tree<'a>(expression: &'a E, matmul: &mut Option<&'a E>) -> bool {
    match expression {
        E::Matmul(operands) => {
            if matmul.is_some() || !operands.iter().all(operand_view) {
                return false;
            }
            *matmul = Some(expression);
            true
        }
        E::Load(_) | E::Constant(_) | E::Index(_) => true,
        _ if is_pointwise(expression) => expression
            .children()
            .iter()
            .all(|child| pointwise_tree(child, matmul)),
        // Side operands may be broadcast/views. Transforming the GEMM result
        // itself needs a separate output-coordinate contract, not pointwise fusion.
        _ => operand_view(expression),
    }
}

fn operand_view(expression: &E) -> bool {
    match expression {
        E::Load(_) => true,
        E::Broadcast { value, .. } | E::Unsqueeze { value, .. } => operand_view(value),
        E::Apply {
            op:
                ValueOp::Transpose
                | ValueOp::Permute
                | ValueOp::Squeeze
                | ValueOp::Unsqueeze
                | ValueOp::Broadcast,
            args,
        } => {
            args.first().is_some_and(operand_view)
                && args[1..]
                    .iter()
                    .all(|arg| matches!(arg, E::Constant(_) | E::Index(_)))
        }
        _ => false,
    }
}
