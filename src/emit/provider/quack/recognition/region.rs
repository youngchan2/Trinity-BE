//! A read-only, whole-domain summary for library providers. Source tiles are
//! expanded only after proving complete loop coverage. The PhysicalPlan is not
//! rewritten; floating-point implementation choices remain a provider contract.
use super::*;

#[derive(Debug, Clone)]
pub enum RegionOperation {
    Gemm,
    GatedGemm { gate: E, up: E },
    Normalization(super::Normalization),
    Softmax { input: E, axis: usize },
    Other,
}

#[derive(Debug, Clone)]
pub struct RegionComputation {
    pub output: TensorAccess,
    pub expression: E,
    pub operation: RegionOperation,
}
impl RegionComputation {
    pub fn kind(&self) -> QuackPatternKind {
        match &self.operation {
            RegionOperation::Gemm => {
                if matches!(self.expression, E::Matmul(_)) {
                    QuackPatternKind::SingleGemm
                } else {
                    QuackPatternKind::GemmEpilogue
                }
            }
            RegionOperation::GatedGemm { .. } => QuackPatternKind::GatedGemm,
            RegionOperation::Normalization(n) => {
                if n.centered {
                    QuackPatternKind::LayerNorm
                } else {
                    QuackPatternKind::RmsNorm
                }
            }
            RegionOperation::Softmax { .. } => QuackPatternKind::Softmax,
            RegionOperation::Other => QuackPatternKind::Other,
        }
    }
    pub fn inputs(&self) -> Vec<ValueInstanceId> {
        self.expression.reads()
    }
}

pub(super) use crate::analysis::views::{integer, tensor_view};
pub(super) fn number(e: &E) -> Option<f64> {
    match e {
        E::Constant(Constant::Integer(n)) => Some(*n as f64),
        E::Constant(Constant::Float32(n)) => Some(f32::from_bits(*n) as f64),
        E::Constant(Constant::Float64(n)) => Some(f64::from_bits(*n)),
        _ => None,
    }
}
pub(super) fn axis_op(e: &E, op: ValueOp) -> Option<(&E, usize)> {
    match e {
        E::ReduceSum { value, axis } if op == ValueOp::ReduceSum => Some((value, *axis)),
        E::Broadcast { value, axis } | E::Unsqueeze { value, axis } if op == ValueOp::Broadcast => {
            Some((value, *axis))
        }
        E::Apply { op: o, args }
            if *o == op || (op == ValueOp::Broadcast && *o == ValueOp::Unsqueeze) =>
        {
            Some((args.first()?, integer(args.get(1)?)?))
        }
        _ => None,
    }
}
pub(super) fn unary(e: &E, op: ValueOp) -> Option<&E> {
    match e {
        E::Apply { op: o, args } if *o == op && args.len() == 1 => args.first(),
        _ => None,
    }
}

#[derive(Clone)]
struct Definition {
    access: TensorAccess,
    value: E,
    dependencies: BTreeSet<OperationId>,
}
struct Summary<'p> {
    plan: &'p PhysicalPlan,
    loops: BTreeMap<String, &'p Loop>,
    written: BTreeSet<ValueInstanceId>,
    definitions: BTreeMap<ValueInstanceId, Definition>,
    budget: usize,
}
impl Summary<'_> {
    fn access(&self, a: &TensorAccess) -> MatchResult<TensorAccess> {
        let mut a = a.clone();
        let shape = a
            .shape(self.plan.value_instance(a.value).unwrap().shape())
            .to_vec();
        for (axis, index) in a.indices.iter_mut().enumerate() {
            let tile = match &*index {
                AccessIndex::Tile { variable, width }
                | AccessIndex::ClippedTile { variable, width } => Some((variable, width)),
                AccessIndex::Slice {
                    start: IndexExpr::Variable(variable),
                    width,
                } if self.loops.contains_key(variable) => Some((variable, width)),
                _ => None,
            };
            if let Some((variable, width)) = tile {
                let l = self
                    .loops
                    .get(variable)
                    .ok_or("access has an unknown tile loop")?;
                if l.domain.start.evaluate(self.plan.bindings()) != Ok(0)
                    || l.domain.stop.evaluate(self.plan.bindings()) != Ok(shape[axis] as i64)
                    || l.domain.step.evaluate(self.plan.bindings()).ok()
                        != width.resolve(self.plan.bindings()).ok().map(|w| w as i64)
                {
                    return Err(
                        "region does not cover a complete non-overlapping tensor axis".into(),
                    );
                }
                *index = AccessIndex::FullTile;
            }
        }
        // Resolve shape symbols in the summary, while original accesses retain them.
        a.view_dimensions = None;
        tensor_view(self.plan, &E::Load(a.clone())).ok_or("unsupported whole-domain view/index")?;
        Ok(a)
    }
    fn expand(&mut self, e: &E, deps: &mut BTreeSet<OperationId>) -> MatchResult<E> {
        self.budget = self
            .budget
            .checked_sub(1)
            .ok_or("region expression summary exceeds its node budget")?;
        if let E::Load(a) = e {
            let a = self.access(a)?;
            if let Some(d) = self.definitions.get(&a.value) {
                if !same_access(self.plan, &a, &d.access) {
                    return Err("local read uses a different view or sub-tile".into());
                }
                deps.extend(&d.dependencies);
                return Ok(d.value.clone());
            }
            if self.written.contains(&a.value) {
                return Err("local value read before its definition".into());
            }
            return Ok(E::Load(a));
        }
        if !e.is_pure() || matches!(e, E::Index(_)) {
            return Err("region contains an unsupported effect/index value".into());
        }
        let mut result = e.clone();
        for child in result.children_mut() {
            *child = self.expand(child, deps)?;
        }
        Ok(result)
    }
    fn operation(
        &mut self,
        id: OperationId,
        serial: Option<&Loop>,
    ) -> MatchResult<ValueInstanceId> {
        let (access, rhs) = store(self.plan, id)?;
        let mut deps = BTreeSet::from([id]);
        let value = if let Some(l) = serial {
            let E::Add(args) = rhs else {
                return Err("serial body is not an additive recurrence".into());
            };
            let term = if self_load(self.plan, &args[0], access) {
                unscaled(&args[1])
            } else if self_load(self.plan, &args[1], access) {
                unscaled(&args[0])
            } else {
                return Err("serial update is not an exact self recurrence".into());
            };
            let prior = self.definitions.get(&access.value);
            let op = self.plan.operation(id).unwrap();
            if let Some(d) = prior {
                if !scalar(&d.value, 0.0) {
                    return Err("nonzero initial recurrence value".into());
                }
                deps.extend(&d.dependencies);
            } else if !op.zero_init().contains(&access.value)
                && op.inflows().contains(&access.value)
            {
                return Err("recurrence has an external seed".into());
            }
            if let E::Matmul(args) = term {
                if !contracts_k(self.plan, l, args, access) {
                    return Err("incomplete GEMM contraction".into());
                }
            } else if let Some((operand, axis)) = axis_op(term, ValueOp::ReduceSum) {
                let accesses = operand.accesses();
                if accesses.is_empty()
                    || accesses.iter().any(|a| {
                        axis >= a.indices.len()
                            || !uses_axis(&a.indices[axis], &l.domain.variable)
                            || a.indices
                                .iter()
                                .enumerate()
                                .any(|(i, v)| i != axis && uses_axis(v, &l.domain.variable))
                    })
                    || access
                        .indices
                        .iter()
                        .any(|a| uses_axis(a, &l.domain.variable))
                {
                    return Err("serial loop is not a reduction-axis traversal".into());
                }
            } else {
                return Err("serial recurrence is neither GEMM nor sum reduction".into());
            }
            self.expand(term, &mut deps)?
        } else {
            self.expand(rhs, &mut deps)?
        };
        let access = self.access(access)?;
        let id = access.value;
        self.definitions.insert(
            id,
            Definition {
                access,
                value,
                dependencies: deps,
            },
        );
        Ok(id)
    }
}

pub fn summarize(
    plan: &PhysicalPlan,
    statement: &Statement,
    scope: &RegionScope,
) -> MatchResult<RegionComputation> {
    let mut body = match statement {
        Statement::Region(b) => b.as_slice(),
        _ => std::slice::from_ref(statement),
    };
    let mut parallel = Vec::new();
    while let [Statement::Loop(l)] = body {
        if l.kind != LoopKind::Parallel {
            break;
        }
        parallel.push(l);
        body = &l.body;
    }
    let mut summary = Summary {
        plan,
        loops: parallel
            .iter()
            .map(|l| (l.domain.variable.clone(), *l))
            .collect(),
        written: scope
            .operations
            .iter()
            .map(|id| store(plan, *id).map(|(a, _)| a.value))
            .collect::<MatchResult<_>>()?,
        definitions: BTreeMap::new(),
        budget: 100_000,
    };
    let mut last = None;
    for s in body {
        match s {
            Statement::Operation(id) => last = Some(summary.operation(*id, None)?),
            Statement::Loop(l) if l.kind == LoopKind::Sequential => {
                summary.loops.insert(l.domain.variable.clone(), l);
                // Reuse the existing proof for product-store + accumulator
                // spellings, rather than requiring the Matmul in the update.
                let initialization = l.body.last().and_then(|s| {
                    let Statement::Operation(id) = s else {
                        return None;
                    };
                    let (a, _) = store(plan, *id).ok()?;
                    let d = summary.definitions.get(&a.value)?;
                    if !scalar(&d.value, 0.0) {
                        return None;
                    }
                    let init = *d.dependencies.iter().next()?;
                    Some((init, store(plan, init).ok()?.0))
                });
                let mut covered = BTreeSet::new();
                if let Some((id, _)) = initialization {
                    covered.insert(id);
                }
                if let Ok(gemm) = reduction_gemm(plan, l, initialization, &mut covered) {
                    let value = summary.expand(gemm.matmul(), &mut covered)?;
                    let access = summary.access(gemm.accumulator)?;
                    last = Some(access.value);
                    summary.definitions.insert(
                        access.value,
                        Definition {
                            access,
                            value,
                            dependencies: covered,
                        },
                    );
                    continue;
                }
                let before = summary.definitions.clone();
                let mut updates = Vec::new();
                for s in &l.body {
                    let Statement::Operation(id) = s else {
                        return Err("nested serial computation is unsupported".into());
                    };
                    // Independent recurrences must not consume another update's
                    // whole-domain summary as if it were a per-iteration value.
                    summary.definitions = before.clone();
                    let v = summary.operation(*id, Some(l))?;
                    updates.push((v, summary.definitions[&v].clone()));
                    last = Some(v);
                }
                summary.definitions = before;
                summary.definitions.extend(updates);
            }
            _ => return Err("region has unsupported nested parallel/split structure".into()),
        }
    }
    let last = last.ok_or("empty region")?;
    let d = &summary.definitions[&last];
    if d.dependencies != scope.operations.iter().copied().collect() {
        return Err("region contains unrelated or discarded stores".into());
    }
    for v in summary.written.iter().filter(|v| **v != last) {
        if plan.value_instance(*v).unwrap().storage() != Storage::Register
            || plan.outputs().iter().any(|b| b.value() == *v)
            || plan.operations().any(|(id, op)| {
                !scope.operations.contains(&id) && op.expression().reads().contains(v)
            })
        {
            return Err("region has additional observable intermediate values".into());
        }
    }
    let final_id = scope.operations.last().ok_or("empty region")?;
    let (original_output, _) = store(plan, *final_id)?;
    for l in parallel {
        if original_output
            .indices
            .iter()
            .filter(|i| uses_axis(i, &l.domain.variable))
            .count()
            != 1
        {
            return Err("parallel loop does not independently tile the final output".into());
        }
    }
    if d.value.reads().contains(&last) {
        return Err("region output aliases an input".into());
    }
    let operation = if let Some((gate, up)) = gated(&d.value) {
        if let (E::Matmul(g), E::Matmul(u)) = (gate, up) {
            if g[0] == u[0] && g[1] != u[1] {
                RegionOperation::GatedGemm {
                    gate: gate.clone(),
                    up: up.clone(),
                }
            } else {
                RegionOperation::Other
            }
        } else {
            RegionOperation::Other
        }
    } else if let Some(n) = super::normalization::recognize(plan, &d.value) {
        RegionOperation::Normalization(n)
    } else if let Some((input, axis)) = super::normalization::softmax(&d.value) {
        RegionOperation::Softmax { input, axis }
    } else {
        let mut mm = None;
        let products: usize = scope
            .operations
            .iter()
            .map(|id| matmul_count(plan.operation(*id).unwrap().expression()))
            .sum();
        if products == 1 && single_product_tree(&d.value, &mut mm) && mm.is_some() {
            RegionOperation::Gemm
        } else {
            RegionOperation::Other
        }
    };
    Ok(RegionComputation {
        output: d.access.clone(),
        expression: d.value.clone(),
        operation,
    })
}

/// Original stores with proven complete leading parallel traversals expanded.
/// Unlike `summarize`, this retains every output and does not inline producers.
/// Serial recurrences require a separate proof (e.g. `recognize_gemm`).
#[derive(Debug, Clone)]
pub struct DomainStore {
    pub operation: OperationId,
    pub destination: TensorAccess,
    pub expression: E,
}

pub fn domain_stores(
    facts: &crate::analysis::regions::RegionFacts<'_>,
) -> MatchResult<Vec<DomainStore>> {
    let mut body = match facts.statement {
        Statement::Region(b) => b.as_slice(),
        s => std::slice::from_ref(s),
    };
    let mut loops = BTreeMap::new();
    while let [Statement::Loop(l)] = body {
        if l.kind != LoopKind::Parallel {
            break;
        }
        loops.insert(l.domain.variable.clone(), l);
        body = &l.body;
    }
    let mut summary = Summary {
        plan: facts.plan,
        loops,
        written: BTreeSet::new(),
        definitions: BTreeMap::new(),
        budget: 100_000,
    };
    body.iter().map(|s| {
        let Statement::Operation(id) = s else {
            return Err("equation access proof supports leading ploop chains and straight-line stores; serial/split/nested bodies need another proof".into());
        };
        let (a, e) = store(facts.plan, *id)?;
        for name in summary.loops.keys() {
            if a.indices.iter().filter(|i| uses_axis(i, name)).count() != 1 {
                return Err("parallel loop does not independently tile each store".into());
            }
        }
        Ok(DomainStore {
            operation: *id,
            destination: summary.access(a)?,
            expression: summary.expand(e, &mut BTreeSet::new())?,
        })
    }).collect()
}
// Expanding a register definition can repeat its Matmul subtree (e.g. SiLU).
// Compare the computation, not the number of copies in this summary tree.
fn single_product_tree<'a>(e: &'a E, product: &mut Option<&'a E>) -> bool {
    match e {
        E::Matmul(args) => {
            if !args.iter().all(operand_view) || product.is_some_and(|old| old != e) {
                return false;
            }
            *product = Some(e);
            true
        }
        E::Load(_) | E::Constant(_) => true,
        _ if is_pointwise(e) => e.children().iter().all(|c| single_product_tree(c, product)),
        _ => operand_view(e),
    }
}
fn gated(e: &E) -> Option<(&E, &E)> {
    fn factors<'a>(e: &'a E, out: &mut Vec<&'a E>) {
        if let E::Mul(a) = e {
            factors(&a[0], out);
            factors(&a[1], out)
        } else {
            out.push(e)
        }
    }
    let mut f = Vec::new();
    factors(e, &mut f);
    if f.len() != 3 {
        return None;
    }
    for x in &f {
        if let E::Sigmoid(g) = x {
            let i = f.iter().position(|v| *v == g.as_ref())?;
            let other = f
                .iter()
                .find(|v| !std::ptr::eq(**v, *x) && !std::ptr::eq(**v, f[i]))?;
            return Some((g, other));
        }
    }
    None
}
