//! Concrete accesses of a composed body. Lexical versions are private to emission.
use super::{EmitError, Region, Work};
use crate::{
    Expression as E, LoopKind, OperationId, OperationPayload, PhysicalPlan, Statement,
    ValueInstanceId,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn fail(s: impl Into<String>) -> EmitError {
    EmitError::Contract(s.into())
}
fn args(e: &E) -> Result<&[E], EmitError> {
    e.list()
        .filter(|xs| !xs.is_empty())
        .map(|xs| &xs[1..])
        .ok_or_else(|| fail("expected expression"))
}
fn atom(e: &E) -> Result<&str, EmitError> {
    e.atom().ok_or_else(|| fail("expected atom"))
}
fn number(e: &E) -> Result<usize, EmitError> {
    atom(e)?.parse().map_err(|_| fail("expected extent"))
}

pub(super) fn region(
    plan: &PhysicalPlan,
    view: &E,
    index: &E,
    bindings: &BTreeMap<String, i64>,
    steps: &BTreeMap<String, i64>,
    rank: usize,
) -> Result<Region, EmitError> {
    let v = args(view)?;
    if view.operator() != Some("view") || v.len() != 2 {
        return Err(fail("invalid view"));
    }
    let base = args(&v[0])?;
    let name = base
        .first()
        .and_then(E::atom)
        .ok_or_else(|| fail("invalid tensor name"))?;
    let (id, value) = plan
        .value_instances()
        .find(|(_, v)| v.name() == Some(name))
        .ok_or_else(|| fail("unknown tensor name"))?;
    let layout = args(&v[1])?;
    if layout.len() != value.shape().len() || index.operator() != Some("keyed_index") {
        return Err(fail("view/index rank mismatch"));
    }
    let mut slots = BTreeMap::new();
    for slot in args(index)? {
        let s = args(slot)?;
        if s.len() != 2 || slots.insert(atom(&s[0])?, &s[1]).is_some() {
            return Err(fail("duplicate/invalid index slot"));
        }
    }
    let mut origin = [0; 3];
    let mut extent = [1; 3];
    for (i, (axis, &size)) in layout.iter().zip(value.shape()).enumerate() {
        let a = args(axis)?;
        if a.len() != 2 || number(&a[1])? != size {
            return Err(fail("view extent differs from value"));
        }
        extent[i] = size;
        if let Some(part) = slots.remove(atom(&a[0])?) {
            if part.atom() == Some("fulltile") {
                continue;
            }
            let p = args(part)?;
            let variable = atom(p.first().ok_or_else(|| fail("missing index"))?)?;
            let binding = *bindings
                .get(variable)
                .ok_or_else(|| fail(format!("unbound index {variable}")))?;
            let start = usize::try_from(binding).map_err(|_| fail("negative index"))?;
            match part.operator() {
                Some("tile" | "clipped_tile") => {
                    let width = number(p.get(1).ok_or_else(|| fail("missing tile width"))?)?;
                    origin[i] = start;
                    extent[i] = if part.operator() == Some("clipped_tile") {
                        width.min(size.saturating_sub(start))
                    } else {
                        width
                    };
                }
                Some("elem") => {
                    let step = *steps
                        .get(variable)
                        .ok_or_else(|| fail("missing elem step"))?;
                    if step <= 0 {
                        return Err(fail("invalid elem step"));
                    }
                    origin[i] = start / step as usize;
                    extent[i] = 1;
                }
                _ => return Err(fail("unsupported access index")),
            }
        }
        if extent[i] == 0
            || origin[i]
                .checked_add(extent[i])
                .is_none_or(|end| end > size)
        {
            return Err(fail("out-of-bounds execution region"));
        }
    }
    if !slots.is_empty() {
        return Err(fail("index slot absent from view"));
    }
    Ok(Region {
        value: id,
        rank,
        origin,
        extent,
    })
}

#[derive(Clone, Debug)]
pub(super) struct Event {
    pub region: Region,
    pub write: bool,
    pub stage: Option<usize>,
}
#[derive(Clone, Debug, Default)]
pub(super) struct Effects {
    pub work: Work,
    pub events: Vec<Event>,
}
impl Effects {
    fn read(&mut self, region: Region, stage: Option<usize>) {
        if let Some(stage) = stage {
            self.work.stages[stage].push(region);
        } else {
            self.work.reads.push(region);
        }
        self.events.push(Event {
            region,
            write: false,
            stage,
        });
    }
    fn write(&mut self, region: Region) {
        self.work.writes.push(region);
        self.events.push(Event {
            region,
            write: true,
            stage: None,
        });
    }
}
fn reads(
    plan: &PhysicalPlan,
    e: &E,
    b: &BTreeMap<String, i64>,
    steps: &BTreeMap<String, i64>,
    rank: usize,
    out: &mut Vec<Region>,
) -> Result<(), EmitError> {
    if e.operator() == Some("load") {
        let a = args(e)?;
        if a.len() != 2 {
            return Err(fail("load arity"));
        }
        out.push(region(plan, &a[0], &a[1], b, steps, rank)?);
        return Ok(());
    }
    if let Some(xs) = e.list() {
        for e in &xs[1..] {
            reads(plan, e, b, steps, rank, out)?;
        }
    }
    Ok(())
}
fn has_matmul(e: &E) -> bool {
    e.operator() == Some("@") || e.list().is_some_and(|xs| xs.iter().any(has_matmul))
}

pub(super) fn describe(
    plan: &PhysicalPlan,
    statements: &[Statement],
    bindings: &BTreeMap<String, i64>,
    steps: &BTreeMap<String, i64>,
    rank: usize,
) -> Result<Effects, EmitError> {
    struct Context<'a> {
        plan: &'a PhysicalPlan,
        rank: usize,
        body_ops: BTreeSet<OperationId>,
        effects: Effects,
    }
    impl Context<'_> {
        fn initialized(&self, id: OperationId, value: ValueInstanceId) -> bool {
            self.plan.inputs().iter().any(|b| b.value() == value)
                || self
                    .plan
                    .operations()
                    .any(|(prior, op)| prior.index() < id.index() && op.outputs().contains(&value))
        }
        fn store(
            &mut self,
            id: OperationId,
            b: &BTreeMap<String, i64>,
            s: &BTreeMap<String, i64>,
            accum: Option<&crate::Loop>,
        ) -> Result<(), EmitError> {
            let op = self.plan.operation(id).unwrap();
            let mut coordinate = [0; 3];
            for (i, e) in op.coordinates().iter().enumerate() {
                coordinate[i] = usize::try_from(e.evaluate(b).map_err(fail)?)
                    .map_err(|_| fail("negative coordinate"))?;
            }
            let backend = match op.payload() {
                OperationPayload::Compute(c) => c.implementation(),
                OperationPayload::Communication(c) => c.implementation(),
            }
            .definition()
            .cuda()
            .ok_or_else(|| fail("missing backend"))?;
            if let Some(work) = backend.work(self.plan, id, self.rank, coordinate)? {
                if accum.is_some() {
                    return Err(fail("custom work inside an accumulation scope"));
                }
                let base = self.effects.work.stages.len();
                self.effects.work.coordinate = work.coordinate;
                self.effects.work.ordered_collective |= work.ordered_collective;
                for region in work.reads {
                    self.effects.read(region, None);
                }
                for (stage, reads) in work.stages.into_iter().enumerate() {
                    self.effects.work.stages.push(Vec::new());
                    for region in reads {
                        self.effects.read(region, Some(base + stage));
                    }
                }
                for region in work.writes {
                    self.effects.write(region);
                }
                return Ok(());
            }
            self.effects.work.coordinate = coordinate;
            let e = op
                .expression()
                .ok_or_else(|| fail("missing compute expression"))?;
            let a = args(e)?;
            if a.len() != 3 {
                return Err(fail("store arity"));
            }
            let output = region(self.plan, &a[0], &a[2], b, s, self.rank)?;
            let rhs = accum
                .and_then(|l| crate::physical::accumulation_rhs(e, &l.domain.variable))
                .unwrap_or(&a[1]);
            if let Some(l) = accum {
                if self.initialized(id, output.value) {
                    self.effects.read(output, None);
                }
                let mut serial = b.clone();
                let mut ss = s.clone();
                let step = l.domain.step.evaluate(b).map_err(fail)?;
                ss.insert(l.domain.variable.clone(), step);
                if has_matmul(rhs) {
                    let start = l.domain.start.evaluate(b).map_err(fail)?;
                    let stop = l.domain.stop.evaluate(b).map_err(fail)?;
                    if step <= 0 || step % 64 != 0 {
                        return Err(fail("invalid WGMMA K step"));
                    }
                    let mut whole = BTreeSet::new();
                    for input in op.inputs() {
                        if *input != output.value
                            && self.plan.operations().any(|(producer, p)| {
                                !self.body_ops.contains(&producer)
                                    && producer.index() < id.index()
                                    && p.outputs().contains(input)
                                    && matches!(p.payload(), OperationPayload::Compute(_))
                            })
                        {
                            whole.insert(*input);
                            self.effects.read(
                                super::graph::full_region(self.plan, *input, self.rank),
                                None,
                            );
                        }
                    }
                    for k in (start..stop).step_by(64) {
                        serial.insert(l.domain.variable.clone(), k);
                        let mut accesses = Vec::new();
                        // Each physical WGMMA stage reads 64 K elements even when
                        // source notation groups several stages into one iteration.
                        let mut expression = rhs.clone();
                        fn stage_tiles(e: &mut E, var: &str) {
                            if let E::List(xs) = e {
                                if xs.len() == 3
                                    && xs[0].atom() == Some("tile")
                                    && xs[1].atom() == Some(var)
                                {
                                    xs[2] = E::Atom("64".into());
                                }
                                for x in xs {
                                    stage_tiles(x, var);
                                }
                            }
                        }
                        stage_tiles(&mut expression, &l.domain.variable);
                        reads(
                            self.plan,
                            &expression,
                            &serial,
                            &ss,
                            self.rank,
                            &mut accesses,
                        )?;
                        let stage = self.effects.work.stages.len();
                        self.effects.work.stages.push(Vec::new());
                        for region in accesses {
                            if !whole.contains(&region.value) {
                                self.effects.read(region, Some(stage));
                            }
                        }
                    }
                } else {
                    for value in l.domain.values(b).map_err(fail)? {
                        serial.insert(l.domain.variable.clone(), value);
                        let mut accesses = Vec::new();
                        reads(self.plan, rhs, &serial, &ss, self.rank, &mut accesses)?;
                        for r in accesses {
                            self.effects.read(r, None);
                        }
                    }
                }
            } else {
                let mut accesses = Vec::new();
                reads(self.plan, rhs, b, s, self.rank, &mut accesses)?;
                for r in accesses {
                    self.effects.read(r, None);
                }
            }
            self.effects.write(output);
            Ok(())
        }
        fn visit(
            &mut self,
            statements: &[Statement],
            b: &BTreeMap<String, i64>,
            s: &BTreeMap<String, i64>,
        ) -> Result<(), EmitError> {
            for statement in statements {
                match statement {
                    Statement::Operation(id) => self.store(*id, b, s, None)?,
                    Statement::Loop(l) => {
                        if l.kind != LoopKind::Sequential {
                            return Err(fail(
                                "parallel work inside sequential loop is unsupported",
                            ));
                        }
                        let accumulator = match l.body.as_slice() {
                            [Statement::Operation(id)] => self
                                .plan
                                .operation(*id)
                                .unwrap()
                                .expression()
                                .and_then(|e| {
                                    crate::physical::accumulation_rhs(e, &l.domain.variable)
                                })
                                .map(|_| *id),
                            _ => None,
                        };
                        if let Some(id) = accumulator {
                            let mut bb = b.clone();
                            bb.insert(
                                l.domain.variable.clone(),
                                l.domain.start.evaluate(b).map_err(fail)?,
                            );
                            self.store(id, &bb, s, Some(l))?;
                        } else {
                            let mut bb = b.clone();
                            let mut ss = s.clone();
                            ss.insert(
                                l.domain.variable.clone(),
                                l.domain.step.evaluate(b).map_err(fail)?,
                            );
                            for v in l.domain.values(b).map_err(fail)? {
                                bb.insert(l.domain.variable.clone(), v);
                                self.visit(&l.body, &bb, &ss)?;
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    }
    let mut c = Context {
        plan,
        rank,
        body_ops: statements.iter().flat_map(Statement::operations).collect(),
        effects: Effects::default(),
    };
    c.visit(statements, bindings, steps)?;
    // Local bindings belong to this concrete task. Validate them before removing
    // their memory effects; they neither allocate a buffer nor publish readiness.
    let local = |r: &Region| {
        matches!(
            plan.value_instance(r.value).unwrap().storage(),
            crate::Storage::Shared | crate::Storage::Register
        )
    };
    let mut produced: Vec<Region> = Vec::new();
    for event in &c.effects.events {
        if !local(&event.region) {
            continue;
        }
        if event.write {
            produced.push(event.region);
        } else if !produced.iter().any(|p| {
            p.value == event.region.value
                && p.rank == event.region.rank
                && (0..3).all(|i| {
                    p.origin[i] <= event.region.origin[i]
                        && p.origin[i] + p.extent[i]
                            >= event.region.origin[i] + event.region.extent[i]
                })
        }) {
            return Err(fail("local binding is not produced in this task's scope"));
        }
    }
    c.effects.events.retain(|e| !local(&e.region));
    c.effects.work.reads.retain(|r| !local(r));
    c.effects.work.writes.retain(|r| !local(r));
    for stage in &mut c.effects.work.stages {
        stage.retain(|r| !local(r));
    }
    Ok(c.effects)
}
