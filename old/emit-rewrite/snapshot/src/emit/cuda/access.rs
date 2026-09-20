//! Concrete accesses of a composed body. Lexical versions are private to emission.
use super::{EmitError, Region, Work};
use crate::{
    Expression as E, LoopKind, OperationId, OperationPayload, PhysicalPlan, Statement,
    ValueInstanceId,
};
use std::collections::{BTreeMap, BTreeSet};

/// Consume bounds-checked accesses without retaining a program-wide event list. Only the
/// persistent collector allocates readiness slots and per-task Work.
pub(super) trait AccessSink {
    fn read(&mut self, region: Region, stage: Option<usize>) -> Result<(), EmitError>;
    fn write(&mut self, region: Region) -> Result<(), EmitError>;
    fn stage(&mut self) -> usize {
        unreachable!("logical validation has no readiness stages")
    }
    fn properties(&mut self, _coordinate: [usize; 3], _ordered_collective: bool) {}
}

pub(super) fn visit(
    plan: &PhysicalPlan,
    statements: &[Statement],
    bindings: &BTreeMap<String, i64>,
    steps: &BTreeMap<String, i64>,
    rank: usize,
    physical: bool,
    sink: &mut dyn AccessSink,
) -> Result<(), EmitError> {
    struct Context<'a, 's> {
        plan: &'a PhysicalPlan,
        rank: usize,
        body_ops: BTreeSet<OperationId>,
        physical: bool,
        sink: &'s mut dyn AccessSink,
        local: Vec<Region>,
    }

    impl Context<'_, '_> {
        fn is_local(&self, r: Region) -> bool {
            matches!(
                self.plan.value_instance(r.value).unwrap().storage(),
                crate::Storage::Shared | crate::Storage::Register
            )
        }

        fn read(&mut self, region: Region, stage: Option<usize>) -> Result<(), EmitError> {
            super::regions::validate_region(self.plan, region)?;
            if self.is_local(region) {
                if !self.local.iter().any(|p| {
                    p.value == region.value
                        && p.rank == region.rank
                        && (0..3).all(|i| {
                            p.origin[i] <= region.origin[i]
                                && p.origin[i] + p.extent[i] >= region.origin[i] + region.extent[i]
                        })
                }) {
                    return Err(fail("local binding is not produced in this task's scope"));
                }
                Ok(())
            } else {
                self.sink.read(region, stage)
            }
        }

        fn write(&mut self, region: Region) -> Result<(), EmitError> {
            super::regions::validate_region(self.plan, region)?;
            if self.is_local(region) {
                self.local.push(region);
                Ok(())
            } else {
                self.sink.write(region)
            }
        }

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

            if let Some(work) = if self.physical {
                backend.work(self.plan, id, self.rank, coordinate)?
            } else {
                backend
                    .accesses(self.plan, id, self.rank, coordinate)?
                    .map(Work::from)
            } {
                if accum.is_some() {
                    return Err(fail("custom work inside an accumulation scope"));
                }
                self.sink
                    .properties(work.coordinate, work.ordered_collective);
                for region in work.reads {
                    self.read(region, None)?;
                }
                for reads in work.stages {
                    let stage = self.sink.stage();
                    for region in reads {
                        self.read(region, Some(stage))?;
                    }
                }
                for region in work.writes {
                    self.write(region)?;
                }
                return Ok(());
            }
            self.sink.properties(coordinate, false);
            let e = op
                .expression()
                .ok_or_else(|| fail("missing compute expression"))?;
            let a = args(e)?;
            if a.len() != 3 {
                return Err(fail("store arity"));
            }
            let output = region(self.plan, &a[0], &a[2], b, s, self.rank)?;
            let rhs = accum
                .and_then(|l| crate::plan::accumulation_rhs(e, &l.domain.variable))
                .unwrap_or(&a[1]);
            if let Some(l) = accum {
                if self.initialized(id, output.value) {
                    self.read(output, None)?;
                }
                let mut serial = b.clone();
                let mut ss = s.clone();
                let step = l.domain.step.evaluate(b).map_err(fail)?;
                ss.insert(l.domain.variable.clone(), step);
                if let Some(pattern) = if self.physical {
                    backend.stage_accesses(self.plan, id, &l.domain, rhs)?
                } else {
                    None
                } {
                    let mut whole = BTreeSet::new();
                    for input in op.inputs() {
                        if pattern.whole_compute_inputs
                            && *input != output.value
                            && self.plan.operations().any(|(producer, p)| {
                                !self.body_ops.contains(&producer)
                                    && producer.index() < id.index()
                                    && p.outputs().contains(input)
                                    && matches!(p.payload(), OperationPayload::Compute(_))
                            })
                        {
                            whole.insert(*input);
                            self.read(
                                super::regions::full_region(self.plan, *input, self.rank),
                                None,
                            )?;
                        }
                    }
                    let (start, stop, physical_step) = pattern.domain.bounds(b).map_err(fail)?;
                    for k in (start..stop).step_by(physical_step as usize) {
                        serial.insert(l.domain.variable.clone(), k);
                        let mut accesses = Vec::new();
                        reads(
                            self.plan,
                            &pattern.expression,
                            &serial,
                            &ss,
                            self.rank,
                            &mut accesses,
                        )?;
                        let stage = self.sink.stage();
                        for region in accesses {
                            if !whole.contains(&region.value) {
                                self.read(region, Some(stage))?;
                            }
                        }
                    }
                } else {
                    let (start, stop, step) = l.domain.bounds(b).map_err(fail)?;
                    for value in (start..stop).step_by(step as usize) {
                        serial.insert(l.domain.variable.clone(), value);
                        let mut accesses = Vec::new();
                        validate_shape(self.plan, rhs, output, &serial, &ss, self.rank)?;
                        reads(self.plan, rhs, &serial, &ss, self.rank, &mut accesses)?;
                        for r in accesses {
                            self.read(r, None)?;
                        }
                    }
                }
            } else {
                let mut accesses = Vec::new();
                validate_shape(self.plan, rhs, output, b, s, self.rank)?;
                reads(self.plan, rhs, b, s, self.rank, &mut accesses)?;
                for r in accesses {
                    self.read(r, None)?;
                }
            }
            self.write(output)?;
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
                                .and_then(|e| crate::plan::accumulation_rhs(e, &l.domain.variable))
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
                            let (start, stop, step) = l.domain.bounds(b).map_err(fail)?;
                            for v in (start..stop).step_by(step as usize) {
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
        physical,
        sink,
        local: Vec::new(),
    };
    c.visit(statements, bindings, steps)
}

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
    super::address::parse(plan, view, index)?.resolve(bindings, steps, rank)
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

fn validate_shape(
    plan: &PhysicalPlan,
    e: &E,
    output: Region,
    b: &BTreeMap<String, i64>,
    steps: &BTreeMap<String, i64>,
    rank: usize,
) -> Result<(), EmitError> {
    fn shape(
        plan: &PhysicalPlan,
        e: &E,
        b: &BTreeMap<String, i64>,
        steps: &BTreeMap<String, i64>,
        rank: usize,
    ) -> Result<Vec<usize>, EmitError> {
        if e.atom().is_some() {
            return Ok(Vec::new());
        }
        let a = args(e)?;
        match e.operator().unwrap() {
            "float_bits" => Ok(Vec::new()),
            "load" => {
                let r = region(plan, &a[0], &a[1], b, steps, rank)?;
                Ok(r.extent[..plan.value_instance(r.value).unwrap().shape().len()].to_vec())
            }
            "relu" | "sqr" | "sqrt" | "sigmoid" => shape(plan, &a[0], b, steps, rank),
            "rsum" | "unsqueeze" | "bcast" => {
                let mut s = shape(plan, &a[0], b, steps, rank)?;
                let axis = number(&a[1])?;
                if e.operator() == Some("rsum") {
                    if axis >= s.len() {
                        return Err(fail("reduction axis out of range"));
                    }
                    s.remove(axis);
                } else {
                    if axis > s.len() {
                        return Err(fail("broadcast axis out of range"));
                    }
                    s.insert(axis, usize::from(e.operator() == Some("unsqueeze")));
                }
                Ok(s)
            }
            "@" => {
                let l = shape(plan, &a[0], b, steps, rank)?;
                let r = shape(plan, &a[1], b, steps, rank)?;
                if l.len() != 2 || r.len() != 2 || l[1] != r[0] {
                    return Err(fail("matmul tile shape mismatch"));
                }
                Ok(vec![l[0], r[1]])
            }
            "+" | "-" | "*" | "/" => {
                let l = shape(plan, &a[0], b, steps, rank)?;
                let r = shape(plan, &a[1], b, steps, rank)?;
                if l.is_empty() {
                    return Ok(r);
                }
                if r.is_empty() {
                    return Ok(l);
                }
                if l.len() != r.len() {
                    return Err(fail("elementwise rank mismatch"));
                }
                l.into_iter()
                    .zip(r)
                    .map(|(a, b)| {
                        if a == b || a <= 1 || b <= 1 {
                            Ok(a.max(b))
                        } else {
                            Err(fail("elementwise tile shape mismatch"))
                        }
                    })
                    .collect()
            }
            _ => Err(fail("unsupported expression")),
        }
    }
    let shape = shape(plan, e, b, steps, rank)?;
    let dims = plan.value_instance(output.value).unwrap().shape().len();
    if shape.len() != dims
        || shape
            .iter()
            .zip(output.extent)
            .any(|(&a, b)| a != 0 && a != b)
    {
        return Err(fail("store tile shape mismatch"));
    }
    Ok(())
}
