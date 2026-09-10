//! CUDA expression bindings and implementation phases, local to one device body.
use super::{Binding, Body, Code, EmitError, Resources, program::TaskDomain};
use crate::{
    DType, Expression as E, Loop, LoopKind, OperationId, PhysicalPlan, Statement, Storage,
    ValueInstanceId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
fn fail(s: impl Into<String>) -> EmitError {
    EmitError::Unsupported(s.into())
}
fn args(e: &E) -> Result<&[E], EmitError> {
    e.list()
        .map(|l| &l[1..])
        .ok_or_else(|| fail("expected expression"))
}
fn atom(e: &E) -> Result<&str, EmitError> {
    e.atom().ok_or_else(|| fail("expected atom"))
}
fn number(e: &E) -> Result<usize, EmitError> {
    atom(e)?
        .parse()
        .map_err(|_| fail("expected nonnegative integer"))
}

#[derive(Clone)]
struct Access {
    value: usize,
    dtype: DType,
    shape: Vec<usize>,
    width: Vec<usize>,
    origin: Vec<String>,
}
struct Context<'a> {
    plan: &'a PhysicalPlan,
    names: BTreeMap<String, String>,
    steps: BTreeMap<String, i64>,
    next: usize,
    resources: Resources,
    concrete: BTreeMap<String, i64>,
    scalar: Option<&'static dyn super::CudaImplementation>,
    current_operation: usize,
    direct: BTreeMap<usize, String>,
    consumed: BTreeSet<OperationId>,
    shared: BTreeMap<usize, (Access, usize)>,
    local_bytes: usize,
    scratch_bytes: usize,
}
impl<'p> Context<'p> {
    fn storage(&self, value: usize) -> Storage {
        self.plan
            .value_instance(ValueInstanceId::from_index(value))
            .unwrap()
            .storage()
    }

    fn pointer(&self, value: usize) -> Result<String, EmitError> {
        let id = ValueInstanceId::from_index(value);
        let dtype = ctype(self.plan.value_instance(id).unwrap().dtype());
        if let Some((_, offset)) = self.shared.get(&value) {
            return Ok(format!(
                "reinterpret_cast<{dtype}*>(static_cast<char*>(memory)+{offset})"
            ));
        }
        Ok(format!(
            "static_cast<{dtype}*>(bindings.values[{}])",
            super::binding_slot(self.plan, id)?
        ))
    }

    fn memory_offset(&self, access: &Access, coords: &[String]) -> String {
        if let Some((owner, _)) = self.shared.get(&access.value) {
            let mut local = access.clone();
            local.shape = owner.width.clone();
            local.origin = access
                .origin
                .iter()
                .zip(&owner.origin)
                .map(|(a, b)| format!("({a})-({b})"))
                .collect();
            self.offset(&local, coords)
        } else {
            self.offset(access, coords)
        }
    }

    fn allocate_shared(&mut self, access: &Access) {
        if self.storage(access.value) == Storage::Shared && !self.shared.contains_key(&access.value)
        {
            let offset = self.scratch_bytes + self.local_bytes;
            self.shared.insert(access.value, (access.clone(), offset));
            self.local_bytes +=
                (access.width.iter().product::<usize>() * access.dtype.size_bytes()).div_ceil(128)
                    * 128;
            self.resources.shared_memory_bytes = self
                .resources
                .shared_memory_bytes
                .max(self.scratch_bytes + self.local_bytes);
        }
    }

    fn interface(
        &self,
        body: &mut Body,
        value: usize,
        name: &str,
        output: bool,
    ) -> super::SymbolId {
        let symbol = body.declare(name, true);
        let id = ValueInstanceId::from_index(value);
        let tensor = self.plan.value_instance(id).unwrap();
        let binding = Binding {
            symbol,
            value: id,
            dtype: tensor.dtype(),
            storage: tensor.storage(),
        };
        if output {
            body.epilogue.outputs.push(binding);
        } else {
            body.epilogue.inputs.push(binding);
        }
        symbol
    }

    /// Materialize each operation's declared dtype before forwarding its value.
    fn output(
        &mut self,
        body: &mut Body,
        access: &Access,
        coords: &[String],
        value: &str,
        followers: &[Statement],
    ) -> Result<Code, EmitError> {
        self.allocate_shared(access);
        let name = format!("result_{}", access.value);
        let symbol = body.declare(&name, false);
        let id = ValueInstanceId::from_index(access.value);
        if self.storage(access.value) == Storage::Register {
            body.epilogue.outputs.push(Binding {
                symbol,
                value: id,
                dtype: access.dtype,
                storage: self.storage(access.value),
            });
        }
        let mut text = format!(
            "{} ${{{name}}}={}({value});\n",
            ctype(access.dtype),
            ctype(access.dtype)
        );
        if self.storage(access.value) != Storage::Register {
            let pointer = format!("buffer_{}", access.value);
            let binding = self.interface(body, access.value, &pointer, true);
            let offset = self.memory_offset(access, coords);
            text.push_str(&format!("${{{pointer}}}[{offset}]=${{{name}}};\n"));
            let mut code = body.cuda_code(&text)?;
            if self.storage(access.value) != Storage::Shared {
                code.substitute(binding, &Code::text(self.pointer(access.value)?));
            }
            return Ok(code);
        }
        let Some((Statement::Operation(consumer), remaining)) = followers.split_first() else {
            return Err(fail(
                "register output needs an adjacent pointwise consumer in the same scope",
            ));
        };
        if !crate::fusion::pointwise(self.plan, *consumer) {
            return Err(fail("unsupported register fragment consumer"));
        }
        let op = self.plan.operation(*consumer).unwrap();
        if !op.inputs().contains(&id) {
            return Err(fail("unconnected register consumer"));
        }
        let expression = op.expression().unwrap();
        self.select(*consumer)?;
        for external in op.inputs().iter().filter(|v| **v != id) {
            self.interface(
                body,
                external.index(),
                &format!("buffer_{}", external.index()),
                false,
            );
        }
        let input = format!("input_{}_{}", consumer.index(), access.value);
        let formal = self.interface(body, access.value, &input, false);
        self.direct.insert(access.value, input);
        let xs = args(expression)?;
        let destination = self.access(&xs[0], &xs[2])?;
        let result = self.rounded_expr(&xs[1], coords, &destination)?;
        self.direct.remove(&access.value);
        if destination.width != access.width || destination.origin != access.origin {
            return Err(fail("register consumer tile differs from producer"));
        }
        self.consumed.insert(*consumer);
        let mut code = body.cuda_code(&text)?;
        code.append(&self.output(body, &destination, coords, &result, remaining)?);
        // Both fragments share one symbol table. Substitution changes identities,
        // including the interface, before any CUDA is rendered.
        let saved = std::mem::replace(&mut body.epilogue.code, code);
        body.substitute(formal, symbol);
        let code = std::mem::replace(&mut body.epilogue.code, saved);
        Ok(code)
    }
    fn select(&mut self, id: crate::OperationId) -> Result<super::CudaPhaseTemplate, EmitError> {
        self.current_operation = id.index();
        let instance = match self.plan.operation(id).unwrap().payload() {
            crate::OperationPayload::Compute(c) => c.implementation(),
            crate::OperationPayload::Communication(c) => c.implementation(),
        };
        self.scalar = instance.definition().cuda();
        let phases = self
            .scalar
            .ok_or_else(|| fail("selected implementation has no CUDA composition backend"))?
            .phases(self.plan, id)?;
        self.resources.sequential(&phases.resources());
        Ok(phases)
    }

    fn initialized(&self, value: usize) -> bool {
        self.plan
            .inputs()
            .iter()
            .any(|input| input.value().index() == value)
            || self.plan.operations().any(|(id, op)| {
                id.index() < self.current_operation
                    && op.outputs().iter().any(|output| output.index() == value)
            })
    }
    fn access(&self, view: &E, index: &E) -> Result<Access, EmitError> {
        let v = args(view)?;
        if view.operator() != Some("view") || v.len() != 2 {
            return Err(fail("invalid view"));
        }
        let name = atom(&args(&v[0])?[0])?;
        let (id, value) = self
            .plan
            .value_instances()
            .find(|(_, v)| v.name() == Some(name))
            .ok_or_else(|| fail("unknown tensor binding"))?;
        let layout = args(&v[1])?;
        let mut slots = BTreeMap::new();
        if index.operator() != Some("keyed_index") {
            return Err(fail("expected keyed_index"));
        }
        for slot in args(index)? {
            let a = args(slot)?;
            if a.len() != 2 || slots.insert(atom(&a[0])?, &a[1]).is_some() {
                return Err(fail("invalid/duplicate index slot"));
            }
        }
        let mut width = Vec::new();
        let mut origin = Vec::new();
        for (axis, extent) in layout.iter().zip(value.shape()) {
            let axis_name = atom(&args(axis)?[0])?;
            let Some(part) = slots.remove(axis_name) else {
                width.push(*extent);
                origin.push("0".into());
                continue;
            };
            match part.operator() {
                Some("tile" | "clipped_tile" | "elem") => {
                    let a = args(part)?;
                    let var = atom(&a[0])?;
                    let binding = self
                        .names
                        .get(var)
                        .ok_or_else(|| fail(format!("unbound index {var}")))?;
                    if matches!(part.operator(), Some("tile" | "clipped_tile")) {
                        let w = number(&a[1])?;
                        if w == 0 {
                            return Err(fail("zero tile width"));
                        }
                        let w = if part.operator() == Some("clipped_tile") {
                            let start = usize::try_from(
                                *self
                                    .concrete
                                    .get(var)
                                    .ok_or_else(|| fail("missing clipped tile coordinate"))?,
                            )
                            .map_err(|_| fail("negative tile coordinate"))?;
                            w.min(extent.saturating_sub(start))
                        } else {
                            w
                        };
                        if w == 0 {
                            return Err(fail("empty clipped tile"));
                        }
                        width.push(w);
                        origin.push(binding.clone());
                    } else {
                        let step = self
                            .steps
                            .get(var)
                            .ok_or_else(|| fail("missing elem step"))?;
                        width.push(1);
                        origin.push(format!("({binding}/{step})"));
                    }
                }
                None if part.atom() == Some("fulltile") => {
                    width.push(*extent);
                    origin.push("0".into());
                }
                _ => return Err(fail("unsupported index")),
            }
        }
        if !slots.is_empty() {
            return Err(fail("index slot absent from layout"));
        }
        Ok(Access {
            value: id.index(),
            dtype: value.dtype(),
            shape: value.shape().to_vec(),
            width,
            origin,
        })
    }
    fn shape(&self, e: &E) -> Result<Vec<usize>, EmitError> {
        if e.atom().is_some() {
            return Ok(Vec::new());
        }
        let a = args(e)?;
        match e.operator().unwrap() {
            "float_bits" => Ok(Vec::new()),
            "load" => Ok(self.access(&a[0], &a[1])?.width),
            "sqr" | "sqrt" | "sigmoid" | "relu" => self.shape(&a[0]),
            "rsum" => {
                let mut s = self.shape(&a[0])?;
                let axis = number(&a[1])?;
                if axis >= s.len() {
                    return Err(fail("reduction axis out of range"));
                }
                s.remove(axis);
                Ok(s)
            }
            "bcast" | "unsqueeze" => {
                let mut s = self.shape(&a[0])?;
                let axis = number(&a[1])?;
                if axis > s.len() {
                    return Err(fail("inserted axis out of range"));
                }
                s.insert(axis, if e.operator() == Some("bcast") { 0 } else { 1 });
                Ok(s)
            }
            "@" => {
                let l = self.shape(&a[0])?;
                let r = self.shape(&a[1])?;
                if l.len() != 2 || r.len() != 2 || l[1] != r[0] {
                    return Err(fail("matmul tile shape mismatch"));
                }
                Ok(vec![l[0], r[1]])
            }
            "+" | "-" | "*" | "/" => {
                let l = self.shape(&a[0])?;
                let r = self.shape(&a[1])?;
                if l.is_empty() {
                    return Ok(r);
                }
                if r.is_empty() {
                    return Ok(l);
                }
                if l.len() != r.len() {
                    return Err(fail("elementwise rank mismatch; use explicit bcast"));
                }
                l.iter()
                    .zip(r)
                    .map(|(&l, r)| {
                        if l == r || l <= 1 || r <= 1 {
                            Ok(l.max(r))
                        } else {
                            Err(fail("elementwise tile shape mismatch"))
                        }
                    })
                    .collect()
            }
            _ => Err(fail("unsupported expression")),
        }
    }
    fn offset(&self, a: &Access, coords: &[String]) -> String {
        super::indexing::offset(&a.shape, &a.origin, coords, self.storage(a.value))
    }
    fn expr(&mut self, e: &E, coords: &[String]) -> Result<String, EmitError> {
        if let Some(a) = e.atom() {
            let _: f32 = a.parse().map_err(|_| fail("invalid numeric literal"))?;
            return Ok(format!("float({a})"));
        }
        let a = args(e)?;
        let shape = self.shape(e)?;
        let coords: Vec<String> = if shape.is_empty() {
            Vec::new()
        } else {
            if shape.len() != coords.len() {
                return Err(fail("coordinate rank mismatch"));
            }
            shape
                .iter()
                .zip(coords)
                .map(|(&d, c)| if d <= 1 { "0".into() } else { c.clone() })
                .collect()
        };
        match e.operator().unwrap() {
            "float_bits" => Ok(format!("__uint_as_float({}u)", number(&a[0])?)),
            "load" => {
                let ac = self.access(&a[0], &a[1])?;
                if let Some(name) = self.direct.get(&ac.value) {
                    return Ok(format!("float(${{{name}}})"));
                }
                if self.storage(ac.value) == Storage::Register {
                    return Err(fail("register input has no fragment binding"));
                }
                Ok(format!(
                    "float(${{buffer_{}}}[{}])",
                    ac.value,
                    self.memory_offset(&ac, &coords)
                ))
            }
            "+" | "-" | "*" | "/" | "sqrt" | "sigmoid" | "sqr" | "relu" => {
                let x = self.expr(&a[0], &coords)?;
                let y = if a.len() == 2 {
                    format!("float y={};", self.expr(&a[1], &coords)?)
                } else {
                    String::new()
                };
                let expression = self
                    .scalar
                    .and_then(|backend| backend.scalar_expression(e.operator().unwrap()))
                    .ok_or_else(|| {
                        fail("selected implementation does not support this scalar expression")
                    })?;
                Ok(format!(
                    "([&]() {{ float x={x}; {y} return {expression}; }}())"
                ))
            }
            "bcast" | "unsqueeze" => {
                let mut c = coords;
                c.remove(number(&a[1])?);
                self.expr(&a[0], &c)
            }
            "rsum" => {
                let axis = number(&a[1])?;
                let size = self.shape(&a[0])?[axis];
                let name = format!("r{}", self.next);
                self.next += 1;
                let mut c = coords;
                c.insert(axis, name.clone());
                let x = self.expr(&a[0], &c)?;
                Ok(format!(
                    "([&]() {{ float sum=0.0f; for(std::int64_t {name}=0;{name}<{size};++{name}) sum += {x}; return sum; }}())"
                ))
            }
            "@" => Err(fail(
                "matmul requires a selected serial accumulation implementation",
            )),
            _ => Err(fail("unsupported scalar expression")),
        }
    }
    // A multiply that used to end at an FP32 store must not contract with an
    // addition in the next operation. Explicit RN intrinsics keep that boundary
    // while leaving contractions within the original operand expressions alone.
    fn rounded_expr(
        &mut self,
        e: &E,
        coords: &[String],
        output: &Access,
    ) -> Result<String, EmitError> {
        if output.dtype == DType::Fp32 && self.storage(output.value) == Storage::Register {
            let intrinsic = match e.operator() {
                Some("*" | "sqr") => Some("__fmul_rn"),
                Some("/") => Some("__fdiv_rn"),
                _ => None,
            };
            if let Some(intrinsic) = intrinsic {
                let xs = args(e)?;
                let x = self.expr(&xs[0], coords)?;
                let y = if xs.len() == 1 {
                    x.clone()
                } else {
                    self.expr(&xs[1], coords)?
                };
                return Ok(format!("{intrinsic}({x},{y})"));
            }
        }
        self.expr(e, coords)
    }
    fn store(
        &mut self,
        id: OperationId,
        e: &E,
        accum: Option<(&Loop, &E)>,
        followers: &[Statement],
    ) -> Result<Body, EmitError> {
        if let Some((l, _)) = accum {
            self.names
                .insert(l.domain.variable.clone(), l.domain.variable.clone());
            self.steps.insert(
                l.domain.variable.clone(),
                l.domain.step.evaluate(&BTreeMap::new()).map_err(fail)?,
            );
        }
        let a = args(e)?;
        let output = self.access(&a[0], &a[2])?;
        let initialized = accum.is_some() && self.initialized(output.value);
        self.allocate_shared(&output);
        let rhs = accum.map(|(_, e)| e).unwrap_or(&a[1]);
        let shape = self.shape(rhs)?;
        if shape.len() != output.width.len()
            || shape
                .iter()
                .zip(&output.width)
                .any(|(&a, &b)| a != b && a != 0)
        {
            return Err(fail(format!(
                "store tile shape mismatch: {shape:?} vs {:?}",
                output.width
            )));
        }
        let count: usize = output.width.iter().product();
        if accum.is_none() && count == 1 && rhs.operator() == Some("rsum") {
            let reduction = args(rhs)?;
            let axis = number(&reduction[1])?;
            let length = self.shape(&reduction[0])?[axis];
            let mut coordinates = vec!["0".to_owned(); output.width.len()];
            coordinates.insert(axis, "column".into());
            let value = self.expr(&reduction[0], &coordinates)?;
            let offset = self.offset(&output, &vec!["0".to_owned(); output.width.len()]);
            self.resources.shared_memory_bytes = self.resources.shared_memory_bytes.max(512);
            let text = format!(
                "{{ float sum=0.0f; for(std::int64_t column=threadIdx.x;column<{length};column+=128) sum+={value}; auto partial=static_cast<float*>(memory); partial[threadIdx.x]=sum; __syncthreads(); for(unsigned stride=64;stride;stride>>=1) {{ if(threadIdx.x<stride) partial[threadIdx.x]+=partial[threadIdx.x+stride]; __syncthreads(); }} if(threadIdx.x==0) ${{buffer_{}}}[{offset}]={} (partial[0]); __syncthreads(); }}",
                output.value,
                ctype(output.dtype)
            );
            let mut body = self.scalar_body(&text, false, id)?;
            self.interface(
                &mut body,
                output.value,
                &format!("buffer_{}", output.value),
                true,
            );
            self.finish_bindings(&mut body)?;
            return Ok(body);
        }

        let coords: Vec<String> = (0..output.width.len())
            .map(|i| {
                format!(
                    "(linear/{})%{}",
                    output.width[i + 1..].iter().product::<usize>(),
                    output.width[i]
                )
            })
            .collect();
        let offset = self.offset(&output, &coords);
        let mut code = format!(
            "{{ for(std::int64_t linear=threadIdx.x;linear<{count};linear+=blockDim.x) {{\n"
        );
        if let Some((l, _)) = accum {
            let var = &l.domain.variable;
            let initial = if initialized {
                format!("float(${{buffer_{}}}[{offset}])", output.value)
            } else {
                "0.0f".into()
            };
            let start = l.domain.start.cpp(&self.names).map_err(fail)?;
            let stop = l.domain.stop.cpp(&self.names).map_err(fail)?;
            let step = l.domain.step.cpp(&self.names).map_err(fail)?;
            self.names.insert(var.clone(), var.clone());
            self.steps.insert(
                var.clone(),
                l.domain.step.evaluate(&BTreeMap::new()).map_err(fail)?,
            );
            let value = self.expr(rhs, &coords)?;
            writeln!(code,"float acc={initial}; for(std::int64_t {var}={start};{var}<{stop};{var}+={step}) {{ acc += {value}; }}").unwrap();
            self.names.remove(var);
            self.steps.remove(var);
        } else {
            let value = self.rounded_expr(rhs, &coords, &output)?;
            writeln!(code, "float acc={value};").unwrap();
        }
        code.push_str("${STORE}\n} __syncthreads(); }\n");
        let mut body = self.scalar_body(&code, accum.is_some(), id)?;
        let output_code = self.output(&mut body, &output, &coords, "${acc}", followers)?;
        body.bind(body.symbol("STORE")?, output_code);
        if initialized {
            self.interface(
                &mut body,
                output.value,
                &format!("buffer_{}", output.value),
                false,
            );
        }
        if let Some(mainloop) = &mut body.mainloop {
            mainloop.inputs.append(&mut body.epilogue.inputs);
            mainloop.outputs.append(&mut body.epilogue.outputs);
        }
        self.finish_bindings(&mut body)?;
        Ok(body)
    }
    fn accumulation(&self, l: &Loop) -> Option<(&'p E, &'p E)> {
        let [Statement::Operation(id)] = l.body.as_slice() else {
            return None;
        };
        let store = self.plan.operation(*id)?.expression()?;
        crate::physical::accumulation_rhs(store, &l.domain.variable).map(|rhs| (store, rhs))
    }
    fn statements(&mut self, statements: &[Statement]) -> Result<Body, EmitError> {
        let mut bodies = Vec::new();
        for (position, statement) in statements.iter().enumerate() {
            match statement {
                Statement::Operation(id) => {
                    if self.consumed.contains(id) {
                        continue;
                    }
                    let mut phases = self.select(*id)?;
                    let op = self.plan.operation(*id).unwrap();
                    if matches!(op.payload(), crate::OperationPayload::Communication(_)) {
                        let mut coordinate = vec!["0".to_owned(); 3];
                        for (i, e) in op.coordinates().iter().enumerate() {
                            coordinate[i] = e.cpp(&self.names).map_err(fail)?;
                        }
                        let mut head = Code::text(format!(
                            "{{ longlong3 tile_coord{{{},{},{}}}; ",
                            coordinate[0], coordinate[1], coordinate[2]
                        ));
                        head.append(&phases.prologue.code);
                        phases.prologue.code = head;
                        phases
                            .epilogue
                            .code
                            .append(&Code::text(" __syncthreads(); }\n"));
                        phases.alpha_rename(&mut self.next);
                        bodies.push(phases);
                    } else {
                        let e = op
                            .expression()
                            .ok_or_else(|| fail("compute expression is missing"))?;
                        let mut body = self.store(*id, e, None, &statements[position + 1..])?;
                        body.alpha_rename(&mut self.next);
                        phases.alpha_rename(&mut self.next);
                        phases.prologue.append(&body.prologue);
                        body.prologue = phases.prologue;
                        body.epilogue.append(&phases.epilogue);
                        bodies.push(body);
                    }
                }

                Statement::Loop(l) => {
                    if l.kind != LoopKind::Sequential {
                        return Err(fail("parallel loop in device body"));
                    }
                    if let Some((store, rhs)) = self.accumulation(l) {
                        self.select(l.body[0].operations()[0])?;
                        let matmul = if rhs.operator() == Some("unsqueeze") {
                            let a = args(rhs)?;
                            if number(&a[1])? != 0 {
                                return Err(fail("only leading unsqueeze around matmul supported"));
                            }
                            &a[0]
                        } else {
                            rhs
                        };
                        if matmul.operator() == Some("@") {
                            bodies.push(self.gemm(
                                l,
                                store,
                                matmul,
                                &statements[position + 1..],
                            )?);
                        } else {
                            let mut body = self.store(
                                l.body[0].operations()[0],
                                store,
                                Some((l, rhs)),
                                &statements[position + 1..],
                            )?;
                            body.alpha_rename(&mut self.next);
                            bodies.push(body);
                        }
                    } else {
                        let start = l.domain.start.cpp(&self.names).map_err(fail)?;
                        let stop = l.domain.stop.cpp(&self.names).map_err(fail)?;
                        let step = l.domain.step.cpp(&self.names).map_err(fail)?;
                        let var = &l.domain.variable;
                        self.names.insert(var.clone(), var.clone());
                        self.steps.insert(
                            var.clone(),
                            l.domain.step.evaluate(&BTreeMap::new()).map_err(fail)?,
                        );
                        let mut inner = self.statements(&l.body)?.into_phase();
                        let mut head = Code::text(format!(
                            "for(std::int64_t {var}={start};{var}<{stop};{var}+={step}) {{ "
                        ));
                        head.append(&inner.code);
                        head.append(&Code::text(" }\n"));
                        inner.code = head;
                        bodies.push(Body {
                            mainloop: Some(inner),
                            ..Body::default()
                        });
                        self.names.remove(var);
                        self.steps.remove(var);
                    }
                }
            }
        }
        Ok(Body::sequence(bodies))
    }
    fn scalar_body(&self, code: &str, repeated: bool, id: OperationId) -> Result<Body, EmitError> {
        let mut names: Vec<String> = [
            "linear", "acc", "sum", "column", "partial", "stride", "x", "y", "STORE",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        names.extend(
            self.plan
                .value_instances()
                .map(|(id, _)| format!("buffer_{}", id.index())),
        );
        let mut body = Body::cuda(
            code,
            repeated,
            &names.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;
        body.epilogue.resources = self.resources.clone();
        let op = self.plan.operation(id).unwrap();
        for value in op.inputs() {
            self.interface(
                &mut body,
                value.index(),
                &format!("buffer_{}", value.index()),
                false,
            );
        }
        Ok(body)
    }
    fn finish_bindings(&self, body: &mut Body) -> Result<(), EmitError> {
        for (id, v) in self.plan.value_instances() {
            if matches!(v.storage(), Storage::Global | Storage::External) {
                let name = format!("buffer_{}", id.index());
                if let Ok(symbol) = body.symbol(&name) {
                    body.bind(symbol, Code::text(self.pointer(id.index())?));
                }
            }
        }
        Ok(())
    }
    fn accumulation_tile(&self, access: &Access) -> Result<super::AccumulationTile, EmitError> {
        let (memory_shape, memory_origin) = if let Some((owner, _)) = self.shared.get(&access.value)
        {
            (
                owner.width.clone(),
                access
                    .origin
                    .iter()
                    .zip(&owner.origin)
                    .map(|(a, b)| format!("({a})-({b})"))
                    .collect(),
            )
        } else {
            (access.shape.clone(), access.origin.clone())
        };
        let storage = self.storage(access.value);
        Ok(super::AccumulationTile {
            value: ValueInstanceId::from_index(access.value),
            dtype: access.dtype,
            storage,
            width: access.width.clone(),
            origin: access.origin.clone(),
            memory_shape,
            memory_origin,
            pointer: if matches!(storage, Storage::Global | Storage::External) {
                Some(self.pointer(access.value)?)
            } else {
                None
            },
        })
    }
    fn gemm(
        &mut self,
        l: &Loop,
        store: &E,
        matmul: &E,
        followers: &[Statement],
    ) -> Result<Body, EmitError> {
        let var = &l.domain.variable;
        let start = l.domain.start.cpp(&self.names).map_err(fail)?;
        let stop = l.domain.stop.cpp(&self.names).map_err(fail)?;
        let step = l.domain.step.evaluate(&BTreeMap::new()).map_err(fail)?;
        self.names.insert(var.clone(), var.clone());
        self.steps.insert(var.clone(), step);
        let mm = args(matmul)?;
        if mm.iter().any(|e| e.operator() != Some("load")) {
            return Err(fail("accumulation operands must be tile loads"));
        }
        let aa = args(&mm[0])?;
        let bb = args(&mm[1])?;
        let st = args(store)?;
        let a = self.access(&aa[0], &aa[1])?;
        let b = self.access(&bb[0], &bb[1])?;
        let c = self.access(&st[0], &st[2])?;
        let [Statement::Operation(id)] = l.body.as_slice() else {
            return Err(fail("accumulation scope must own one operation"));
        };
        let crate::OperationPayload::Compute(compute) = self.plan.operation(*id).unwrap().payload()
        else {
            return Err(fail("expected compute implementation"));
        };
        let backend = compute
            .implementation()
            .definition()
            .cuda()
            .ok_or_else(|| fail("missing CUDA backend"))?;
        self.allocate_shared(&c);
        let scope = super::AccumulationScope {
            variable: var.clone(),
            start,
            stop,
            step,
            inputs: [self.accumulation_tile(&a)?, self.accumulation_tile(&b)?],
            output: self.accumulation_tile(&c)?,
            initialized: self.initialized(c.value),
        };
        let super::AccumulationBody {
            body: mut phases,
            output_coordinates,
            output_expression,
            output_symbol,
        } = backend.accumulation(self.plan, *id, &scope)?;
        let output = self.output(
            &mut phases,
            &c,
            &output_coordinates,
            &output_expression,
            followers,
        )?;
        phases.bind(output_symbol, output);
        self.finish_bindings(&mut phases)?;
        self.resources.sequential(&phases.resources());
        phases.alpha_rename(&mut self.next);
        self.names.remove(var);
        self.steps.remove(var);
        Ok(phases)
    }
}
fn ctype(dtype: DType) -> &'static str {
    match dtype {
        DType::Bf16 => "cutlass::bfloat16_t",
        DType::Fp32 => "float",
    }
}
pub(super) fn render(
    plan: &PhysicalPlan,
    id: usize,
    body: &TaskDomain,
) -> Result<(String, Body), EmitError> {
    validate_local_scopes(plan, &body.statements)?;
    let variables: Vec<_> = body.bindings[0].keys().cloned().collect();
    let mut code = String::new();
    if !variables.is_empty() {
        writeln!(
            code,
            "__device__ const std::int64_t body_{id}_args[{}][{}] = {{",
            body.bindings.len(),
            variables.len()
        )
        .unwrap();
        for b in &body.bindings {
            writeln!(
                code,
                "{{{}}},",
                variables
                    .iter()
                    .map(|v| b[v].to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .unwrap();
        }
        code.push_str("};\n");
    }
    writeln!(code,"template<class Runtime> __device__ bool operation_{id}(Bindings const& bindings, Tile const& tile_arguments, void* memory, Runtime const& runtime) {{").unwrap();
    for (i, v) in variables.iter().enumerate() {
        writeln!(
            code,
            "std::int64_t {v}=body_{id}_args[tile_arguments.x][{i}];"
        )
        .unwrap();
    }
    // Sequential implementations reuse scratch. Promoted values must survive
    // across those calls, so place them after the largest selected requirement.
    // The common scalar row reduction uses 128 FP32 partials.
    let mut scratch_bytes = 128 * DType::Fp32.size_bytes();
    for operation in body.statements.iter().flat_map(Statement::operations) {
        let op = plan.operation(operation).unwrap();
        let instance = match op.payload() {
            crate::OperationPayload::Compute(c) => c.implementation(),
            crate::OperationPayload::Communication(c) => c.implementation(),
        };
        let backend = instance
            .definition()
            .cuda()
            .ok_or_else(|| fail("missing CUDA backend"))?;
        scratch_bytes = scratch_bytes.max(
            backend
                .phases(plan, operation)?
                .resources()
                .shared_memory_bytes,
        );
    }
    let scratch_bytes = scratch_bytes.div_ceil(128) * 128;
    let mut context = Context {
        plan,
        names: variables.iter().map(|v| (v.clone(), v.clone())).collect(),
        steps: body.steps.clone(),
        next: 0,
        resources: Resources::default(),
        concrete: body.bindings[0].clone(),
        scalar: None,
        current_operation: 0,
        direct: BTreeMap::new(),
        consumed: BTreeSet::new(),
        shared: BTreeMap::new(),
        local_bytes: 0,
        scratch_bytes,
    };
    let mut phases = context.statements(&body.statements)?;
    phases.prologue.resources.sequential(&context.resources);
    // Connect shared fragment pointers across independently renamed operation scopes.
    for &value in context.shared.keys() {
        let parameters: BTreeSet<_> = phases
            .phases()
            .flat_map(|p| p.inputs.iter().chain(&p.outputs))
            .filter(|b| b.value.index() == value)
            .map(|b| b.symbol)
            .collect();
        let name = format!("shared_value_{value}");
        let pointer = phases.declare(&name, false);
        for parameter in parameters {
            phases.substitute(parameter, pointer);
        }
        let mut declaration =
            phases.code(&format!("auto* ${{{name}}}={};\n", context.pointer(value)?))?;
        declaration.append(&phases.prologue.code);
        phases.prologue.code = declaration;
    }
    let cursors: BTreeSet<_> = phases
        .phases()
        .flat_map(|p| &p.symbols)
        .filter(|s| s.name == "stage_cursor")
        .map(|s| s.id)
        .collect();
    if !cursors.is_empty() {
        let cursor = phases.declare("task_stage_cursor", false);
        for parameter in cursors {
            phases.substitute(parameter, cursor);
        }
        let mut declaration = phases.code("int ${task_stage_cursor}=0;\n")?;
        declaration.append(&phases.prologue.code);
        phases.prologue.code = declaration;
    }
    code.push_str(&phases.render()?);
    code.push_str("return true; }\n");
    Ok((code, phases))
}

fn validate_local_scopes(plan: &PhysicalPlan, statements: &[Statement]) -> Result<(), EmitError> {
    fn visit(
        plan: &PhysicalPlan,
        statements: &[Statement],
        scope: &mut Vec<String>,
        owners: &mut BTreeMap<ValueInstanceId, Vec<String>>,
    ) -> Result<(), EmitError> {
        for statement in statements {
            match statement {
                Statement::Operation(id) => {
                    let op = plan.operation(*id).unwrap();
                    for value in op.inputs().iter().chain(op.outputs()) {
                        if matches!(
                            plan.value_instance(*value).unwrap().storage(),
                            Storage::Shared | Storage::Register
                        ) {
                            let owner = owners.entry(*value).or_insert_with(|| scope.clone());
                            if owner != scope {
                                return Err(fail(
                                    "local binding crosses a sequential state lifetime",
                                ));
                            }
                        }
                    }
                }
                Statement::Loop(l) => {
                    let accumulation = match l.body.as_slice() {
                        [Statement::Operation(id)] => plan
                            .operation(*id)
                            .and_then(|op| op.expression())
                            .is_some_and(|e| {
                                crate::physical::accumulation_rhs(e, &l.domain.variable).is_some()
                            }),
                        _ => false,
                    };
                    if !accumulation {
                        scope.push(l.domain.variable.clone());
                    }
                    visit(plan, &l.body, scope, owners)?;
                    if !accumulation {
                        scope.pop();
                    }
                }
            }
        }
        Ok(())
    }
    visit(plan, statements, &mut Vec::new(), &mut BTreeMap::new())
}
