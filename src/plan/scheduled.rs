//! Import the shared scheduled-IR model into an owned PhysicalPlan.
//! No source AST or alternative schedule is attached to the resulting plan.
use super::*;
use crate::{DType, analysis as a};
use std::collections::{BTreeMap, BTreeSet};

pub struct ScheduledConfig {
    pub target: crate::TargetCapability,
    pub bindings: a::Bindings,
    pub default_dtype: DType,
    pub dtypes: BTreeMap<String, DType>,
}

fn invalid(s: impl Into<String>) -> PhysicalInvariantError {
    PhysicalInvariantError::InvalidProgram(s.into())
}

pub(crate) fn index(
    ir: &a::ScheduledIr,
    expr: &a::IndexExpr,
) -> Result<IndexExpr, PhysicalInvariantError> {
    use a::IndexExpr as I;
    Ok(match expr {
        I::Integer(n) => IndexExpr::Constant(*n),
        I::Symbol(s) => IndexExpr::Variable(s.clone()),
        I::LoopVar(s) => {
            IndexExpr::Variable(ir.scope(*s).loop_info.as_ref().unwrap().variable.clone())
        }
        I::Apply(op, args) if args.len() == 2 => {
            let x = Box::new(index(ir, &args[0])?);
            let y = Box::new(index(ir, &args[1])?);
            match op.as_str() {
                "+" => IndexExpr::Add(x, y),
                "-" => IndexExpr::Sub(x, y),
                "*" => IndexExpr::Mul(x, y),
                "/" | "//" => IndexExpr::Div(x, y),
                _ => return Err(invalid(format!("unsupported scalar operator {op}"))),
            }
        }
        _ => return Err(invalid("invalid index expression")),
    })
}

impl PhysicalPlanBuilder {
    /// Import a selected computation graph, retaining source kernel boundaries,
    /// lexical loops, value identities, per-access views and symbolic parameters.
    pub fn from_scheduled(
        ir: &a::ScheduledIr,
        mut config: ScheduledConfig,
    ) -> Result<PhysicalPlan, PhysicalInvariantError> {
        let metadata = a::TensorMetadata::collect(ir, &mut config.bindings)
            .map_err(|e| invalid(e.to_string()))?;
        let facts = a::ProgramFacts::resolve(ir, &mut config.bindings, metadata)
            .map_err(|e| invalid(e.to_string()))?;
        let storage = a::storage::infer(ir, &config.bindings, &facts.kernels)
            .map_err(|e| invalid(e.to_string()))?;
        let mut b = Self::new(config.target, 1);
        let mut ids = Vec::new();
        let mut outputs = Vec::new();
        let mut mutations = BTreeSet::new();
        for (i, t) in ir.tensors().iter().enumerate() {
            let tid = a::TensorId(i);
            let input = t.declarations.contains(&a::TensorKind::Input);
            let output = t.declarations.contains(&a::TensorKind::Output);
            let dtype = config
                .dtypes
                .get(&t.name)
                .copied()
                .unwrap_or(config.default_dtype);
            let id = b.add_named_value(
                &t.name,
                dtype,
                config.bindings.shapes[&t.name].clone(),
                storage.values[&tid],
            );
            b.values.values[id.index()].dtype_explicit =
                input || output || config.dtypes.contains_key(&t.name);
            // An explicitly supplied ABI shape remains the caller's shape. Keep
            // symbolic dimensions only when they describe that same base view.
            let dims = &facts.metadata.shapes[&tid];
            if dims.len() == b.values.values[id.index()].shape.len() {
                b.values.values[id.index()].dimensions = Some(
                    dims.iter()
                        .map(|d| index(ir, d))
                        .collect::<Result<_, _>>()?,
                );
            }
            if input {
                b.bind_input(&t.name, id);
            }
            if output {
                outputs.push(TensorBinding::new(&t.name, id));
            }
            if ir.mutated_inputs().contains(&tid) {
                mutations.insert(id);
            }
            ids.push(id);
        }
        let accesses: Vec<_> = ir
            .accesses()
            .iter()
            .map(|access| {
                let id = ids[access.tensor.index()];
                let width = |expr: &a::IndexExpr| -> Result<TileWidth, PhysicalInvariantError> {
                    Ok(match expr {
                        a::IndexExpr::Symbol(s) => TileWidth::Symbol(s.clone()),
                        _ if !a::scalar::symbols(expr).is_empty() => {
                            return Err(invalid("tile width must be a constant or a single configuration symbol; refusing to freeze a parameterized expression"));
                        }
                        _ => TileWidth::Constant(
                            a::scalar::positive(expr, &config.bindings.symbols)
                                .map_err(|e| invalid(e.to_string()))?,
                        ),
                    })
                };
                let indices = access
                    .index
                    .iter()
                    .map(|i| {
                        Ok(match i {
                            a::IndexDim::FullTile => AccessIndex::FullTile,
                            a::IndexDim::Elem(a::IndexExpr::LoopVar(s)) => AccessIndex::Elem(
                                ir.scope(*s).loop_info.as_ref().unwrap().variable.clone(),
                            ),
                            a::IndexDim::Elem(expr) => AccessIndex::Element(index(ir, expr)?),
                            a::IndexDim::Tile {
                                start: a::IndexExpr::LoopVar(s),
                                width: w,
                            } => AccessIndex::Tile {
                                variable: ir.scope(*s).loop_info.as_ref().unwrap().variable.clone(),
                                width: width(w)?,
                            },
                            a::IndexDim::Tile { start, width: w }
                            | a::IndexDim::ConstTile { start, width: w } => AccessIndex::Slice {
                                start: index(ir, start)?,
                                width: width(w)?,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, PhysicalInvariantError>>()?;
                let mut result = TensorAccess::new(id, indices);
                if let Some(shape) = &access.view_shape {
                    result.view_shape = Some(
                        shape
                            .iter()
                            .map(|e| {
                                a::scalar::positive(e, &config.bindings.symbols)
                                    .map_err(|e| invalid(e.to_string()))
                            })
                            .collect::<Result<_, _>>()?,
                    );
                    result.view_dimensions = Some(
                        shape
                            .iter()
                            .map(|d| index(ir, d))
                            .collect::<Result<_, _>>()?,
                    );
                }
                // Legacy bare fulltile applies to all base axes.
                if access.view_shape.is_none() && result.indices.as_ref() == [AccessIndex::FullTile]
                {
                    result.indices =
                        vec![AccessIndex::FullTile; b.values.values[id.index()].shape.len()].into();
                }
                Ok(result)
            })
            .collect::<Result<_, PhysicalInvariantError>>()?;
        let flow = &facts.kernels;
        for kernel in flow {
            for (tid, info) in &kernel.tensors {
                if info.entry_value == a::EntryValue::Unavailable {
                    return Err(invalid(format!(
                        "{}: first read is not a defined value or additive accumulator",
                        ir.tensor(*tid).name
                    )));
                }
            }
        }
        let mut ops = Vec::new();
        for (si, s) in ir.statements().iter().enumerate() {
            let destination = accesses[s.accesses.last().unwrap().index()].clone();
            let expr = expression(ir, &s.expression, &accesses)?;
            let reads = expr.reads();
            let op = b.add_operation(
                reads,
                [destination.value],
                Expression::Store {
                    destination: destination.clone(),
                    value: Box::new(expr),
                },
            );
            let info = &flow[ir.scope(s.scope).kernel.index()].tensors
                [&ir.access(*s.accesses.last().unwrap()).tensor];
            if info.entry_value == a::EntryValue::ZeroRecurrence
                && ir.access(info.accesses[0]).statement.index() == si
            {
                b.operations.values[op.index()]
                    .zero_init
                    .push(destination.value);
            }
            ops.push(op);
        }
        fn scope(
            ir: &a::ScheduledIr,
            id: a::ScopeId,
            ops: &[OperationId],
        ) -> Result<Vec<Statement>, PhysicalInvariantError> {
            ir.scope(id)
                .children
                .iter()
                .map(|item| {
                    Ok(match item {
                        a::ScopeItem::Statement(s) => Statement::Operation(ops[s.index()]),
                        a::ScopeItem::Scope(s) => {
                            let info = ir.scope(*s);
                            let l = info.loop_info.as_ref().unwrap();
                            Statement::Loop(Loop {
                                kind: match info.kind {
                                    a::ScopeKind::ParallelLoop => LoopKind::Parallel,
                                    a::ScopeKind::SplitLoop => LoopKind::Split,
                                    _ => LoopKind::Sequential,
                                },
                                domain: LoopDomain {
                                    variable: l.variable.clone(),
                                    start: index(ir, &l.start)?,
                                    stop: index(ir, &l.end)?,
                                    step: index(ir, &l.step)?,
                                },
                                body: scope(ir, *s, ops)?,
                            })
                        }
                    })
                })
                .collect()
        }
        let statements = ir
            .kernels()
            .iter()
            .map(|k| scope(ir, k.root_scope, &ops).map(Statement::Region))
            .collect::<Result<_, _>>()?;
        b.build_program(statements, outputs, mutations, config.bindings.symbols)
    }
}

fn expression(
    ir: &a::ScheduledIr,
    value: &a::ValueExpr,
    accesses: &[TensorAccess],
) -> Result<Expression, PhysicalInvariantError> {
    use Expression as E;
    use a::ValueExpr as V;
    Ok(match value {
        V::Load(id) => E::Load(accesses[id.index()].clone()),
        V::Index(i) => E::Index(index(ir, i)?),
        V::Literal(s) => E::Constant(if let Ok(i) = s.parse() {
            Constant::Integer(i)
        } else {
            Constant::Float64(
                s.parse::<f64>()
                    .map_err(|_| invalid("invalid literal"))?
                    .to_bits(),
            )
        }),
        V::Apply(op, args) => {
            if op == "const" && args.len() == 1 {
                return expression(ir, &args[0], accesses);
            }
            if op == "cast" && args.len() == 2 {
                let V::Index(a::IndexExpr::Symbol(dtype)) = &args[0] else {
                    return Err(invalid("cast dtype"));
                };
                return Ok(E::Apply {
                    op: ValueOp::Cast(dtype.clone()),
                    args: vec![expression(ir, &args[1], accesses)?].into(),
                });
            }
            let mut values = args
                .iter()
                .map(|e| expression(ir, e, accesses))
                .collect::<Result<Vec<_>, _>>()?;
            if ["+", "-", "*", "/", "@"].contains(&op.as_str()) && values.len() == 2 {
                let [x, y]: [E; 2] = values.try_into().unwrap();
                let pair = Box::new([x, y]);
                match op.as_str() {
                    "+" => E::Add(pair),
                    "-" => E::Sub(pair),
                    "*" => E::Mul(pair),
                    "/" => E::Div(pair),
                    _ => E::Matmul(pair),
                }
            } else if ["sqr", "sqrt", "sigmoid"].contains(&op.as_str()) && values.len() == 1 {
                let v = Box::new(values.remove(0));
                match op.as_str() {
                    "sqr" => E::Sqr(v),
                    "sqrt" => E::Sqrt(v),
                    _ => E::Sigmoid(v),
                }
            } else {
                let op = match op.as_str() {
                    "exp" => ValueOp::Exp,
                    "erf" => ValueOp::Erf,
                    "abs" => ValueOp::Abs,
                    "transpose" => ValueOp::Transpose,
                    "permute" | "permute3" | "permute4" => ValueOp::Permute,
                    "rsum" => ValueOp::ReduceSum,
                    "rmax" => ValueOp::ReduceMax,
                    "rmin" => ValueOp::ReduceMin,
                    "bcast" => ValueOp::Broadcast,
                    "unsqueeze" => ValueOp::Unsqueeze,
                    "squeeze" => ValueOp::Squeeze,
                    "concat" => ValueOp::Concat,
                    "<=" => ValueOp::LessEqual,
                    "max" => ValueOp::Maximum,
                    "min" => ValueOp::Minimum,
                    _ => return Err(invalid(format!("unsupported expression {op}"))),
                };
                E::Apply {
                    op,
                    args: values.into(),
                }
            }
        }
    })
}
