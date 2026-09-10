//! Build per-kernel storage, initialization, publication and access bindings.
use super::super::plan::{InitialValue, Initialization, KernelPlan, Storage, TensorPlan};
use super::super::shape::{loop_range, product};
use super::super::{Error, Options, invalid};
use super::dependencies::{covers, unowned_axes, validate_materialized_reads};
use crate::analysis::dependencies::{common_scope, same_region, tensor_loop_dependencies};
use crate::analysis::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn plan_kernel(
    ki: usize,
    ir: &ProgramAnalysis,
    options: &Options,
    globals: &mut BTreeSet<TensorId>,
    previously_written: &mut BTreeSet<TensorId>,
    previous_definitions: &mut Vec<AccessId>,
) -> Result<KernelPlan, Error> {
    let kernel = &ir.kernels()[ki];
    let mut parallel = Vec::new();
    for (si, scope) in ir
        .scopes()
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kernel.index() == ki && s.loop_info.is_some())
    {
        super::loops::validate(ir, ScopeId(si), options)?;
        if scope.kind.is_parallel() {
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
            let (start, end, step) = loop_range(ir, *id, options)?;
            Ok(((end - start) as usize).div_ceil(step))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    product(&grid)?;
    let grid_extents = parallel
        .iter()
        .map(|id| {
            let info = ir.scope(*id).loop_info.as_ref().unwrap();
            if ir.scope(*id).kind == ScopeKind::SplitLoop {
                return info.end.clone();
            }
            let apply = |op: &str, a, b| IndexExpr::Apply(op.into(), vec![a, b]);
            apply(
                "//",
                apply(
                    "-",
                    apply(
                        "+",
                        apply("-", info.end.clone(), info.start.clone()),
                        info.step.clone(),
                    ),
                    IndexExpr::Integer(1),
                ),
                info.step.clone(),
            )
        })
        .collect();
    let dependencies = tensor_loop_dependencies(ir, kernel);
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
        let local_subviews = !all_same
            && !writes.is_empty()
            && writes
                .iter()
                .all(|w| same_region(ir.access(*w), ir.access(representative)))
            && uses
                .iter()
                .filter(|a| ir.access(**a).kind == AccessKind::Read)
                .all(|r| {
                    writes
                        .iter()
                        .rev()
                        .any(|w| super::access::local_read(ir, *w, *r, options).is_some())
                        || (same_region(ir.access(*r), ir.access(representative))
                            && additive_update(
                                ir,
                                ir.access(representative).statement,
                                representative,
                            ))
                });
        let storage = if input || writes.is_empty() {
            Storage::Global
        } else if all_same || local_subviews {
            Storage::Register
        } else {
            Storage::Materialized
        };
        if storage != Storage::Register || publish {
            globals.insert(tensor);
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
                while ir.scope(scope).kind == ScopeKind::SequentialLoop && !deps.contains(&scope) {
                    scope = ir.scope(scope).parent.unwrap();
                }
            }
            initialization = Some(Initialization {
                scope,
                access: representative,
                value,
            });
            if value == InitialValue::Global {
                globals.insert(tensor);
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
                options,
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
                        && covers(ir, ir.access(*id), ir.access(read), options)
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
                if !unowned_axes(ir, ir.access(*read), &parallel, options)?.is_empty() {
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
                for axis in unowned_axes(ir, ir.access(*write), &parallel, options)? {
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
    let mut local_reads = BTreeMap::new();
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
                if ir.access(access).kind == AccessKind::Read
                    && let Some(binding) = kernel
                        .accesses
                        .iter()
                        .rev()
                        .filter(|w| ir.access(**w).kind == AccessKind::Write)
                        .find_map(|w| super::access::local_read(ir, *w, access, options))
                {
                    local_reads.insert(access, binding);
                }
            }
        }
    }
    Ok(KernelPlan {
        root_scope: kernel.root_scope,
        parallel_loops: parallel,
        grid,
        grid_extents,
        tensors,
        register_accesses,
        local_reads,
    })
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
