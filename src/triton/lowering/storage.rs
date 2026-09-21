//! Build per-kernel storage, initialization, publication and access bindings.
use super::super::plan::{InitialValue, Initialization, KernelPlan, Storage, TensorPlan};
use super::super::shape::{loop_range, product};
use super::super::{Error, Options, invalid};
use super::dependencies::{unowned_axes, validate_materialized_reads};
use crate::analysis::access::covers;
use crate::analysis::dependencies::{common_scope, same_region};
use crate::analysis::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn plan_kernel(
    ki: usize,
    ir: &ScheduledIr,
    options: &Options,
    globals: &mut BTreeSet<TensorId>,
    bindings: &Bindings,
    facts: &KernelDataflow,
) -> Result<KernelPlan, Error> {
    let kernel = &ir.kernels()[ki];
    let mut parallel = Vec::new();
    for (si, scope) in ir
        .scopes()
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kernel.index() == ki && s.loop_info.is_some())
    {
        super::loops::validate(ir, ScopeId(si))?;
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
    let dependencies = &facts.loop_dependencies;
    let mut tensors = BTreeMap::new();
    for (&tensor, flow) in &facts.tensors {
        let uses = &flow.accesses;
        let writes = &flow.writes;
        let input = flow.input;
        let publish = flow.live_out;
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
                        .any(|w| super::access::local_read(ir, *w, *r, bindings).is_some())
                        || (same_region(ir.access(*r), ir.access(representative))
                            && flow
                                .additive_updates
                                .contains(&ir.access(representative).statement))
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
        if writes.is_empty() && !input && !flow.previously_written {
            return Err(invalid(format!(
                "{}: read before any producer",
                ir.tensor(tensor).name
            )));
        }
        let first = ir.access(uses[0]);
        let mut initialization = None;
        if !writes.is_empty() && !input && first.kind == AccessKind::Read {
            let value = match flow.entry_value {
                EntryValue::EarlierKernel => InitialValue::Global,
                EntryValue::ZeroRecurrence => InitialValue::Zero,
                _ => {
                    return Err(invalid(format!(
                        "{}: first read is not a defined value or additive accumulator",
                        ir.tensor(tensor).name
                    )));
                }
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
            let common = flow.common_scope;
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
                uses,
                representative,
                initialization.as_ref(),
                bindings,
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
                if !facts.previous_definitions.iter().any(|id| {
                    ir.access(*id).tensor == tensor
                        && covers(ir, ir.access(*id), ir.access(read), bindings)
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
                if !unowned_axes(ir, ir.access(*read), &parallel, bindings)?.is_empty() {
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
                    .unwrap_or(flow.common_scope),
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
            for write in writes {
                for axis in unowned_axes(ir, ir.access(*write), &parallel, bindings)? {
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
        let accumulators = flow.additive_updates.clone();
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
                        .find_map(|w| super::access::local_read(ir, *w, access, bindings))
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
