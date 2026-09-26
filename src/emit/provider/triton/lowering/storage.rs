//! Specialize the common storage plan with Triton grid and SSA requirements.
use super::super::plan::{InitialValue, Initialization, KernelPlan, Storage};
use super::super::shape::{loop_range, product};
use super::super::{Error, Options, invalid};
use crate::analysis::storage::KernelStoragePlan;
use crate::analysis::*;

pub(super) fn plan_kernel(
    ki: usize,
    ir: &ScheduledIr,
    options: &Options,
    facts: &KernelDataflow,
    storage: &KernelStoragePlan,
) -> Result<KernelPlan, Error> {
    let kernel = &ir.kernels()[ki];
    for read in storage.local_reads.values() {
        if read
            .split_last
            .is_some_and(|f| f.iter().any(|n| !n.is_power_of_two()))
        {
            return Err(invalid(
                "Triton register final-axis factorization currently requires power-of-two factors to preserve padding positions",
            ));
        }
    }
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
    let mut tensors = storage.tensors.clone();
    for (tensor, plan) in &mut tensors {
        if plan.initialization.is_none() && plan.storage == Storage::Register {
            let common = facts.tensors[tensor].common_scope;
            if common != ir.access(plan.representative).scope {
                // Triton SSA needs an incoming value for a definition inside a
                // loop that is used outside, even when the loop is nonempty.
                plan.initialization = Some(Initialization {
                    scope: common,
                    access: plan.representative,
                    value: InitialValue::Zero,
                });
            }
        }
    }
    Ok(KernelPlan {
        root_scope: kernel.root_scope,
        parallel_loops: parallel,
        grid,
        grid_extents,
        tensors,
        register_accesses: storage.register_accesses.clone(),
        local_reads: storage.local_reads.clone(),
    })
}
