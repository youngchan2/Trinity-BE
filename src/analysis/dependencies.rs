//! Queries over collected IR facts, independent of Triton storage decisions.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn common_scope(ir: &ScheduledIr, uses: &[AccessId]) -> ScopeId {
    let mut scope = ir.access(uses[0]).scope;
    for access in &uses[1..] {
        while !ir.is_within(ir.access(*access).scope, scope) {
            scope = ir.scope(scope).parent.unwrap();
        }
    }
    scope
}

/// Tensor-level loop-variable dependence; not a versioned definition-use graph.
pub(crate) fn tensor_loop_dependencies(
    ir: &ScheduledIr,
    kernel: &KernelInfo,
) -> BTreeMap<TensorId, BTreeSet<ScopeId>> {
    fn expr_deps(
        expr: &ValueExpr,
        ir: &ScheduledIr,
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
                let mut incoming = expr_deps(&ir.statement(a.statement).expression, ir, &deps);
                loop {
                    let old = incoming.len();
                    let bounds: BTreeSet<_> = incoming
                        .iter()
                        .flat_map(|scope| {
                            let info = ir.scope(*scope).loop_info.as_ref().unwrap();
                            info.start
                                .loop_dependencies()
                                .into_iter()
                                .chain(info.end.loop_dependencies())
                        })
                        .collect();
                    incoming.extend(bounds);
                    if incoming.len() == old {
                        break;
                    }
                }
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

pub(crate) fn same_region(a: &AccessInfo, b: &AccessInfo) -> bool {
    a.index == b.index && a.view_shape == b.view_shape
}
