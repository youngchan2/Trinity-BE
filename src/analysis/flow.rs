//! Ordered uses, publication and recurrence patterns, without storage decisions.
use super::dependencies::{common_scope, same_region, tensor_loop_dependencies};
use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// Origin of a tensor's first value in a source kernel region. This preserves
/// the scheduled-source convention; it does not allocate or initialize storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryValue {
    Input,
    DefinedInKernel,
    EarlierKernel,
    ZeroRecurrence,
    Unavailable,
}

#[derive(Debug, Clone)]
pub struct TensorDataflow {
    /// Reads and writes in source evaluation order.
    pub accesses: Vec<AccessId>,
    pub writes: Vec<AccessId>,
    pub input: bool,
    /// Boundary tensor or consumed by a later source kernel region.
    pub live_out: bool,
    pub previously_written: bool,
    pub entry_value: EntryValue,
    /// Least enclosing lexical scope of this region's accesses.
    pub common_scope: ScopeId,
    /// Additive self-updates, including scaled self terms. This does not assert
    /// zero initialization or the narrower PhysicalPlan reduction convention.
    pub additive_updates: BTreeSet<StatementId>,
}

#[derive(Debug, Clone)]
pub struct KernelDataflow {
    pub tensors: BTreeMap<TensorId, TensorDataflow>,
    /// Ordered earlier writes, not a full versioned reaching-definition graph.
    pub previous_definitions: Vec<AccessId>,
    pub loop_dependencies: BTreeMap<TensorId, BTreeSet<ScopeId>>,
}

pub(super) fn analyze(ir: &ScheduledIr) -> Vec<KernelDataflow> {
    let mut previous_definitions = Vec::new();
    let mut previously_written = BTreeSet::new();
    let mut kernels = Vec::new();
    for (ki, kernel) in ir.kernels().iter().enumerate() {
        let mut tensors = BTreeMap::new();
        for tensor in kernel
            .read_writes
            .reads
            .union(&kernel.read_writes.writes)
            .copied()
        {
            let accesses: Vec<_> = kernel
                .accesses
                .iter()
                .copied()
                .filter(|a| ir.access(*a).tensor == tensor)
                .collect();
            let writes: Vec<_> = accesses
                .iter()
                .copied()
                .filter(|a| ir.access(*a).kind == AccessKind::Write)
                .collect();
            let input = ir.tensor(tensor).declarations.contains(&TensorKind::Input);
            let live_out = input
                || ir.tensor(tensor).declarations.contains(&TensorKind::Output)
                || ir.kernels()[ki + 1..]
                    .iter()
                    .any(|k| k.read_writes.reads.contains(&tensor));
            let additive_updates: BTreeSet<_> = writes
                .iter()
                .filter_map(|w| {
                    let statement = ir.access(*w).statement;
                    additive_update(ir, statement, *w).then_some(statement)
                })
                .collect();
            let first = ir.access(accesses[0]);
            let entry_value = if input {
                EntryValue::Input
            } else if first.kind == AccessKind::Write {
                EntryValue::DefinedInKernel
            } else if previously_written.contains(&tensor) {
                EntryValue::EarlierKernel
            } else if additive_updates.contains(&first.statement) {
                EntryValue::ZeroRecurrence
            } else {
                EntryValue::Unavailable
            };
            let scope = common_scope(ir, &accesses);
            tensors.insert(
                tensor,
                TensorDataflow {
                    accesses,
                    writes,
                    input,
                    live_out,
                    previously_written: previously_written.contains(&tensor),
                    entry_value,
                    common_scope: scope,
                    additive_updates,
                },
            );
        }
        kernels.push(KernelDataflow {
            tensors,
            previous_definitions: previous_definitions.clone(),
            loop_dependencies: tensor_loop_dependencies(ir, kernel),
        });
        previously_written.extend(kernel.read_writes.writes.iter().copied());
        previous_definitions.extend(
            kernel
                .accesses
                .iter()
                .copied()
                .filter(|a| ir.access(*a).kind == AccessKind::Write),
        );
    }
    kernels
}

fn additive_update(ir: &ScheduledIr, statement: StatementId, write: AccessId) -> bool {
    let target = ir.access(write);
    fn self_term(expr: &ValueExpr, ir: &ScheduledIr, target: &AccessInfo) -> bool {
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
