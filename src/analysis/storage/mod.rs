//! Shared storage planning from ordered accesses, region boundaries and dependencies.
//! Value storage is the program contract; AccessMode describes accesses within a
//! region (a Global/External value may still have a local accumulator).
use self::dependencies::{unowned_axes, validate_materialized_reads};
use super::access::covers;
use super::dependencies::{common_scope, same_region};
use super::*;
use std::collections::{BTreeMap, BTreeSet};
mod access;
mod dependencies;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    Register,
    Global,
    /// Local accumulator accesses coexist with materialized accesses in the region.
    Materialized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialValue {
    Zero,
    Global,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Initialization {
    pub scope: ScopeId,
    pub access: AccessId,
    pub value: InitialValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStoragePlan {
    pub storage: AccessMode,
    pub representative: AccessId,
    pub initialization: Option<Initialization>,
    /// A register's final store executes at the end of this scope.
    pub export_scope: Option<ScopeId>,
    /// Writes must reach backing storage, either for a later region/output or
    /// to honor an explicitly selected Global allocation.
    pub publish: bool,
    /// Additive recurrences, separately from ordinary assignments/epilogues.
    pub accumulators: BTreeSet<StatementId>,
}

impl TensorStoragePlan {
    pub(crate) fn has_global(&self) -> bool {
        self.storage != AccessMode::Register
            || self.publish
            || self
                .initialization
                .as_ref()
                .is_some_and(|i| i.value == InitialValue::Global)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRead {
    pub definition: AccessId,
    /// Corresponding storage axes, excluding statically singleton view axes.
    pub axes: Vec<(usize, usize)>,
    /// Exact contiguous final-axis factorization, before taking a subview.
    /// Logical extents only; provider padding/layout policy is separate.
    pub split_last: Option<[usize; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelStoragePlan {
    pub tensors: BTreeMap<TensorId, TensorStoragePlan>,
    pub register_accesses: BTreeSet<AccessId>,
    pub local_reads: BTreeMap<AccessId, LocalRead>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoragePlan {
    /// Definitive program-wide backing storage, indexed by the input model's IDs.
    pub values: BTreeMap<TensorId, crate::Storage>,
    pub kernels: Vec<KernelStoragePlan>,
    pub globals: BTreeSet<TensorId>,
}

/// Choose storage for scheduled source without selecting a backend or new schedule.
pub fn infer(
    ir: &ScheduledIr,
    bindings: &Bindings,
    facts: &[KernelDataflow],
) -> Result<StoragePlan, ResolveError> {
    plan(ir, bindings, facts, None)
}

/// Infer source storage before tile parameters are bound. Shape and ordered-use
/// facts suffice; no backend defaults or sample tile sizes are introduced.
pub fn infer_source(
    ir: &ScheduledIr,
    bindings: &mut Bindings,
) -> Result<StoragePlan, ResolveError> {
    let mut metadata = TensorMetadata::collect(ir, bindings)?;
    metadata.resolve_shapes(ir, bindings)?;
    infer(ir, bindings, &super::facts::flow::analyze(ir))
}

/// Consume an existing PhysicalPlan's storage contract. Explicit Global values
/// retain global publication even if their computation can use a local tile.
pub fn for_values(
    ir: &ScheduledIr,
    bindings: &Bindings,
    facts: &[KernelDataflow],
    values: &BTreeMap<TensorId, crate::Storage>,
) -> Result<StoragePlan, ResolveError> {
    plan(ir, bindings, facts, Some(values))
}

fn plan(
    ir: &ScheduledIr,
    bindings: &Bindings,
    facts: &[KernelDataflow],
    contracts: Option<&BTreeMap<TensorId, crate::Storage>>,
) -> Result<StoragePlan, ResolveError> {
    let mut globals = BTreeSet::new();
    let kernels = facts
        .iter()
        .enumerate()
        .map(|(ki, facts)| plan_kernel(ki, ir, bindings, facts, contracts, &mut globals))
        .collect::<Result<Vec<_>, _>>()?;
    let mut values = BTreeMap::new();
    for (i, tensor) in ir.tensors().iter().enumerate() {
        let id = TensorId(i);
        let boundary = tensor.declarations.contains(&TensorKind::Input)
            || tensor.declarations.contains(&TensorKind::Output);
        let owners = kernels
            .iter()
            .filter(|k| k.tensors.contains_key(&id))
            .count();
        let required = if boundary {
            crate::Storage::External
        } else if globals.contains(&id) || owners > 1 {
            crate::Storage::Global
        } else {
            crate::Storage::Register
        };
        let chosen = if let Some(contracts) = contracts {
            let chosen = *contracts.get(&id).ok_or_else(|| {
                invalid(format!(
                    "{}: missing physical storage contract",
                    tensor.name
                ))
            })?;
            if boundary != (chosen == crate::Storage::External) {
                return Err(invalid(format!(
                    "{}: External storage must match an input/output binding",
                    tensor.name
                )));
            }
            if chosen == crate::Storage::Shared {
                return Err(invalid(
                    "explicit Shared transport requires a shared-memory layout contract",
                ));
            }
            if chosen == crate::Storage::Register && required != crate::Storage::Register {
                return Err(invalid(format!(
                    "{}: Register storage cannot satisfy materialization or cross-region uses",
                    tensor.name
                )));
            }
            chosen
        } else {
            required
        };
        values.insert(id, chosen);
    }
    Ok(StoragePlan {
        values,
        kernels,
        globals,
    })
}

fn plan_kernel(
    ki: usize,
    ir: &ScheduledIr,
    bindings: &Bindings,
    facts: &KernelDataflow,
    contracts: Option<&BTreeMap<TensorId, crate::Storage>>,
    globals: &mut BTreeSet<TensorId>,
) -> Result<KernelStoragePlan, ResolveError> {
    let kernel = &ir.kernels()[ki];
    let parallel: Vec<_> = ir
        .scopes()
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kernel.index() == ki && s.kind.is_parallel())
        .map(|(i, _)| ScopeId(i))
        .collect();
    let dependencies = &facts.loop_dependencies;
    let mut tensors = BTreeMap::new();
    for (&tensor, flow) in &facts.tensors {
        let uses = &flow.accesses;
        let writes = &flow.writes;
        let input = flow.input;
        let publish = flow.live_out
            || contracts.is_some_and(|c| c.get(&tensor) == Some(&crate::Storage::Global));
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
                        .any(|w| access::local_read(ir, *w, *r, bindings).is_some())
                        || (same_region(ir.access(*r), ir.access(representative))
                            && flow
                                .additive_updates
                                .contains(&ir.access(representative).statement))
                });
        let storage = if input || writes.is_empty() {
            AccessMode::Global
        } else if all_same || local_subviews {
            AccessMode::Register
        } else {
            AccessMode::Materialized
        };
        if storage != AccessMode::Register || publish {
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
        if storage == AccessMode::Materialized {
            validate_materialized_reads(
                ir,
                uses,
                representative,
                initialization.as_ref(),
                bindings,
            )?;
        }
        if !input {
            let global_reads: Vec<_> = if storage == AccessMode::Global {
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
                if !unowned_axes(ir, ir.access(*read), &parallel, bindings).is_empty() {
                    return Err(invalid(
                        "a mutated input is read across program ownership boundaries",
                    ));
                }
            }
        }
        let export_scope = if (storage == AccessMode::Register && publish)
            || (storage == AccessMode::Materialized && initialization.is_some())
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
        if publish || storage == AccessMode::Materialized {
            for write in writes {
                for axis in unowned_axes(ir, ir.access(*write), &parallel, bindings) {
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
            TensorStoragePlan {
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
        let direct_store = tp.storage == AccessMode::Register
            && tp.publish
            && tp.initialization.is_none()
            && !uses.iter().any(|a| ir.access(*a).kind == AccessKind::Read);
        if direct_store {
            tp.export_scope = None;
        }
        for access in uses {
            if (tp.storage == AccessMode::Register && !direct_store)
                || (tp.storage == AccessMode::Materialized
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
                        .find_map(|w| access::local_read(ir, *w, access, bindings))
                {
                    local_reads.insert(access, binding);
                }
            }
        }
    }
    Ok(KernelStoragePlan {
        tensors,
        register_accesses,
        local_reads,
    })
}
