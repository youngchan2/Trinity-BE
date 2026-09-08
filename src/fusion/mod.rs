//! Action rewrites and exhaustive physical fusion candidate enumeration.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::{Action, OperationId, PhysicalInvariantError, PhysicalPlan, Storage, ValueInstanceId};

mod graph;

/// Proposes rewrites that merge a producer and consumer Action into one Action.
pub trait FusionRule: Sync {
    /// Returns separate rewrites for storage alternatives (e.g. Shared/Register).
    /// An empty `Ok` means unsupported; `Err` aborts [`fuse`].
    ///
    /// Check backend compatibility of all merged operations and their storage,
    /// including earlier fusions. Rewrite IDs refer to `plan` and may change
    /// when the resulting plan is finalized.
    fn apply(
        &self,
        plan: &PhysicalPlan,
        producer: &Action,
        consumer: &Action,
    ) -> Result<Vec<FusionRewrite>, FusionError>;
}

/// Replaces two Actions with their operation union and promotes internal values.
///
/// Storage updates must be unique Global-to-Shared/Register promotions. ABI
/// values and values used outside the combined Action cannot be promoted.
/// Action inputs and outputs are derived by physical finalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionRewrite {
    operations: Vec<OperationId>,
    storage_updates: Vec<(ValueInstanceId, Storage)>,
}

impl FusionRewrite {
    pub fn new(
        operations: impl IntoIterator<Item = OperationId>,
        storage_updates: impl IntoIterator<Item = (ValueInstanceId, Storage)>,
    ) -> Self {
        Self {
            operations: operations.into_iter().collect(),
            storage_updates: storage_updates.into_iter().collect(),
        }
    }

    pub fn operations(&self) -> &[OperationId] {
        &self.operations
    }

    pub fn storage_updates(&self) -> &[(ValueInstanceId, Storage)] {
        &self.storage_updates
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FusionError {
    #[error("invalid fusion rewrite: {reason}")]
    InvalidRewrite { reason: &'static str },

    #[error(transparent)]
    InvalidPlan(#[from] PhysicalInvariantError),
}

impl FusionError {
    fn invalid(reason: &'static str) -> Self {
        Self::InvalidRewrite { reason }
    }
}

/// Enumerates the original plan and all unique plans reachable by these rules.
///
/// Traversal is breadth-first, with canonical Action pairs and caller-provided
/// rule order. Every rewrite removes exactly one Action; no cost pruning or
/// candidate limit is applied. The input plan is never mutated.
pub fn fuse(
    plan: &PhysicalPlan,
    rules: &[&dyn FusionRule],
) -> Result<Vec<PhysicalPlan>, FusionError> {
    let mut plans = vec![plan.clone()];
    let mut seen = BTreeMap::from([(plan.hash(), vec![0usize])]);

    let mut cursor = 0;

    while cursor < plans.len() {
        let source = &plans[cursor];
        let graph = graph::ActionGraph::new(source);
        let mut successors = Vec::new();

        for (producer_id, consumer_id) in graph.mergeable_pairs() {
            let producer = source.action(producer_id).unwrap();
            let consumer = source.action(consumer_id).unwrap();

            for rule in rules {
                for rewrite in rule.apply(source, producer, consumer)? {
                    validate_rewrite(source, producer, consumer, &rewrite)?;
                    successors.push(crate::physical::rewrite_actions(
                        source,
                        producer_id,
                        consumer_id,
                        rewrite.operations(),
                        rewrite.storage_updates(),
                    )?);
                }
            }
        }

        for candidate in successors {
            let bucket = seen.entry(candidate.hash()).or_default();
            if !bucket
                .iter()
                .any(|index| candidate.same_body(&plans[*index]))
            {
                bucket.push(plans.len());
                plans.push(candidate);
            }
        }
        cursor += 1;
    }

    Ok(plans)
}

fn validate_rewrite(
    plan: &PhysicalPlan,
    producer: &Action,
    consumer: &Action,
    rewrite: &FusionRewrite,
) -> Result<(), FusionError> {
    let expected = producer
        .operations()
        .iter()
        .chain(consumer.operations())
        .copied()
        .collect::<BTreeSet<_>>();
    let actual = rewrite.operations.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected || actual.len() != rewrite.operations.len() {
        return Err(FusionError::invalid(
            "operations must be exactly the union of the two Actions, without duplicates",
        ));
    }

    let mut updated = BTreeSet::new();
    for (value_id, storage) in &rewrite.storage_updates {
        if !updated.insert(*value_id) {
            return Err(FusionError::invalid("a value has multiple storage updates"));
        }
        let Some(value) = plan.value_instance(*value_id) else {
            return Err(FusionError::invalid(
                "storage update refers to an unknown value",
            ));
        };
        if value.storage() != Storage::Global
            || !matches!(storage, Storage::Shared | Storage::Register)
        {
            return Err(FusionError::invalid(
                "storage updates must promote Global values to Shared or Register",
            ));
        }
        if plan.output().value() == *value_id
            || plan.inputs().iter().any(|input| input.value() == *value_id)
        {
            return Err(FusionError::invalid("ABI values cannot be promoted"));
        }
        let internal_producer = expected.iter().any(|operation| {
            plan.operation(*operation)
                .unwrap()
                .outputs()
                .contains(value_id)
        });
        let external_consumer = plan.operations().any(|(id, operation)| {
            operation.inputs().contains(value_id) && !expected.contains(&id)
        });
        if !internal_producer || external_consumer {
            return Err(FusionError::invalid(
                "promoted values must be produced and used exclusively inside the fused Action",
            ));
        }
    }
    Ok(())
}
