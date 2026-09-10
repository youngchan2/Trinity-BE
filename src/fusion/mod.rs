//! Ordered task-set fusion and backend binding/storage proposals.

use thiserror::Error;

use crate::{
    OperationId, PhysicalInvariantError, PhysicalPlan, Statement, Storage, ValueInstanceId,
};

use std::collections::BTreeSet;

/// Proposes operation and storage rewrites for producer and consumer statements.
/// Proposals preserve the original operation order and sequential scopes.
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
        producer: &Statement,
        consumer: &Statement,
    ) -> Result<Vec<FusionRewrite>, FusionError>;
}

/// A fusion proposal listing merged operations and storage promotions.
///
/// Storage updates must be unique Global-to-Shared/Register promotions. ABI
/// values and values used outside the merged operations cannot be promoted.
/// The common rewriter constructs the replacement Statement tree and CUDA body.
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
    #[error("downstream fusion is not connected to the structured Statement model")]
    UnsupportedStatementModel,
    #[error("invalid fusion rewrite: {reason}")]
    InvalidRewrite { reason: &'static str },

    #[error(transparent)]
    InvalidPlan(#[from] PhysicalInvariantError),
}

/// Enumerate the original plan and legal adjacent fusions to a fixed point.
/// Every candidate is rebuilt transactionally and checked by common CUDA lowering.
pub fn fuse(
    plan: &PhysicalPlan,
    rules: &[&dyn FusionRule],
) -> Result<Vec<PhysicalPlan>, FusionError> {
    let mut results = vec![plan.clone()];
    let mut cursor = 0;
    while cursor < results.len() {
        let current = results[cursor].clone();
        for left in 0..current.statements().len().saturating_sub(1) {
            let producer = &current.statements()[left];
            let consumer = &current.statements()[left + 1];
            for rule in rules {
                for proposal in rule.apply(&current, producer, consumer)? {
                    validate(&current, producer, consumer, &proposal)?;
                    let Some(candidate) =
                        crate::physical::rewrite_statements(&current, left, &proposal)?
                    else {
                        continue;
                    };
                    // This checks task-local lifetimes, layout adapters, accesses,
                    // stages, resource limits and the reconstructed dependency DAG.
                    if crate::emit(&candidate).is_err() {
                        continue;
                    }
                    if !results.iter().any(|p| p.same_body(&candidate)) {
                        results.push(candidate);
                    }
                }
            }
        }
        cursor += 1;
    }
    Ok(results)
}

fn validate(
    plan: &PhysicalPlan,
    producer: &Statement,
    consumer: &Statement,
    proposal: &FusionRewrite,
) -> Result<(), FusionError> {
    let invalid = |reason| FusionError::InvalidRewrite { reason };
    let expected: Vec<_> = producer
        .operations()
        .into_iter()
        .chain(consumer.operations())
        .collect();
    if proposal.operations != expected || proposal.storage_updates.is_empty() {
        return Err(invalid(
            "rewrite must preserve the adjacent regions and promote a binding",
        ));
    }
    let members: BTreeSet<_> = expected.iter().copied().collect();
    let mut updated = BTreeSet::new();
    for &(id, storage) in &proposal.storage_updates {
        let Some(value) = plan.value_instance(id) else {
            return Err(invalid("unknown promoted value"));
        };
        if !updated.insert(id)
            || value.storage() != Storage::Global
            || !matches!(storage, Storage::Shared | Storage::Register)
            || plan.output().value() == id
            || plan.inputs().iter().any(|b| b.value() == id)
            || plan.operations().any(|(op, p)| {
                !members.contains(&op) && (p.inputs().contains(&id) || p.outputs().contains(&id))
            })
            || plan
                .operations()
                .filter(|(_, p)| p.outputs().contains(&id))
                .count()
                != 1
        {
            return Err(invalid(
                "promotion needs a unique internal producer and no external uses",
            ));
        }
        if !producer
            .operations()
            .iter()
            .any(|op| plan.operation(*op).unwrap().outputs().contains(&id))
            || !consumer
                .operations()
                .iter()
                .any(|op| plan.operation(*op).unwrap().inputs().contains(&id))
        {
            return Err(invalid("promoted binding must connect the two regions"));
        }
    }
    Ok(())
}

/// Equal-coordinate pointwise expressions can reuse a producer's tile traversal.
pub(crate) fn pointwise(plan: &PhysicalPlan, id: OperationId) -> bool {
    let Some(op) = plan.operation(id) else {
        return false;
    };
    let crate::OperationPayload::Compute(compute) = op.payload() else {
        return false;
    };
    let selected = compute.implementation().id();
    if selected.as_str() != "cuda.simt.expression"
        && !crate::pointwise_implementations(plan.target())
            .iter()
            .any(|d| d.id() == selected)
    {
        return false;
    }
    let Some(store) = op.expression().and_then(crate::Expression::list) else {
        return false;
    };
    if store.len() != 4 || op.outputs().len() != 1 {
        return false;
    }
    let shape = plan.value_instance(op.outputs()[0]).unwrap().shape();
    if op
        .inputs()
        .iter()
        .any(|v| plan.value_instance(*v).unwrap().shape() != shape)
    {
        return false;
    }
    fn check(e: &crate::Expression, index: &crate::Expression) -> bool {
        if e.atom().is_some() {
            return true;
        }
        let Some(xs) = e.list() else { return false };
        match e.operator() {
            Some("load") => xs.len() == 3 && &xs[2] == index,
            Some("float_bits") => true,
            Some("+" | "-" | "*" | "/" | "relu" | "sqrt" | "sigmoid" | "sqr") => {
                xs[1..].iter().all(|x| check(x, index))
            }
            _ => false,
        }
    }
    !op.inputs().contains(&op.outputs()[0]) && check(&store[2], &store[3])
}
