//! Hopper rules for a linear body: optional pull, GEMMs, optional push.

use std::collections::BTreeSet;

use crate::{
    Action, FusionError, FusionRewrite, FusionRule, Operation, OperationId, PhysicalPlan, Storage,
    ValueInstanceId,
};

use super::hopper_wgmma::{FusionGemmShape, fusion_shape};
use super::peer::{PEER_PULL_ID, PEER_PUSH_ID, supports_operation};

pub(super) static RULES: &[&dyn FusionRule] = &[
    &HopperFusionRule(Kind::Producer),
    &HopperFusionRule(Kind::Consumer),
    &HopperFusionRule(Kind::Computation),
];

enum Kind {
    Producer,
    Consumer,
    Computation,
}
struct HopperFusionRule(Kind);

enum BodyOperation<'a> {
    Pull(&'a Operation),
    Gemm(&'a Operation, FusionGemmShape),
    Push(&'a Operation),
}

impl<'a> BodyOperation<'a> {
    fn new(plan: &PhysicalPlan, operation: &'a Operation) -> Option<Self> {
        if let Some(shape) = fusion_shape(plan, operation) {
            Some(Self::Gemm(operation, shape))
        } else if supports_operation(plan, operation, PEER_PULL_ID) {
            Some(Self::Pull(operation))
        } else if supports_operation(plan, operation, PEER_PUSH_ID) {
            Some(Self::Push(operation))
        } else {
            None
        }
    }

    fn operation(&self) -> &'a Operation {
        match self {
            Self::Pull(operation) | Self::Gemm(operation, _) | Self::Push(operation) => operation,
        }
    }
}

impl FusionRule for HopperFusionRule {
    fn apply(
        &self,
        plan: &PhysicalPlan,
        producer: &Action,
        consumer: &Action,
    ) -> Result<Vec<FusionRewrite>, FusionError> {
        let Some(left) = producer
            .operations()
            .last()
            .and_then(|id| plan.operation(*id))
            .and_then(|operation| BodyOperation::new(plan, operation))
        else {
            return Ok(Vec::new());
        };
        let Some(right) = consumer
            .operations()
            .first()
            .and_then(|id| plan.operation(*id))
            .and_then(|operation| BodyOperation::new(plan, operation))
        else {
            return Ok(Vec::new());
        };
        let matches = matches!(
            (&self.0, &left, &right),
            (
                Kind::Producer,
                BodyOperation::Gemm(..),
                BodyOperation::Push(_)
            ) | (
                Kind::Consumer,
                BodyOperation::Pull(_),
                BodyOperation::Gemm(..)
            ) | (
                Kind::Computation,
                BodyOperation::Gemm(..),
                BodyOperation::Gemm(..)
            )
        );
        if !matches {
            return Ok(Vec::new());
        }
        let bridge = left.operation().outputs()[0];
        if !right.operation().inputs().contains(&bridge)
            || plan.value_instance(bridge).unwrap().storage() != Storage::Global
        {
            return Ok(Vec::new());
        }
        let mut operations = producer
            .operations()
            .iter()
            .chain(consumer.operations())
            .copied()
            .collect::<Vec<_>>();
        operations.sort_unstable();
        if operations.windows(2).any(|pair| pair[0] == pair[1]) {
            return Ok(Vec::new());
        }

        let storages: &[Storage] = match self.0 {
            Kind::Consumer => &[Storage::Shared],
            Kind::Producer | Kind::Computation => &[Storage::Shared, Storage::Register],
        };
        Ok(storages
            .iter()
            .copied()
            .filter(|storage| supports_body(plan, &operations, bridge, *storage))
            .map(|storage| FusionRewrite::new(operations.iter().copied(), [(bridge, storage)]))
            .collect())
    }
}

fn supports_body(
    plan: &PhysicalPlan,
    ids: &[OperationId],
    promoted: ValueInstanceId,
    storage: Storage,
) -> bool {
    let Some(body) = ids
        .iter()
        .map(|id| BodyOperation::new(plan, plan.operation(*id)?))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    if !body.iter().any(|op| matches!(op, BodyOperation::Gemm(..))) {
        return false;
    }
    let members = ids.iter().copied().collect::<BTreeSet<_>>();
    for (index, current) in body.iter().enumerate() {
        let valid_endpoint = match current {
            BodyOperation::Pull(operation) => {
                index == 0
                    && matches!(
                        plan.value_instance(operation.inputs()[0])
                            .unwrap()
                            .storage(),
                        Storage::External | Storage::Global
                    )
            }
            BodyOperation::Push(operation) => {
                index + 1 == body.len()
                    && matches!(
                        plan.value_instance(operation.outputs()[0])
                            .unwrap()
                            .storage(),
                        Storage::External | Storage::Global
                    )
            }
            BodyOperation::Gemm(..) => true,
        };
        if !valid_endpoint {
            return false;
        }
        // Every internal input must come from the immediately preceding op.
        // This excludes forks, shortcuts, two-input internal reuse, and cycles.
        for input in current.operation().inputs() {
            if let Some((owner, _)) = plan
                .operations()
                .find(|(_, op)| op.outputs().contains(input))
                && members.contains(&owner)
                && (index == 0 || owner != ids[index - 1])
            {
                return false;
            }
        }
    }

    for (index, pair) in body.windows(2).enumerate() {
        let value_id = pair[0].operation().outputs()[0];
        let value = plan.value_instance(value_id).unwrap();
        let value_storage = if value_id == promoted {
            storage
        } else {
            value.storage()
        };
        if plan.output().value() == value_id
            || plan
                .inputs()
                .iter()
                .any(|binding| binding.value() == value_id)
            || plan
                .operations()
                .any(|(id, op)| op.inputs().contains(&value_id) && id != ids[index + 1])
        {
            return false;
        }
        match (&pair[0], &pair[1]) {
            (BodyOperation::Pull(_), BodyOperation::Gemm(gemm, _)) => {
                if value_storage != Storage::Shared
                    || gemm.inputs().iter().filter(|id| **id == value_id).count() != 1
                {
                    return false;
                }
            }
            (BodyOperation::Gemm(_, lhs), BodyOperation::Gemm(gemm, rhs)) => {
                if gemm.inputs()[0] != value_id
                    || gemm.inputs()[1] == value_id
                    || lhs.m != rhs.m
                    || lhs.n != 128
                    || rhs.k != 128
                    || rhs.n != 128
                    || !matches!(value_storage, Storage::Shared | Storage::Register)
                {
                    return false;
                }
            }
            (BodyOperation::Gemm(..), BodyOperation::Push(push)) => {
                if push.inputs()[0] != value_id
                    || !matches!(value_storage, Storage::Shared | Storage::Register)
                {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}
