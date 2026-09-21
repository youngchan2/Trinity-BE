//! Peer AllGather enumeration and schedule normalization.

use super::GatherContext;
use crate::implementation::{
    AllGatherImplementation, AttributeSet, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};
use crate::implementation::{OperationSchedule, ScheduleError};
use crate::plan::normalize::*;
use crate::{CommunicationKind, DType, Operation, OperationPayload, PhysicalPlan};
use crate::{LoopKind, OperationId};

const TRANSFER_TILE: usize = 64;
pub(super) const PEER_PUSH_ID: ImplementationId = ImplementationId::new("nvshmem.peer_push");
pub(super) const PEER_PULL_ID: ImplementationId = ImplementationId::new("nvshmem.peer_pull");

pub(super) static PEER_PUSH: PeerAllGather = PeerAllGather(PEER_PUSH_ID);
pub(super) static PEER_PULL: PeerAllGather = PeerAllGather(PEER_PULL_ID);

pub(super) struct PeerAllGather(ImplementationId);

impl ImplementationDefinition for PeerAllGather {
    fn id(&self) -> ImplementationId {
        self.0
    }

    fn schedule(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationSchedule, ScheduleError> {
        let c = self.context(plan, id)?;
        let shape = if c.push {
            [c.source_rows, c.source_cols]
        } else {
            [c.target_rows, c.target_cols]
        };
        Ok(OperationSchedule {
            dimensions: vec![
                dimension("row", shape[0], c.chunk, LoopKind::Parallel),
                dimension("col", shape[1], c.chunk, LoopKind::Parallel),
            ],
            coordinates: vec![variable("row"), variable("col")],
            ..Default::default()
        })
    }
}

impl AllGatherImplementation for PeerAllGather {
    fn enumerate(
        &'static self,
        dtype: DType,
        shapes: [&[usize]; 2],
        shard_axis: usize,
        world_size: usize,
    ) -> Vec<ImplementationInstance> {
        if PeerGeometry::new(dtype, shapes, shard_axis, world_size).is_none() {
            return Vec::new();
        }
        vec![ImplementationInstance::new(
            self,
            AttributeSet::new([
                ("shard_axis", shard_axis),
                ("tile_rows", TRANSFER_TILE),
                ("tile_cols", TRANSFER_TILE),
            ]),
        )]
    }
}

/// The same coordinate contract is used to validate both peer definitions.
struct PeerGeometry;

impl PeerGeometry {
    pub(super) fn new(
        dtype: DType,
        shapes: [&[usize]; 2],
        axis: usize,
        world_size: usize,
    ) -> Option<Self> {
        let source: [usize; 2] = shapes[0].try_into().ok()?;
        let target: [usize; 2] = shapes[1].try_into().ok()?;
        if dtype != DType::Bf16
            || axis >= 2
            || world_size <= 1
            || source
                .iter()
                .chain(&target)
                .any(|extent| *extent == 0 || !extent.is_multiple_of(TRANSFER_TILE))
        {
            return None;
        }
        // Exact shard coverage bounds every rank-local coordinate by target.
        if !target[axis].is_multiple_of(world_size)
            || target[axis] / world_size != source[axis]
            || target[1 - axis] != source[1 - axis]
        {
            return None;
        }
        Some(Self)
    }
}

pub(super) fn supports_operation(
    plan: &PhysicalPlan,
    operation: &Operation,
    id: ImplementationId,
) -> bool {
    let OperationPayload::Communication(communication) = operation.payload() else {
        return false;
    };
    let instance = communication.implementation();
    if communication.kind() != CommunicationKind::AllGather || instance.id() != id {
        return false;
    }
    let ([input], [output]) = (operation.inputs(), operation.outputs()) else {
        return false;
    };
    let (Some(input), Some(output)) = (plan.value_instance(*input), plan.value_instance(*output))
    else {
        return false;
    };
    let attributes = instance.attributes();
    let Some(axis) = attributes.get("shard_axis") else {
        return false;
    };
    input.dtype() == output.dtype()
        && attributes.get("tile_rows") == Some(TRANSFER_TILE)
        && attributes.get("tile_cols") == Some(TRANSFER_TILE)
        && PeerGeometry::new(
            input.dtype(),
            [input.shape(), output.shape()],
            axis,
            plan.world_size(),
        )
        .is_some()
}

impl PeerAllGather {
    fn context(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<GatherContext, ScheduleError> {
        let op = plan.operation(id).unwrap();
        if !supports_operation(plan, op, self.id()) {
            return Err(unsupported("peer AllGather contract"));
        }
        let OperationPayload::Communication(c) = op.payload() else {
            unreachable!()
        };
        let input = op.inputs()[0];
        let output = op.outputs()[0];
        if [input, output].iter().any(|&value| {
            matches!(
                plan.value_instance(value).unwrap().storage(),
                crate::Storage::Shared | crate::Storage::Register
            )
        }) {
            return Err(ScheduleError::Contract(
                "local value has no launch binding".into(),
            ));
        }
        let source = plan.value_instance(input).unwrap().shape();
        let target = plan.value_instance(output).unwrap().shape();
        Ok(GatherContext {
            source_rows: source[0],
            source_cols: source[1],
            target_rows: target[0],
            target_cols: target[1],
            axis: c.implementation().attributes().get("shard_axis").unwrap(),
            push: self.id() == PEER_PUSH_ID,
            chunk: TRANSFER_TILE,
        })
    }
}
