//! Direct peer-memory AllGather contracts, shared by standalone and fused plans.
//!
//! Execution must make symmetric source data visible before peer reads, preserve
//! it until all readers finish, and publish completion only after peer writes
//! are visible. Push writes each rank's own contribution to every destination;
//! pull reads each requested region from its owning peer. Runtime allocation,
//! readiness signals, fences and completion protocols belong to execution
//! concretization, not the physical Action graph.
//!
//! A fused push accepts BF16 Shared or Register output tiles. A fused pull
//! stages the requested operand panels in Shared memory; its remote source
//! always remains External/Global. Register push also rounds FP32 accumulators
//! to BF16 before sending, preserving the unfused value's dtype boundary.

use super::super::{
    AllGatherImplementation, AttributeSet, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};
use crate::{CommunicationKind, DType, Operation, OperationPayload, PhysicalPlan};

pub(super) const TRANSFER_TILE: usize = 64;
pub(super) const PEER_PUSH_ID: ImplementationId = ImplementationId::new("nvshmem.peer_push");
pub(super) const PEER_PULL_ID: ImplementationId = ImplementationId::new("nvshmem.peer_pull");

pub(super) static PEER_PUSH: PeerAllGather = PeerAllGather(PEER_PUSH_ID);
pub(super) static PEER_PULL: PeerAllGather = PeerAllGather(PEER_PULL_ID);

pub(super) struct PeerAllGather(ImplementationId);

impl ImplementationDefinition for PeerAllGather {
    fn cuda(&self) -> Option<&dyn crate::emit::cuda::CudaImplementation> {
        Some(self)
    }

    fn id(&self) -> ImplementationId {
        self.0
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
pub(super) struct PeerGeometry {
    source: [usize; 2],
    target: [usize; 2],
    axis: usize,
    world_size: usize,
}

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
        Some(Self {
            source,
            target,
            axis,
            world_size,
        })
    }

    pub(super) fn push_destination(&self, rank: usize, local: [usize; 2]) -> Option<[usize; 2]> {
        if rank >= self.world_size || (0..2).any(|axis| local[axis] >= self.source[axis]) {
            return None;
        }
        let mut global = local;
        global[self.axis] = rank * self.source[self.axis] + local[self.axis];
        Some(global)
    }

    pub(super) fn pull_source(&self, global: [usize; 2]) -> Option<(usize, [usize; 2])> {
        if (0..2).any(|axis| global[axis] >= self.target[axis]) {
            return None;
        }
        let rank = global[self.axis] / self.source[self.axis];
        let mut local = global;
        local[self.axis] %= self.source[self.axis];
        Some((rank, local))
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn push_and_pull_cover_each_replicated_element_once_on_both_axes() {
        for axis in 0..2 {
            let source = [128, 192];
            let mut target = source;
            target[axis] *= 3;
            let geometry = PeerGeometry::new(DType::Bf16, [&source, &target], axis, 3).unwrap();
            let mut written = BTreeSet::new();
            for rank in 0..3 {
                for row in 0..source[0] {
                    for column in 0..source[1] {
                        let local = [row, column];
                        let global = geometry.push_destination(rank, local).unwrap();
                        assert!(written.insert(global));
                        // Independent concatenation oracle, not just a roundtrip.
                        let expected = if axis == 0 {
                            [rank * 128 + row, column]
                        } else {
                            [row, rank * 192 + column]
                        };
                        assert_eq!(global, expected);
                        assert_eq!(geometry.pull_source(expected), Some((rank, local)));
                    }
                }
            }
            assert_eq!(written.len(), target[0] * target[1]);
            assert_eq!(geometry.push_destination(3, [0, 0]), None);
            assert_eq!(geometry.push_destination(0, source), None);
            assert_eq!(geometry.pull_source(target), None);
        }
    }

    #[test]
    fn peer_definitions_reject_invalid_geometry_and_preserve_attributes() {
        for definition in [&PEER_PUSH, &PEER_PULL] {
            let instances = definition.enumerate(DType::Bf16, [&[64, 128], &[64, 256]], 1, 2);
            assert_eq!(instances.len(), 1);
            assert_eq!(instances[0].id(), definition.id());
            assert_eq!(
                instances[0].attributes().iter().collect::<Vec<_>>(),
                [("shard_axis", 1), ("tile_cols", 64), ("tile_rows", 64),]
            );
            for (dtype, source, target, axis, world) in [
                (DType::Fp32, [64, 128], [64, 256], 1, 2),
                (DType::Bf16, [64, 128], [64, 256], 2, 2),
                (DType::Bf16, [64, 128], [64, 128], 1, 1),
                (DType::Bf16, [64, 128], [64, 256], 1, 0),
                (DType::Bf16, [64, 128], [128, 256], 1, 2),
                (DType::Bf16, [64, 128], [64, 384], 1, 2),
                (DType::Bf16, [64, 128], [64, 256], 1, 3),
                (DType::Bf16, [96, 128], [96, 256], 1, 2),
                (DType::Bf16, [0, 128], [0, 256], 1, 2),
                (DType::Bf16, [64, 128], [64, 256], 1, usize::MAX),
            ] {
                assert!(
                    definition
                        .enumerate(dtype, [&source, &target], axis, world)
                        .is_empty()
                );
            }
            assert!(
                definition
                    .enumerate(DType::Bf16, [&[64], &[128]], 0, 2)
                    .is_empty()
            );
        }
    }
}
