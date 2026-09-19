//! Peer AllGather enumeration and schedule normalization.

use crate::DType;
use crate::implementation::{
    AllGatherImplementation, AttributeSet, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};

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

/// Operand geometry supported by both peer definitions.
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
