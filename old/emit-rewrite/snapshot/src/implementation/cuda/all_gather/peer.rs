//! Direct peer-memory AllGather contracts, shared by standalone and fused plans.
//!
//! Execution must make symmetric source data visible before peer reads, preserve
//! it until all readers finish, and publish completion only after peer writes
//! are visible. Push writes each rank's own contribution to every destination;
//! pull reads each requested region from its owning peer. Runtime allocation,
//! readiness signals, fences and completion protocols belong to execution
//! concretization, not the physical Statement graph.
//!
//! A fused push accepts BF16 Shared or Register output tiles. A fused pull
//! stages the requested operand panels in Shared memory; its remote source
//! always remains External/Global. Register push also rounds FP32 accumulators
//! to BF16 before sending, preserving the unfused value's dtype boundary.

use super::GatherContext;
use crate::emit::cuda::{
    Accesses, CudaImplementation, CudaPhaseTemplate, EmitError, OperationSchedule, Region,
    render_template,
};
use crate::implementation::{
    AllGatherImplementation, AttributeSet, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};
use crate::plan::normalize::*;
use crate::{CommunicationKind, DType, Operation, OperationPayload, PhysicalPlan};
use crate::{LoopKind, OperationId};

const TRANSFER_TILE: usize = 64;
pub(in crate::implementation::cuda) const PEER_PUSH_ID: ImplementationId =
    ImplementationId::new("nvshmem.peer_push");
pub(in crate::implementation::cuda) const PEER_PULL_ID: ImplementationId =
    ImplementationId::new("nvshmem.peer_pull");

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
struct PeerGeometry {
    source: [usize; 2],
    target: [usize; 2],
    axis: usize,
    world_size: usize,
}

impl PeerGeometry {
    pub(in crate::implementation::cuda) fn new(
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

    pub(in crate::implementation::cuda) fn push_destination(
        &self,
        rank: usize,
        local: [usize; 2],
    ) -> Option<[usize; 2]> {
        if rank >= self.world_size || (0..2).any(|axis| local[axis] >= self.source[axis]) {
            return None;
        }
        let mut global = local;
        global[self.axis] = rank * self.source[self.axis] + local[self.axis];
        Some(global)
    }

    pub(in crate::implementation::cuda) fn pull_source(
        &self,
        global: [usize; 2],
    ) -> Option<(usize, [usize; 2])> {
        if (0..2).any(|axis| global[axis] >= self.target[axis]) {
            return None;
        }
        let rank = global[self.axis] / self.source[self.axis];
        let mut local = global;
        local[self.axis] %= self.source[self.axis];
        Some((rank, local))
    }
}

pub(in crate::implementation::cuda) fn supports_operation(
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
    fn context(&self, plan: &PhysicalPlan, id: OperationId) -> Result<GatherContext, EmitError> {
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
            return Err(EmitError::Contract(
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
impl CudaImplementation for PeerAllGather {
    fn schedule(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationSchedule, EmitError> {
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
    fn phases(&self, plan: &PhysicalPlan, id: OperationId) -> Result<CudaPhaseTemplate, EmitError> {
        let c = self.context(plan, id)?;
        let op = plan.operation(id).unwrap();
        let text = render_template(include_str!("peer.cu.j2"), &c)?;
        let mut body = CudaPhaseTemplate::cuda_with_parameters(
            &text,
            false,
            &["Element", "src", "dst", "row", "col", "peer", "i"],
            &["input_buffer", "output_buffer"],
        )?;
        for (name, value, output) in [
            ("input_buffer", op.inputs()[0], false),
            ("output_buffer", op.outputs()[0], true),
        ] {
            let tensor = plan.value_instance(value).unwrap();
            let binding = crate::emit::Binding {
                symbol: body.symbol(name)?,
                value,
                dtype: tensor.dtype(),
                storage: tensor.storage(),
            };
            if output {
                body.epilogue.outputs.push(binding);
            } else {
                body.epilogue.inputs.push(binding);
            }
        }
        body.epilogue.resources.symmetric_values = if c.push {
            vec![op.outputs()[0]]
        } else {
            vec![op.inputs()[0]]
        };
        Ok(body)
    }
    fn accesses(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
        rank: usize,
        coordinate: [usize; 3],
    ) -> Result<Option<Accesses>, EmitError> {
        let c = self.context(plan, id)?;
        let op = plan.operation(id).unwrap();
        let input = op.inputs()[0];
        let output = op.outputs()[0];
        let source = [c.source_rows, c.source_cols];
        let target = [c.target_rows, c.target_cols];
        let geometry =
            PeerGeometry::new(DType::Bf16, [&source, &target], c.axis, plan.world_size()).unwrap();
        let [row, col, _] = coordinate;
        let (reads, writes, coordinate) = if c.push {
            let origin = geometry
                .push_destination(rank, [row, col])
                .ok_or_else(|| unsupported("peer push coordinate"))?;
            (
                vec![Region::new(input, rank, [row, col], [64, 64])],
                (0..plan.world_size())
                    .map(|peer| Region::new(output, peer, origin, [64, 64]))
                    .collect(),
                [row, col, rank],
            )
        } else {
            let (peer, origin) = geometry
                .pull_source([row, col])
                .ok_or_else(|| unsupported("peer pull coordinate"))?;
            (
                vec![Region::new(input, peer, origin, [64, 64])],
                vec![Region::new(output, rank, [row, col], [64, 64])],
                [row, col, peer],
            )
        };
        Ok(Some(Accesses {
            coordinate,
            reads,
            writes,
            ..Default::default()
        }))
    }
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
