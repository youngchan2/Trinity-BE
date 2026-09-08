//! CUDA specialization stays with the concrete backend definitions.
use serde::Serialize;

use super::hopper_wgmma::{HopperWgmmaBf16, fusion_shape};
use super::nvls::NvlsOneShotPushNbi;
use super::peer::{PEER_PUSH_ID, PeerAllGather, PeerGeometry, TRANSFER_TILE, supports_operation};
use crate::emit::cuda::{
    CudaImplementation, EmitError, OperationEmission, Region, Work, full_region, render_template,
};
use crate::{DType, ImplementationDefinition, OperationId, OperationPayload, PhysicalPlan};

#[derive(Serialize)]
struct GemmContext {
    operation: usize,
    lhs: usize,
    rhs: usize,
    output: usize,
    m: usize,
    n: usize,
    k: usize,
}

impl CudaImplementation for HopperWgmmaBf16 {
    fn specialize(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationEmission, EmitError> {
        let op = plan.operation(id).unwrap();
        let shape = fusion_shape(plan, op)
            .ok_or_else(|| EmitError::Unsupported("WGMMA geometry or attributes".into()))?;

        let [lhs, rhs] = *<&[_; 2]>::try_from(op.inputs()).unwrap();
        let output = op.outputs()[0];
        let compute_produced = |value| {
            plan.operations().any(|(_, p)| {
                p.outputs().contains(&value) && matches!(p.payload(), OperationPayload::Compute(_))
            })
        };

        let lhs_compute = compute_produced(lhs);
        let rhs_compute = compute_produced(rhs);

        let work = (0..plan.world_size())
            .map(|rank| {
                let mut work = Vec::new();
                for row in (0..shape.m).step_by(128) {
                    for col in (0..shape.n).step_by(128) {
                        let mut reads = Vec::new();
                        if lhs_compute {
                            reads.push(full_region(plan, lhs, rank));
                        }
                        if rhs_compute {
                            reads.push(full_region(plan, rhs, rank));
                        }
                        let stages = (0..shape.k)
                            .step_by(64)
                            .map(|k| {
                                let mut reads = Vec::new();
                                if !lhs_compute {
                                    reads.push(Region::new(lhs, rank, [row, k], [128, 64]));
                                }
                                if !rhs_compute {
                                    reads.push(Region::new(rhs, rank, [k, col], [64, 128]));
                                }
                                reads
                            })
                            .collect();
                        work.push(Work {
                            coordinate: [row / 128, col / 128, 0],
                            reads,
                            stages,
                            writes: vec![Region::new(output, rank, [row, col], [128, 128])],
                            ..Work::default()
                        });
                    }
                }
                work
            })
            .collect();

        Ok(OperationEmission {
            body: render_template(
                include_str!("templates/wgmma.cu.j2"),
                &GemmContext {
                    operation: id.index(),
                    lhs: lhs.index(),
                    rhs: rhs.index(),
                    output: output.index(),
                    m: shape.m,
                    n: shape.n,
                    k: shape.k,
                },
            )?,
            work,
            shared_memory_bytes: 2 * (128 * 64 + 128 * 64) * 2,
            symmetric_values: vec![],
            nvls: false,
        })
    }
}

#[derive(Serialize)]
struct GatherContext {
    operation: usize,
    input: usize,
    output: usize,
    source_rows: usize,
    source_cols: usize,
    target_rows: usize,
    target_cols: usize,
    axis: usize,
    push: bool,
    chunk: usize,
}

impl CudaImplementation for PeerAllGather {
    fn specialize(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationEmission, EmitError> {
        let op = plan.operation(id).unwrap();
        if !supports_operation(plan, op, self.id()) {
            return Err(EmitError::Unsupported("peer AllGather contract".into()));
        }
        let OperationPayload::Communication(comm) = op.payload() else {
            unreachable!()
        };
        let axis = comm
            .implementation()
            .attributes()
            .get("shard_axis")
            .unwrap();
        let input = op.inputs()[0];
        let output = op.outputs()[0];
        let source = plan.value_instance(input).unwrap().shape();
        let target = plan.value_instance(output).unwrap().shape();
        let geometry =
            PeerGeometry::new(DType::Bf16, [source, target], axis, plan.world_size()).unwrap();
        let push = self.id() == PEER_PUSH_ID;
        let shape = if push { source } else { target };
        let work = (0..plan.world_size())
            .map(|rank| {
                let mut work = Vec::new();
                for row in (0..shape[0]).step_by(TRANSFER_TILE) {
                    for col in (0..shape[1]).step_by(TRANSFER_TILE) {
                        let (reads, writes, coordinate) = if push {
                            let origin = geometry.push_destination(rank, [row, col]).unwrap();
                            (
                                vec![Region::new(input, rank, [row, col], [64, 64])],
                                (0..plan.world_size())
                                    .map(|peer| Region::new(output, peer, origin, [64, 64]))
                                    .collect(),
                                [row, col, rank],
                            )
                        } else {
                            let (peer, origin) = geometry.pull_source([row, col]).unwrap();
                            (
                                vec![Region::new(input, peer, origin, [64, 64])],
                                vec![Region::new(output, rank, [row, col], [64, 64])],
                                [row, col, peer],
                            )
                        };
                        work.push(Work {
                            coordinate,
                            reads,
                            writes,
                            ..Work::default()
                        });
                    }
                }
                work
            })
            .collect();
        Ok(OperationEmission {
            body: render_template(
                include_str!("templates/peer.cu.j2"),
                &GatherContext {
                    operation: id.index(),
                    input: input.index(),
                    output: output.index(),
                    source_rows: source[0],
                    source_cols: source[1],
                    target_rows: target[0],
                    target_cols: target[1],
                    axis,
                    push,
                    chunk: 64,
                },
            )?,
            work,
            shared_memory_bytes: 0,
            symmetric_values: if push { vec![output] } else { vec![input] },
            nvls: false,
        })
    }
}

impl CudaImplementation for NvlsOneShotPushNbi {
    fn specialize(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationEmission, EmitError> {
        let op = plan.operation(id).unwrap();
        let OperationPayload::Communication(comm) = op.payload() else {
            return Err(EmitError::Unsupported("NVLS payload".into()));
        };
        let ([input], [output]) = (op.inputs(), op.outputs()) else {
            return Err(EmitError::Unsupported("NVLS arity".into()));
        };
        let source = plan.value_instance(*input).unwrap().shape();
        let target = plan.value_instance(*output).unwrap().shape();
        let axis = (0..2)
            .find(|&axis| source[axis] != target[axis])
            .ok_or_else(|| EmitError::Unsupported("NVLS shard axis".into()))?;
        let chunk = comm
            .implementation()
            .attributes()
            .get("chunk_extent")
            .unwrap_or(0);
        // NVSHMEM's BF16 tile path transfers at least a packed pair per lane.
        if !source[1].is_multiple_of(2) {
            return Err(EmitError::Unsupported(
                "NVLS requires an even row-major column extent".into(),
            ));
        }
        if plan.world_size() <= 1
            || chunk != 128
            || !source[axis].is_multiple_of(chunk)
            || source[1 - axis] != target[1 - axis]
            || source[axis].checked_mul(plan.world_size()) != Some(target[axis])
        {
            return Err(EmitError::Unsupported("NVLS geometry or attributes".into()));
        }
        let work = (0..plan.world_size())
            .map(|rank| {
                let mut work = Vec::new();
                for start in (0..source[axis]).step_by(chunk) {
                    let mut origin = [0, 0];
                    origin[axis] = start;
                    let mut extent = [source[0], source[1]];
                    extent[axis] = chunk;
                    // Collective entry requires source readiness on every rank. No
                    // participant can wait for a producer while holding a collective.
                    let reads = (0..plan.world_size())
                        .map(|peer| Region::new(*input, peer, origin, extent))
                        .collect();
                    let writes = (0..plan.world_size())
                        .map(|peer| {
                            let mut destination = origin;
                            destination[axis] += peer * source[axis];
                            Region::new(*output, rank, destination, extent)
                        })
                        .collect();
                    work.push(Work {
                        coordinate: [start, 0, 0],
                        reads,
                        writes,
                        ordered_collective: true,
                        ..Work::default()
                    });
                }
                work
            })
            .collect();
        Ok(OperationEmission {
            body: render_template(
                include_str!("templates/nvls.cu.j2"),
                &GatherContext {
                    operation: id.index(),
                    input: input.index(),
                    output: output.index(),
                    source_rows: source[0],
                    source_cols: source[1],
                    target_rows: target[0],
                    target_cols: target[1],
                    axis,
                    push: true,
                    chunk,
                },
            )?,
            work,
            shared_memory_bytes: 0,
            symmetric_values: vec![*input, *output],
            nvls: true,
        })
    }
}
