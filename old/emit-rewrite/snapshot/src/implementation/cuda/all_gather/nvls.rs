use super::GatherContext;
use crate::DType;
use crate::emit::cuda::{
    Accesses, CudaImplementation, CudaPhaseTemplate, EmitError, OperationSchedule, Region,
    render_template,
};
use crate::implementation::{
    AllGatherImplementation, AttributeSet, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};
use crate::plan::normalize::*;
use crate::{LoopKind, OperationId};
use crate::{OperationPayload, PhysicalPlan};

pub(super) static NVLS_ONE_SHOT_PUSH_NBI: NvlsOneShotPushNbi = NvlsOneShotPushNbi;

pub(super) struct NvlsOneShotPushNbi;

impl AllGatherImplementation for NvlsOneShotPushNbi {
    fn enumerate(
        &'static self,
        dtype: DType,
        shapes: [&[usize]; 2],
        shard_axis: usize,
        world_size: usize,
    ) -> Vec<ImplementationInstance> {
        let [source_shape, target_shape] = shapes;
        if dtype != DType::Bf16
            || world_size <= 1
            || source_shape.len() != 2
            || target_shape.len() != 2
            || shard_axis >= source_shape.len()
        {
            return Vec::new();
        }

        let mut expected = source_shape.to_vec();
        let Some(global_extent) = expected[shard_axis].checked_mul(world_size) else {
            return Vec::new();
        };
        expected[shard_axis] = global_extent;
        if expected != target_shape {
            return Vec::new();
        }

        self.chunk_candidates()
            .filter(|chunk| source_shape[shard_axis].is_multiple_of(*chunk))
            .filter(|&chunk| supports_tile(source_shape, target_shape, shard_axis, chunk))
            .map(|chunk| {
                ImplementationInstance::new(self, AttributeSet::new([("chunk_extent", chunk)]))
            })
            .collect()
    }
}

fn supports_tile(source: &[usize], target: &[usize], axis: usize, chunk: usize) -> bool {
    let rows = if axis == 0 { chunk } else { source[0] };
    let cols = if axis == 1 { chunk } else { source[1] };
    // NVSHMEM 3.7.2's tile_allgather kernels take int dimensions/strides and
    // count packed BF16 pairs in an int. This bounds a tile, not the full tensor.
    [rows, cols, source[1], target[1]]
        .into_iter()
        .all(|n| n <= i32::MAX as usize)
        && rows * (cols / 2) <= i32::MAX as usize
}

impl NvlsOneShotPushNbi {
    /// Returns the concrete chunk extents implemented by Trinity's NVLS path.
    fn chunk_candidates(&self) -> impl ExactSizeIterator<Item = usize> {
        // TODO: Source chunk candidates from the autotuner.
        [128].into_iter()
    }
}

impl ImplementationDefinition for NvlsOneShotPushNbi {
    fn cuda(&self) -> Option<&dyn crate::emit::cuda::CudaImplementation> {
        Some(self)
    }

    fn id(&self) -> ImplementationId {
        ImplementationId::new("nvls.one_shot_push_nbi")
    }
}
impl NvlsOneShotPushNbi {
    fn context(&self, plan: &PhysicalPlan, id: OperationId) -> Result<GatherContext, EmitError> {
        let op = plan.operation(id).unwrap();
        let OperationPayload::Communication(c) = op.payload() else {
            return Err(unsupported("NVLS payload"));
        };
        let ([input], [output]) = (op.inputs(), op.outputs()) else {
            return Err(unsupported("NVLS arity"));
        };
        let source = plan.value_instance(*input).unwrap();
        let target = plan.value_instance(*output).unwrap();
        let ([sr, sc], [tr, tc]) = (source.shape(), target.shape()) else {
            return Err(unsupported("NVLS requires BF16 matrices"));
        };
        if source.dtype() != DType::Bf16
            || target.dtype() != DType::Bf16
            || c.implementation().id() != self.id()
        {
            return Err(unsupported("NVLS requires BF16 matrices"));
        }
        let axis = (0..2)
            .find(|&i| source.shape()[i] != target.shape()[i])
            .ok_or_else(|| unsupported("NVLS shard axis"))?;
        let chunk = c
            .implementation()
            .attributes()
            .get("chunk_extent")
            .unwrap_or(0);
        if !sc.is_multiple_of(2) {
            return Err(unsupported("NVLS requires an even row-major column extent"));
        }
        if plan.world_size() <= 1
            || chunk != 128
            || !source.shape()[axis].is_multiple_of(chunk)
            || source.shape()[1 - axis] != target.shape()[1 - axis]
            || source.shape()[axis].checked_mul(plan.world_size()) != Some(target.shape()[axis])
            || !supports_tile(source.shape(), target.shape(), axis, chunk)
        {
            return Err(unsupported("NVLS geometry or attributes"));
        }
        if [source, target].iter().any(|value| {
            matches!(
                value.storage(),
                crate::Storage::Shared | crate::Storage::Register
            )
        }) {
            return Err(EmitError::Contract(
                "local value has no launch binding".into(),
            ));
        }
        Ok(GatherContext {
            source_rows: *sr,
            source_cols: *sc,
            target_rows: *tr,
            target_cols: *tc,
            axis,
            push: true,
            chunk,
        })
    }
}
impl CudaImplementation for NvlsOneShotPushNbi {
    fn schedule(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationSchedule, EmitError> {
        let c = self.context(plan, id)?;
        let source = [c.source_rows, c.source_cols];
        Ok(OperationSchedule {
            dimensions: vec![dimension(
                "chunk",
                source[c.axis],
                c.chunk,
                LoopKind::Parallel,
            )],
            coordinates: vec![variable("chunk")],
            ..Default::default()
        })
    }
    fn phases(&self, plan: &PhysicalPlan, id: OperationId) -> Result<CudaPhaseTemplate, EmitError> {
        let c = self.context(plan, id)?;
        let op = plan.operation(id).unwrap();
        let text = render_template(include_str!("nvls.cu.j2"), &c)?;
        let mut body = CudaPhaseTemplate::cuda_with_parameters(
            &text,
            false,
            &[
                "Element",
                "algorithm",
                "Rows",
                "Cols",
                "layout",
                "destination_layout",
                "src",
                "dst",
                "source_tensor",
                "destination_tensor",
                "start",
                "boundary",
                "status",
            ],
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
        body.epilogue.resources.symmetric_values = vec![op.inputs()[0], op.outputs()[0]];
        body.epilogue.resources.nvls = true;
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
        let source = [c.source_rows, c.source_cols];
        let mut origin = [0, 0];
        origin[c.axis] = coordinate[0];
        let mut extent = source;
        extent[c.axis] = c.chunk;
        let reads = (0..plan.world_size())
            .map(|peer| Region::new(op.inputs()[0], peer, origin, extent))
            .collect();
        let writes = (0..plan.world_size())
            .map(|peer| {
                let mut dst = origin;
                dst[c.axis] += peer * source[c.axis];
                Region::new(op.outputs()[0], rank, dst, extent)
            })
            .collect();
        Ok(Some(Accesses {
            coordinate,
            reads,
            writes,
            ordered_collective: true,
        }))
    }
}
