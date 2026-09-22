use crate::DType;
use crate::emit::implementation::{
    AllGatherImplementation, AttributeSet, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};

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
    fn id(&self) -> ImplementationId {
        ImplementationId::new("nvls.one_shot_push_nbi")
    }
}
