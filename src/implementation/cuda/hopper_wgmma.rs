use super::super::{
    AttributeSet, GemmImplementation, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};
use crate::{DType, Operation, OperationPayload, PhysicalPlan};

pub(super) const WGMMA_ID: ImplementationId = ImplementationId::new("hopper.wgmma.bf16");

pub(super) static HOPPER_WGMMA_BF16: HopperWgmmaBf16 = HopperWgmmaBf16;

pub(super) struct HopperWgmmaBf16;

impl GemmImplementation for HopperWgmmaBf16 {
    fn enumerate(
        &'static self,
        dtypes: [DType; 3],
        shapes: [&[usize]; 3],
    ) -> Vec<ImplementationInstance> {
        let [lhs_shape, rhs_shape, output_shape] = shapes;

        if dtypes != [DType::Bf16; 3] {
            return Vec::new();
        }

        let ([m, k], [rhs_k, n], [output_m, output_n]) = (lhs_shape, rhs_shape, output_shape)
        else {
            return Vec::new();
        };

        assert_eq!(k, rhs_k, "a prevalidated GEMM must have matching K extents");
        assert_eq!(
            [output_m, output_n],
            [m, n],
            "a prevalidated GEMM output must have shape [M, N]"
        );

        self.tile_candidates()
            .filter(|[tile_m, tile_n, tile_k]| {
                m.is_multiple_of(*tile_m) && n.is_multiple_of(*tile_n) && k.is_multiple_of(*tile_k)
            })
            .map(|[tile_m, tile_n, tile_k]| {
                ImplementationInstance::new(
                    self,
                    AttributeSet::new([("tile_m", tile_m), ("tile_n", tile_n), ("tile_k", tile_k)]),
                )
            })
            .collect()
    }
}

impl HopperWgmmaBf16 {
    /// Returns the concrete WGMMA tile shapes.
    fn tile_candidates(&self) -> impl ExactSizeIterator<Item = [usize; 3]> {
        // TODO: Source tile candidates from the autotuner.
        [[128, 128, 64]].into_iter()
    }
}

impl ImplementationDefinition for HopperWgmmaBf16 {
    fn cuda(&self) -> Option<&dyn crate::emit::cuda::CudaImplementation> {
        Some(self)
    }

    fn id(&self) -> ImplementationId {
        WGMMA_ID
    }
}

pub(super) struct FusionGemmShape {
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
}

/// The fixed 128x128x64 body supports Shared A/B and Register A operands.
/// Register handoff uses the Hopper C-to-A fragment layout conversion (as in
/// CUTLASS's Hopper FMHA), with the same M ownership in both warp groups.
/// FP32 accumulators must be rounded to BF16 before either Shared or Register
/// handoff. The next GEMM starts its own FP32 accumulation; fusion never
/// reassociates the two GEMMs or bypasses the intermediate rounding.
pub(super) fn fusion_shape(plan: &PhysicalPlan, operation: &Operation) -> Option<FusionGemmShape> {
    let OperationPayload::Compute(compute) = operation.payload() else {
        return None;
    };
    let instance = compute.implementation();
    let attributes = instance.attributes();
    if instance.id() != WGMMA_ID
        || attributes.get("tile_m") != Some(128)
        || attributes.get("tile_n") != Some(128)
        || attributes.get("tile_k") != Some(64)
    {
        return None;
    }
    let ([lhs, rhs], [output]) = (operation.inputs(), operation.outputs()) else {
        return None;
    };
    let lhs = plan.value_instance(*lhs)?;
    let rhs = plan.value_instance(*rhs)?;
    let output = plan.value_instance(*output)?;
    if [lhs.dtype(), rhs.dtype(), output.dtype()] != [DType::Bf16; 3] {
        return None;
    }
    let ([m, k], [rhs_k, n], [out_m, out_n]) = (lhs.shape(), rhs.shape(), output.shape()) else {
        return None;
    };
    if k != rhs_k
        || m != out_m
        || n != out_n
        || *m == 0
        || *n == 0
        || *k == 0
        || !m.is_multiple_of(128)
        || !n.is_multiple_of(128)
        || !k.is_multiple_of(64)
    {
        return None;
    }
    Some(FusionGemmShape {
        m: *m,
        n: *n,
        k: *k,
    })
}
