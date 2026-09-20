use crate::implementation::{
    AttributeSet, GemmImplementation, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};
use crate::implementation::{OperationSchedule, ScheduleError};
use crate::plan::normalize::*;
use crate::{DType, PhysicalPlan};
use crate::{LoopKind, OperationId};

const WGMMA_ID: ImplementationId = ImplementationId::new("hopper.wgmma.bf16");

pub(super) static HOPPER_WGMMA_BF16: HopperWgmmaBf16 = HopperWgmmaBf16;

pub(super) struct HopperWgmmaBf16;

impl GemmImplementation for HopperWgmmaBf16 {
    fn enumerate_scheduled(
        &'static self,
        dtypes: [DType; 3],
        tiles: [&[usize]; 3],
    ) -> Vec<ImplementationInstance> {
        let ([m, k], [bk, n]) = (tiles[0], tiles[1]) else {
            return Vec::new();
        };
        if dtypes[..2] != [DType::Bf16; 2]
            || *m == 0
            || *m > 128
            || *n != 128
            || *k == 0
            || !k.is_multiple_of(64)
            || k != bk
            || !(tiles[2] == [*m, *n] || tiles[2] == [1, *m, *n])
        {
            return Vec::new();
        }
        vec![ImplementationInstance::new(
            self,
            AttributeSet::new([("tile_m", 128), ("tile_n", 128), ("tile_k", 64)]),
        )]
    }

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
    fn id(&self) -> ImplementationId {
        WGMMA_ID
    }

    fn schedule(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationSchedule, ScheduleError> {
        let op = plan.operation(id).unwrap();
        let ([lhs, rhs], [out]) = (op.inputs(), op.outputs()) else {
            return Err(unsupported("WGMMA arity"));
        };
        let (a, b, c) = (
            plan.value_instance(*lhs).unwrap(),
            plan.value_instance(*rhs).unwrap(),
            plan.value_instance(*out).unwrap(),
        );
        let ([m, k], [bk, n]) = (a.shape(), b.shape()) else {
            return Err(unsupported("WGMMA matrices"));
        };
        if k != bk
            || c.shape() != [*m, *n]
            || *m == 0
            || *n == 0
            || *k == 0
            || !m.is_multiple_of(128)
            || !n.is_multiple_of(128)
            || !k.is_multiple_of(64)
        {
            return Err(unsupported("WGMMA geometry"));
        }
        let ai = vec![tile("m", 128), tile("k", 64)];
        let bi = vec![tile("k", 64), tile("n", 128)];
        let ci = vec![tile("m", 128), tile("n", 128)];
        let rhs = expr(
            "+",
            [
                load(plan, *out, ci.clone()),
                expr("@", [load(plan, *lhs, ai), load(plan, *rhs, bi)]),
            ],
        );
        Ok(OperationSchedule {
            expression: Some(store(plan, *out, rhs, ci)),
            dimensions: vec![
                dimension("m", *m, 128, LoopKind::Parallel),
                dimension("n", *n, 128, LoopKind::Parallel),
                dimension("k", *k, 64, LoopKind::Sequential),
            ],
            coordinates: vec![quotient("m", 128), quotient("n", 128)],
        })
    }
}
