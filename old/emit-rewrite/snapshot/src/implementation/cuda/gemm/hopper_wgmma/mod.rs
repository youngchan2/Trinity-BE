use crate::emit::cuda::{AccumulationBody, AccumulationScope, Binding};
use crate::emit::cuda::{CudaImplementation, CudaPhaseTemplate, EmitError, OperationSchedule};
use crate::implementation::{
    AttributeSet, GemmImplementation, ImplementationDefinition, ImplementationId,
    ImplementationInstance,
};
use crate::plan::normalize::*;
use crate::{DType, Operation, OperationPayload, PhysicalPlan, Storage};
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
    fn cuda(&self) -> Option<&dyn crate::emit::cuda::CudaImplementation> {
        Some(self)
    }

    fn id(&self) -> ImplementationId {
        WGMMA_ID
    }
}

pub(in crate::implementation::cuda) struct FusionGemmShape {
    pub(in crate::implementation::cuda) m: usize,
    pub(in crate::implementation::cuda) n: usize,
    pub(in crate::implementation::cuda) k: usize,
}

/// Geometry accepted by the fixed WGMMA body. The SS adapter accepts shared A;
/// a pointwise consumer can instead use the producer's register traversal.
/// Every handoff preserves the declared output conversion before consumption.
pub(in crate::implementation::cuda) fn fusion_shape(
    plan: &PhysicalPlan,
    operation: &Operation,
) -> Option<FusionGemmShape> {
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
    if [lhs.dtype(), rhs.dtype()] != [DType::Bf16; 2] {
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

impl CudaImplementation for HopperWgmmaBf16 {
    fn stage_accesses(
        &self,
        _plan: &PhysicalPlan,
        _id: OperationId,
        domain: &crate::LoopDomain,
        expression: &crate::Expression,
    ) -> Result<Option<crate::emit::cuda::StageAccessPattern>, EmitError> {
        let step = domain
            .step
            .evaluate(&std::collections::BTreeMap::new())
            .map_err(fail)?;
        if step <= 0 || step % 64 != 0 {
            return Err(fail("invalid WGMMA K step"));
        }
        fn physical_tiles(e: &mut crate::Expression, var: &str) {
            if let crate::Expression::List(xs) = e {
                if xs.len() == 3 && xs[0].atom() == Some("tile") && xs[1].atom() == Some(var) {
                    xs[2] = crate::Expression::Atom("64".into());
                }
                for x in xs {
                    physical_tiles(x, var);
                }
            }
        }
        let mut expression = expression.clone();
        physical_tiles(&mut expression, &domain.variable);
        let mut domain = domain.clone();
        domain.step = crate::IndexExpr::Constant(64);
        Ok(Some(crate::emit::cuda::StageAccessPattern {
            domain,
            expression,
            whole_compute_inputs: true,
        }))
    }
    fn accumulation(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
        scope: &AccumulationScope,
    ) -> Result<AccumulationBody, EmitError> {
        self.bind_accumulation(plan, id, scope)
    }
    fn schedule(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationSchedule, EmitError> {
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
    fn phases(&self, plan: &PhysicalPlan, id: OperationId) -> Result<CudaPhaseTemplate, EmitError> {
        let op = plan.operation(id).unwrap();
        let OperationPayload::Compute(c) = op.payload() else {
            return Err(unsupported("WGMMA payload"));
        };
        let attrs = c.implementation().attributes();
        if attrs.get("tile_m") != Some(128)
            || attrs.get("tile_n") != Some(128)
            || attrs.get("tile_k") != Some(64)
        {
            return Err(unsupported("WGMMA attributes"));
        }
        let mut body = CudaPhaseTemplate::template(
            include_str!("prologue.cu"),
            Some(include_str!("mainloop.cu")),
            include_str!("epilogue.cu"),
            &[
                "Element",
                "la",
                "lb",
                "Shared",
                "shared",
                "mma",
                "tmma",
                "cc",
                "accumulator",
                "load",
                "sa",
                "sb",
                "linear",
                "row",
                "col",
                "dst",
                "bytes",
                "src",
                "slot",
                "stage_base",
                "stage_count",
                "iteration",
                "prefetch",
                "fa",
                "fb",
                "i",
                "copy",
            ],
            &[
                "stage_cursor",
                "COPY_A",
                "INITIALIZER",
                "VAR",
                "START",
                "STOP",
                "AO",
                "BO",
                "CO",
                "CTYPE",
                "M",
                "A",
                "B",
                "C",
                "OUTPUT",
            ],
        )?;
        body.prologue.resources.shared_memory_bytes = 65536;
        Ok(body)
    }
}

fn fail(s: impl Into<String>) -> EmitError {
    EmitError::Unsupported(s.into())
}
fn ctype(dtype: DType) -> &'static str {
    match dtype {
        DType::Bf16 => "cutlass::bfloat16_t",
        DType::Fp32 => "float",
    }
}
impl HopperWgmmaBf16 {
    fn bind_accumulation(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
        scope: &AccumulationScope,
    ) -> Result<AccumulationBody, EmitError> {
        let [a, b] = &scope.inputs;
        let c = &scope.output;
        if a.dtype != DType::Bf16
            || b.dtype != DType::Bf16
            || a.width.len() != 2
            || b.width.len() != 2
            || a.width[0] > 128
            || a.width[0] == 0
            || b.width[1] != 128
            || a.width[1] != b.width[0]
            || a.width[1] != scope.step as usize
            || scope.step % 64 != 0
            || a.origin[1] != scope.variable
            || b.origin[0] != scope.variable
        {
            return Err(fail(
                "WGMMA requires M<=128, N tile=128 and contiguous K tiles divisible by 64",
            ));
        }
        let m_capacity = a.width[0];
        let m = &a.valid_width[0];
        let expected = if c.width.len() == 3 {
            vec![1, m_capacity, 128]
        } else {
            vec![m_capacity, 128]
        };
        if c.width != expected
            || b.valid_width[1] != "128"
            || a.valid_width[1] != a.width[1].to_string()
            || b.valid_width[0] != b.width[0].to_string()
        {
            return Err(fail("WGMMA output tile mismatch"));
        }
        let ao = a.offset(&["${row}".into(), "${col}".into()]);
        let bo = b.offset(&["${row}".into(), "${col}".into()]);
        let mut coords = vec![
            "int(get<0>(${cc}(${i})))".into(),
            "int(get<1>(${cc}(${i})))".into(),
        ];
        if c.width.len() == 3 {
            coords.insert(0, "0".into());
        }
        let co = c.offset(&coords);
        let initialized = scope.initialized;
        let initial = if initialized {
            format!(
                "CUTE_UNROLL\n  for(int ${{i}}=0;${{i}}<size(${{accumulator}});++${{i}}) ${{accumulator}}(${{i}})=get<0>(${{cc}}(${{i}}))<{m}?float(${{buffer_{}}}[{co}]):0.0f;",
                c.value.index()
            )
        } else {
            "clear(${accumulator});".into()
        };
        let mut phases = self.phases(plan, id)?;
        for name in ["x", "y"] {
            phases.declare(name, false);
        }
        for (value, _) in plan.value_instances() {
            phases.declare(&format!("buffer_{}", value.index()), true);
        }
        if initialized {
            let symbol = phases.symbol(&format!("buffer_{}", c.value.index()))?;
            phases.prologue.inputs.push(Binding {
                symbol,
                value: c.value,
                dtype: c.dtype,
                storage: c.storage,
            });
        }
        if b.storage == Storage::Shared {
            return Err(fail(
                "shared B adapter is not supported by this WGMMA backend",
            ));
        }
        let copy = if a.storage == Storage::Shared {
            "for(int ${copy}=0;${copy}<8;++${copy}) ${sa}(${row},${col}+${copy})=${row}<${M}?${src}[${copy}]:${Element}(0.0f);"
        } else {
            "asm volatile(\"cp.async.ca.shared.global [%0], [%1], 16, %2;\" :: \"r\"(${dst}), \"l\"(${src}), \"r\"(${bytes}) : \"memory\");"
        };
        phases.bind(phases.symbol("COPY_A")?, phases.code(copy)?);
        for (name, text) in [("INITIALIZER", initial), ("AO", ao), ("BO", bo), ("CO", co)] {
            let code = phases.code(&text)?;
            phases.bind(phases.symbol(name)?, code);
        }
        for (name, access) in [("A", &a), ("B", &b)] {
            let value = access.value;
            let binding = Binding {
                symbol: phases.symbol(name)?,
                value,
                dtype: access.dtype,
                storage: access.storage,
            };
            phases.prologue.inputs.push(binding.clone());
            phases.mainloop.as_mut().unwrap().inputs.push(binding);
        }
        for (name, text) in [
            ("VAR", scope.variable.clone()),
            ("START", scope.start.clone()),
            ("STOP", scope.stop.clone()),
            ("CTYPE", ctype(c.dtype).into()),
            ("M", m.to_string()),
        ] {
            phases.bind_text(name, text)?;
        }
        for (name, access) in [("A", &a), ("B", &b)] {
            if access.storage != Storage::Shared {
                phases.bind_text(
                    name,
                    access
                        .pointer
                        .clone()
                        .ok_or_else(|| fail("accumulation input has no address"))?,
                )?;
            }
        }
        Ok(AccumulationBody {
            output_symbol: phases.symbol("OUTPUT")?,
            body: phases,
            output_coordinates: coords,
            output_expression: "${accumulator}(${i})".into(),
        })
    }
}
