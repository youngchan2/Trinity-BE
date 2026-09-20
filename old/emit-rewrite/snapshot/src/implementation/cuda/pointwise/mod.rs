//! Pointwise SIMT implementations.
use super::expression::{THREADS, scalar_expression, supported_shape};
use crate::LoopKind;
use crate::PointwiseImplementation;
use crate::emit::cuda::{CudaImplementation, CudaPhaseTemplate, EmitError, OperationSchedule};
use crate::plan::normalize::*;
use crate::{
    AttributeSet, DType, ImplementationDefinition, ImplementationId, ImplementationInstance,
    OperationId, OperationPayload, PhysicalPlan,
};
#[derive(Clone, Copy)]
enum Kind {
    Add,
    Mul,
    Div,
    ScalarDiv,
    Square,
    Sqrt,
    Sigmoid,
    Relu,
}

pub(super) struct Pointwise {
    kind: Kind,
    name: &'static str,
}

macro_rules! definition {
    ($symbol:ident, $kind:ident, $name:literal) => {
        static $symbol: Pointwise = Pointwise {
            kind: Kind::$kind,
            name: $name,
        };
    };
}
definition!(ADD, Add, "cuda.add");
definition!(MUL, Mul, "cuda.mul");
definition!(DIV, Div, "cuda.div");
definition!(SCALAR_DIV, ScalarDiv, "cuda.scalar_div");
definition!(SQUARE, Square, "cuda.square");
definition!(SQRT, Sqrt, "cuda.sqrt");
definition!(SIGMOID, Sigmoid, "cuda.sigmoid");
definition!(RELU, Relu, "cuda.relu");
pub(super) static IMPLEMENTATIONS: &[&dyn PointwiseImplementation] = &[
    &ADD,
    &MUL,
    &DIV,
    &SCALAR_DIV,
    &SQUARE,
    &SQRT,
    &SIGMOID,
    &RELU,
];

impl Pointwise {
    fn attributes(&self, scalar: Option<f32>) -> AttributeSet {
        let mut attrs = vec![("block_threads", THREADS)];
        if let Some(scalar) = scalar {
            attrs.push(("scalar_bits", scalar.to_bits() as usize));
        }
        AttributeSet::new(attrs)
    }

    fn supports(&self, dtypes: &[DType], shapes: &[&[usize]], scalar: Option<f32>) -> bool {
        if dtypes.len() != self.input_count() + 1
            || shapes.len() != dtypes.len()
            || shapes.iter().any(|s| !supported_shape(s))
            || scalar.is_some() != self.requires_scalar()
        {
            return false;
        }
        shapes.iter().all(|s| *s == shapes[0])
    }
}

impl ImplementationDefinition for Pointwise {
    fn id(&self) -> ImplementationId {
        ImplementationId::new(self.name)
    }
    fn cuda(&self) -> Option<&dyn CudaImplementation> {
        Some(self)
    }
}

impl PointwiseImplementation for Pointwise {
    fn input_count(&self) -> usize {
        if matches!(self.kind, Kind::Add | Kind::Mul | Kind::Div) {
            2
        } else {
            1
        }
    }
    fn requires_scalar(&self) -> bool {
        matches!(self.kind, Kind::ScalarDiv)
    }
    fn enumerate(
        &'static self,
        dtypes: &[DType],
        shapes: &[&[usize]],
        scalar: Option<f32>,
    ) -> Vec<ImplementationInstance> {
        if !self.supports(dtypes, shapes, scalar) {
            return vec![];
        }
        vec![ImplementationInstance::new(self, self.attributes(scalar))]
    }
}

impl CudaImplementation for Pointwise {
    fn scalar_expression(&self, operator: &str) -> Option<&'static str> {
        scalar_expression(operator)
    }
    fn schedule(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationSchedule, EmitError> {
        self.phases(plan, id)?;
        let op = plan.operation(id).unwrap();
        let out = op.outputs()[0];
        let shape = plan.value_instance(out).unwrap().shape();
        let vector = shape.len() == 1;
        let columns = if vector { shape[0] } else { shape[1] };
        let parts = if vector {
            vec![clipped_tile("col", 128)]
        } else {
            vec![tile("row", 1), clipped_tile("col", 128)]
        };
        let operand = |id| load(plan, id, parts.clone());
        let x = operand(op.inputs()[0]);
        let rhs = match self.kind {
            Kind::Add | Kind::Mul | Kind::Div => expr(
                match self.kind {
                    Kind::Add => "+",
                    Kind::Mul => "*",
                    _ => "/",
                },
                [x, operand(op.inputs()[1])],
            ),
            Kind::ScalarDiv => {
                let OperationPayload::Compute(c) = op.payload() else {
                    unreachable!()
                };
                expr(
                    "/",
                    [
                        x,
                        expr(
                            "float_bits",
                            [atom(
                                c.implementation().attributes().get("scalar_bits").unwrap(),
                            )],
                        ),
                    ],
                )
            }
            Kind::Square | Kind::Sqrt | Kind::Sigmoid | Kind::Relu => expr(
                match self.kind {
                    Kind::Square => "sqr",
                    Kind::Sqrt => "sqrt",
                    Kind::Sigmoid => "sigmoid",
                    _ => "relu",
                },
                [x],
            ),
        };
        let mut dimensions = Vec::new();
        if !vector {
            dimensions.push(dimension("row", shape[0], 1, LoopKind::Parallel));
        }
        dimensions.push(dimension(
            "col",
            columns.div_ceil(128) * 128,
            128,
            LoopKind::Parallel,
        ));
        Ok(OperationSchedule {
            expression: Some(store(plan, out, rhs, parts)),
            dimensions,
            coordinates: if vector {
                vec![crate::IndexExpr::Constant(0), variable("col")]
            } else {
                vec![variable("row"), variable("col")]
            },
        })
    }
    fn phases(&self, plan: &PhysicalPlan, id: OperationId) -> Result<CudaPhaseTemplate, EmitError> {
        let invalid = || unsupported("tensor geometry, dtype or attributes");
        let op = plan.operation(id).unwrap();
        let OperationPayload::Compute(c) = op.payload() else {
            return Err(invalid());
        };
        let attrs = c.implementation().attributes();
        let scalar = attrs
            .get("scalar_bits")
            .map(u32::try_from)
            .transpose()
            .map_err(|_| invalid())?
            .map(f32::from_bits);
        // Expression normalization deduplicates logical bindings. A binary
        // operation can still use that one tensor in both operand positions.
        let mut inputs = op.inputs().to_vec();
        if self.input_count() == 2 && inputs.len() == 1 && op.expression().is_some() {
            inputs.push(inputs[0]);
        }
        let values = inputs
            .iter()
            .chain(op.outputs())
            .map(|&v| plan.value_instance(v).ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?;
        let shapes: Vec<_> = values.iter().map(|v| v.shape()).collect();
        let dtypes: Vec<_> = values.iter().map(|v| v.dtype()).collect();
        if inputs.len() != self.input_count()
            || op.outputs().len() != 1
            || *attrs != self.attributes(scalar)
            || !self.supports(&dtypes, &shapes, scalar)
        {
            return Err(invalid());
        }
        Ok(CudaPhaseTemplate::default())
    }
}
