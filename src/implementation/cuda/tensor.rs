//! SIMT tensor bodies shared by streamed and persistent execution.
use crate::emit::cuda::{
    CudaImplementation, EmitError, OperationEmission, Region, Work, render_template,
};
use crate::{
    AttributeSet, BroadcastImplementation, DType, ImplementationDefinition, ImplementationId,
    ImplementationInstance, OperationId, OperationPayload, PhysicalPlan, PointwiseImplementation,
    ReduceSumImplementation,
};
use serde::Serialize;

const THREADS: usize = 128;

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
    ReduceSum,
    Broadcast,
}

pub(super) struct TensorImplementation {
    kind: Kind,
    name: &'static str,
}

macro_rules! definition {
    ($symbol:ident, $kind:ident, $name:literal) => {
        static $symbol: TensorImplementation = TensorImplementation {
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
pub(super) static REDUCE_SUM: TensorImplementation = TensorImplementation {
    kind: Kind::ReduceSum,
    name: "cuda.reduce_sum",
};
pub(super) static BROADCAST: TensorImplementation = TensorImplementation {
    kind: Kind::Broadcast,
    name: "cuda.broadcast",
};
pub(super) static POINTWISE: &[&dyn PointwiseImplementation] = &[
    &ADD,
    &MUL,
    &DIV,
    &SCALAR_DIV,
    &SQUARE,
    &SQRT,
    &SIGMOID,
    &RELU,
];

fn supported_shape(shape: &[usize]) -> bool {
    matches!(shape.len(), 1 | 2)
        && !shape.contains(&0)
        && shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .is_some_and(|n| n <= i32::MAX as usize)
}

impl TensorImplementation {
    fn attributes(&self, scalar: Option<f32>) -> AttributeSet {
        let mut attrs = vec![("block_threads", THREADS)];
        if let Some(scalar) = scalar {
            attrs.push(("scalar_bits", scalar.to_bits() as usize));
        }
        if matches!(self.kind, Kind::ReduceSum | Kind::Broadcast) {
            attrs.push(("axis", 1));
        }
        AttributeSet::new(attrs)
    }

    fn supports(
        &self,
        dtypes: &[DType],
        shapes: &[&[usize]],
        scalar: Option<f32>,
        axis: usize,
    ) -> bool {
        if dtypes.len() != self.input_count() + 1
            || shapes.len() != dtypes.len()
            || shapes.iter().any(|s| !supported_shape(s))
            || scalar.is_some() != self.requires_scalar()
        {
            return false;
        }
        match self.kind {
            Kind::ReduceSum => {
                axis == 1
                    && shapes[0].len() == 2
                    && shapes[1] == &shapes[0][..1]
                    && dtypes[1] == DType::Fp32
            }
            Kind::Broadcast => {
                axis == 1
                    && shapes[1].len() == 2
                    && shapes[0] == &shapes[1][..1]
                    && dtypes[0] == dtypes[1]
            }
            _ => shapes.iter().all(|s| *s == shapes[0]),
        }
    }
}

impl ImplementationDefinition for TensorImplementation {
    fn id(&self) -> ImplementationId {
        ImplementationId::new(self.name)
    }
    fn cuda(&self) -> Option<&dyn CudaImplementation> {
        Some(self)
    }
}

impl PointwiseImplementation for TensorImplementation {
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
        if !self.supports(dtypes, shapes, scalar, 1) {
            return vec![];
        }
        vec![ImplementationInstance::new(self, self.attributes(scalar))]
    }
}

impl ReduceSumImplementation for TensorImplementation {
    fn enumerate(
        &'static self,
        dtypes: [DType; 2],
        shapes: [&[usize]; 2],
        axis: usize,
    ) -> Vec<ImplementationInstance> {
        if !self.supports(&dtypes, &shapes, None, axis) {
            return vec![];
        }
        vec![ImplementationInstance::new(self, self.attributes(None))]
    }
}

impl BroadcastImplementation for TensorImplementation {
    fn enumerate(
        &'static self,
        dtype: DType,
        shapes: [&[usize]; 2],
        axis: usize,
    ) -> Vec<ImplementationInstance> {
        if !self.supports(&[dtype; 2], &shapes, None, axis) {
            return vec![];
        }
        vec![ImplementationInstance::new(self, self.attributes(None))]
    }
}

#[derive(Serialize)]
struct Context {
    operation: usize,
    inputs: Vec<usize>,
    input_types: Vec<&'static str>,
    output: usize,
    output_type: &'static str,
    vector: bool,
    columns: usize,
    length: usize,
    reduction: bool,
    broadcast: bool,
    expression: &'static str,
    scalar_bits: u32,
}

fn cuda_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bf16 => "cutlass::bfloat16_t",
        DType::Fp32 => "float",
    }
}

impl CudaImplementation for TensorImplementation {
    fn specialize(
        &self,
        plan: &PhysicalPlan,
        id: OperationId,
    ) -> Result<OperationEmission, EmitError> {
        let invalid =
            || EmitError::Unsupported(format!("{} operand/attribute contract", self.name));
        let op = plan.operation(id).ok_or_else(invalid)?;
        let OperationPayload::Compute(compute) = op.payload() else {
            return Err(invalid());
        };
        let instance = compute.implementation();
        let attrs = instance.attributes();
        let scalar_bits = attrs
            .get("scalar_bits")
            .map(u32::try_from)
            .transpose()
            .map_err(|_| invalid())?;
        let scalar = scalar_bits.map(f32::from_bits);
        let values = op
            .inputs()
            .iter()
            .chain(op.outputs())
            .map(|&v| plan.value_instance(v).ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?;
        let shapes: Vec<_> = values.iter().map(|v| v.shape()).collect();
        let dtypes: Vec<_> = values.iter().map(|v| v.dtype()).collect();
        if op.inputs().len() != self.input_count()
            || op.outputs().len() != 1
            || instance.id() != self.id()
            || *attrs != self.attributes(scalar)
            || !self.supports(&dtypes, &shapes, scalar, attrs.get("axis").unwrap_or(1))
        {
            return Err(invalid());
        }
        let output = op.outputs()[0];
        let out_shape = shapes[shapes.len() - 1];
        let reduction = matches!(self.kind, Kind::ReduceSum);
        let broadcast = matches!(self.kind, Kind::Broadcast);
        let vector = out_shape.len() == 1;
        let columns = if reduction {
            shapes[0][1]
        } else if vector {
            1
        } else {
            out_shape[1]
        };
        let work = (0..plan.world_size())
            .map(|rank| {
                let mut work = vec![];
                if reduction {
                    for row in 0..out_shape[0] {
                        work.push(Work {
                            coordinate: [row, 0, 0],
                            reads: vec![Region::new(op.inputs()[0], rank, [row, 0], [1, columns])],
                            writes: vec![Region::new(output, rank, [row, 0], [1, 1])],
                            ..Work::default()
                        });
                    }
                } else {
                    let rows = if vector { 1 } else { out_shape[0] };
                    let width = if vector { out_shape[0] } else { columns };
                    for row in 0..rows {
                        for start in (0..width).step_by(THREADS) {
                            let valid = THREADS.min(width - start);
                            let (origin, extent) = if vector {
                                ([start, 0], [valid, 1])
                            } else {
                                ([row, start], [1, valid])
                            };
                            work.push(Work {
                                coordinate: [row, start, 0],
                                reads: op
                                    .inputs()
                                    .iter()
                                    .map(|&input| {
                                        if broadcast {
                                            Region::new(input, rank, [row, 0], [1, 1])
                                        } else {
                                            Region::new(input, rank, origin, extent)
                                        }
                                    })
                                    .collect(),
                                writes: vec![Region::new(output, rank, origin, extent)],
                                ..Work::default()
                            });
                        }
                    }
                }
                work
            })
            .collect();
        let expression = match self.kind {
            Kind::Add => "x + y",
            Kind::Mul => "x * y",
            Kind::Div => "x / y",
            Kind::ScalarDiv => "x / __uint_as_float(scalar_bits)",
            Kind::Square => "x * x",
            Kind::Sqrt => "sqrtf(x)",
            Kind::Sigmoid => "1.0f / (1.0f + expf(-x))",
            // Comparison preserves NaN and signed zero, unlike fmaxf(x, 0).
            Kind::Relu => "x < 0.0f ? 0.0f : x",
            Kind::ReduceSum | Kind::Broadcast => "x",
        };
        Ok(OperationEmission {
            body: render_template(
                include_str!("templates/tensor.cu.j2"),
                &Context {
                    operation: id.index(),
                    inputs: op.inputs().iter().map(|v| v.index()).collect(),
                    input_types: dtypes[..op.inputs().len()]
                        .iter()
                        .map(|&d| cuda_type(d))
                        .collect(),
                    output: output.index(),
                    output_type: cuda_type(dtypes[dtypes.len() - 1]),
                    vector,
                    columns,
                    length: out_shape[0],
                    reduction,
                    broadcast,
                    expression,
                    scalar_bits: scalar_bits.unwrap_or(0),
                },
            )?,
            work,
            shared_memory_bytes: if reduction { THREADS * 4 } else { 0 },
            symmetric_values: vec![],
            nvls: false,
        })
    }
}
