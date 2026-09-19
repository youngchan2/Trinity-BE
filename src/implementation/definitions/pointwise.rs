//! Pointwise SIMT implementations.
use super::expression::{THREADS, supported_shape};
use crate::PointwiseImplementation;
use crate::{
    AttributeSet, DType, ImplementationDefinition, ImplementationId, ImplementationInstance,
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
