//! Row-wise FP32 accumulation across a sequential loop.

use crate::Storage;
use crate::emit::native::{KernelInterface, KernelPort, RegisterLayout};

use super::{access::Access, accumulation, unsupported};
use crate::emit::native::{
    Kernel, KernelBindings, KernelImplementation, KernelRequirements, SpecifiedKernel, ThreadPolicy,
};
use crate::emit::provider::{KernelContext, ProviderError};
use crate::{DType, Expression};

mod render;

pub(in crate::emit::provider::cute) struct ReduceSumKernel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) struct ReduceSumSpecification {
    pub requirements: KernelRequirements,
    pub iteration: String,
    pub input: Access,
    pub output: Access,
    pub square: bool,
}

impl KernelImplementation for ReduceSumKernel {
    fn specify(&self, context: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError> {
        let (loop_, rhs) = accumulation(context)?;
        let Expression::ReduceSum { value, axis: 1 } = rhs else {
            return Err(unsupported(
                "CuTe reduction requires an axis-1 sum accumulation",
            ));
        };
        let (value, square) = match value.as_ref() {
            Expression::Sqr(value) => (value.as_ref(), true),
            value => (value, false),
        };
        let Expression::Load(source) = value else {
            return Err(unsupported("CuTe reduction supports load or sqr(load)"));
        };
        let Expression::Store { destination, .. } = context
            .prepared
            .plan
            .operation(context.operation)
            .unwrap()
            .expression()
        else {
            unreachable!()
        };
        let input = Access::from_plan(context, source)?;
        let output = Access::from_plan(context, destination)?;
        if !matches!(input.storage, Storage::External | Storage::Global)
            || output.storage == Storage::Shared
        {
            return Err(unsupported(
                "reduction requires a memory input and memory or Register output",
            ));
        }
        if input.shape.len() != 2
            || output.shape.len() != 1
            || output.dtype != DType::Fp32
            || output.shape[0] != input.shape[0]
            || output.axes[0] != input.axes[0]
        {
            return Err(unsupported(
                "CuTe reduction requires matching rows and a rank-1 FP32 output",
            ));
        }
        let requirements = KernelRequirements {
            shared_memory_bytes: 0,
            shared_memory_alignment: 1,
            thread_policy: ThreadPolicy::Subgroup { threads: 128 },
            alignments: vec![(input.value, input.dtype.size_bytes()), (output.value, 4)],
        };
        Ok(SpecifiedKernel::CuTeReduceSum(ReduceSumSpecification {
            iteration: loop_.domain.variable.clone(),
            requirements,
            input,
            output,
            square,
        }))
    }

    fn render(
        &self,
        specification: &SpecifiedKernel,
        bindings: &KernelBindings,
    ) -> Result<Kernel, ProviderError> {
        let SpecifiedKernel::CuTeReduceSum(specification) = specification else {
            return Err(ProviderError::Failed(
                "expected CuTe ReduceSum specification".into(),
            ));
        };
        render::render(specification, bindings)
    }
}

impl ReduceSumSpecification {
    pub(in crate::emit) fn interface(&self) -> KernelInterface {
        KernelInterface {
            inputs: vec![KernelPort::memory(&self.input)],
            outputs: vec![KernelPort {
                access: self.output.clone(),
                register: Some(RegisterLayout::Fixed {
                    representation: "cuda.scalar",
                    distribution: "cute.warp_rows.128".into(),
                }),
                projection: (0..self.output.axes.len()).collect(),
            }],
            iteration: Some(self.iteration.clone()),
        }
    }
}
