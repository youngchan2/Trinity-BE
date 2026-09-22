//! CuTe kernel provider and operator dispatch.

use super::{
    Kernel, KernelBindings, KernelContext, KernelImplementation, KernelProvider, ProviderError,
    SpecifiedKernel,
};
use crate::TargetCapability;
use crate::emit::execution::ExecutionModel;

mod operator;
pub(in crate::emit) use operator::{
    HopperGemmSpecification, PointwiseSpecification, ReduceSumSpecification, Sm80GemmSpecification,
};

pub(in crate::emit) struct CuTeKernelProvider;

impl KernelProvider for CuTeKernelProvider {
    fn name(&self) -> &str {
        "cute"
    }

    fn specify(&self, context: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError> {
        if context
            .prepared
            .plan
            .operation(context.operation)
            .unwrap()
            .expression()
            .accesses()
            .iter()
            .any(|a| {
                context
                    .prepared
                    .plan
                    .value_instance(a.value)
                    .unwrap()
                    .dtype()
                    == crate::DType::Fp16
            })
        {
            return Err(ProviderError::Unsupported(
                "CuTe templates currently support BF16/FP32; FP16 uses Triton".into(),
            ));
        }
        match (context.prepared.plan.target(), context.execution) {
            (
                TargetCapability::Cuda(_),
                ExecutionModel::CudaStreamed | ExecutionModel::CudaPersistent,
            ) => {}
        }

        let implementations: [&dyn KernelImplementation; 4] = [
            &operator::HopperGemmKernel,
            &operator::Sm80GemmKernel,
            &operator::ReduceSumKernel,
            &operator::PointwiseKernel,
        ];
        let mut reasons = Vec::new();
        for implementation in implementations {
            match implementation.specify(context) {
                Ok(specification) => return Ok(specification),
                Err(ProviderError::Unsupported(reason)) => reasons.push(reason),
                Err(error) => return Err(error),
            }
        }
        Err(ProviderError::Unsupported(reasons.join("; ")))
    }

    fn render(
        &self,
        specification: &SpecifiedKernel,
        bindings: &KernelBindings,
    ) -> Result<Kernel, ProviderError> {
        match specification {
            SpecifiedKernel::CuTeSm80Gemm(_) => {
                operator::Sm80GemmKernel.render(specification, bindings)
            }
            SpecifiedKernel::CuTeHopperGemm(_) => {
                operator::HopperGemmKernel.render(specification, bindings)
            }
            SpecifiedKernel::CuTeReduceSum(_) => {
                operator::ReduceSumKernel.render(specification, bindings)
            }
            SpecifiedKernel::CuTePointwise(_) => {
                operator::PointwiseKernel.render(specification, bindings)
            }
        }
    }
}
