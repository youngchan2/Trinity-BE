//! Triton independently launched kernels using the existing fallback emitter.
use super::{KernelContext, KernelProvider, ProviderError};
use crate::emit::{candidate::CandidateSpecification, request::KernelRequest};
use crate::triton::TritonPlan;
use crate::{CudaTargetCapability, TargetCapability};

mod adapter;
mod program;
pub use program::TritonProgram;

pub struct TritonKernelProvider;

#[derive(Debug, Clone)]
pub struct TritonSpecification {
    request: KernelRequest,
    plan: TritonPlan,
}
impl TritonSpecification {
    pub fn request(&self) -> &KernelRequest {
        &self.request
    }
    pub fn plan(&self) -> &TritonPlan {
        &self.plan
    }
    pub fn emit_python(&self) -> String {
        let arguments = self
            .request
            .tensors
            .keys()
            .map(|id| format!("v{}=values[{}]", id.index(), id.index()))
            .collect::<Vec<_>>()
            .join(", ");
        let id = self.request.output.index();
        let expected: Vec<_> = self
            .request
            .tensors
            .values()
            .map(|v| {
                (
                    v.value.index(),
                    crate::triton::TensorDType::from(v.dtype).python(),
                    &v.shape,
                )
            })
            .collect();
        let inputs: Vec<_> = self.request.inputs.iter().map(|v| v.index()).collect();
        let TargetCapability::Cuda(target) = self.request.target;
        let capability = match target {
            CudaTargetCapability::Hopper => (9, 0),
            CudaTargetCapability::Sm89 => (8, 9),
            CudaTargetCapability::Sm120 => (12, 0),
        };
        format!(
            "{}\n_EXPECTED = {expected:?}\n_INPUT_IDS = {inputs:?}\n_OUTPUT_ID = {id}\n_CAPABILITY = {capability:?}\n{}",
            self.plan.emit(),
            include_str!("wrapper.py.in").replace("@ARGUMENTS@", &arguments)
        )
    }
}
impl KernelProvider for TritonKernelProvider {
    fn name(&self) -> &str {
        "triton"
    }
    fn candidates(
        &self,
        context: &KernelContext<'_, '_>,
    ) -> Result<Vec<CandidateSpecification>, ProviderError> {
        let request = KernelRequest::from_context(context)?;
        let plan = adapter::lower(&request)
            .map_err(|error| ProviderError::Unsupported(error.to_string()))?;
        Ok(vec![CandidateSpecification::Triton(Box::new(
            TritonSpecification { request, plan },
        ))])
    }
}
