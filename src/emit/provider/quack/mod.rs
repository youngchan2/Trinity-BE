//! Optional Quack host calls. These are opaque kernels, not CuTe C++ fragments.

use super::{KernelContext, KernelProvider, ProviderError};
use crate::emit::candidate::CandidateSpecification;
use crate::emit::execution::ExecutionModel;
use crate::emit::pattern::{self, Addend};
use crate::emit::request::KernelRequest;
pub use crate::emit::request::TensorArgument;
use crate::{CudaTargetCapability, DType, TargetCapability, TensorAccess};

mod render;

pub(in crate::emit) struct QuackKernelProvider;

/// Entry points in quack.gemm_interface. Quack owns device-code generation and
/// its internal launch configuration; Trinity supplies the requested operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuackApi {
    Gemm,
    GemmAdd,
    GemmAct,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuackSpecification {
    target: CudaTargetCapability,
    api: QuackApi,
    lhs: TensorArgument,
    rhs: TensorArgument,
    output: TensorArgument,
    residual: Option<TensorArgument>,
    bias: Option<TensorArgument>,
}

impl QuackSpecification {
    pub fn api(&self) -> QuackApi {
        self.api
    }
    pub fn target(&self) -> CudaTargetCapability {
        self.target
    }
    pub fn lhs(&self) -> &TensorArgument {
        &self.lhs
    }
    pub fn rhs(&self) -> &TensorArgument {
        &self.rhs
    }
    pub fn output(&self) -> &TensorArgument {
        &self.output
    }
    pub fn residual(&self) -> Option<&TensorArgument> {
        self.residual.as_ref()
    }
    pub fn bias(&self) -> Option<&TensorArgument> {
        self.bias.as_ref()
    }

    /// A standalone Python `run(values, *, tuned=False)` wrapper. `values` maps
    /// canonical PhysicalPlan ValueInstanceId indices to caller-owned tensors.
    /// Quack is imported lazily; generation requires neither Python nor CUDA.
    pub fn emit_python(&self) -> String {
        render::render(self)
    }
}

fn unsupported(reason: &str) -> ProviderError {
    ProviderError::Unsupported(reason.into())
}

impl KernelProvider for QuackKernelProvider {
    fn name(&self) -> &str {
        "quack"
    }

    fn candidates(
        &self,
        context: &KernelContext<'_, '_>,
    ) -> Result<Vec<CandidateSpecification>, ProviderError> {
        if context.execution != ExecutionModel::CudaStreamed
            || context.prepared.plan.world_size() != 1
        {
            return Err(unsupported(
                "Quack host calls require single-GPU streamed execution",
            ));
        }
        let TargetCapability::Cuda(target) = context.prepared.plan.target();
        if !matches!(
            target,
            CudaTargetCapability::Hopper | CudaTargetCapability::Sm120
        ) {
            return Err(unsupported(
                "Quack adapter supports Hopper and SM120 targets",
            ));
        }
        if !context.loops.is_empty() {
            return Err(unsupported(
                "Quack adapter requires a whole-matrix operation outside loops; loop/view coverage is not implemented",
            ));
        }
        let request = KernelRequest::from_context(context)?;
        let operation = context
            .prepared
            .plan
            .operation(context.operation)
            .ok_or_else(|| ProviderError::Failed("missing operation".into()))?;
        let pattern = pattern::gemm_epilogue(operation.expression()).ok_or_else(|| {
            unsupported("expected one store of [relu](A @ B [+ residual or column bias])")
        })?;
        let lhs = argument(&request, pattern.lhs, 2)?;
        let rhs = argument(&request, pattern.rhs, 2)?;
        let output = argument(&request, pattern.output, 2)?;
        if request.tensors.values().any(|v| v.dtype == DType::Fp16) {
            return Err(unsupported(
                "Quack adapter currently supports BF16/FP32 storage",
            ));
        }
        if lhs.dtype != DType::Bf16 || rhs.dtype != DType::Bf16 {
            return Err(unsupported(
                "Quack adapter currently requires BF16 matrix inputs",
            ));
        }
        if lhs.shape[1] != rhs.shape[0] || output.shape != [lhs.shape[0], rhs.shape[1]] {
            return Err(unsupported("GEMM shapes do not agree"));
        }
        if !lhs.shape[1].is_multiple_of(8) || !rhs.shape[1].is_multiple_of(8) {
            return Err(unsupported(
                "Quack adapter requires 16-byte aligned contiguous row strides (K and N multiples of 8)",
            ));
        }
        let (residual, bias) = match pattern.addend {
            None => (None, None),
            Some(Addend::Residual(access)) => {
                let residual = argument(&request, access, 2)?;
                if residual.shape != output.shape {
                    return Err(unsupported("residual must have the output matrix shape"));
                }
                (Some(residual), None)
            }
            Some(Addend::ColumnBias(access)) => {
                let bias = argument(&request, access, 1)?;
                if bias.shape != [rhs.shape[1]] {
                    return Err(unsupported("column bias must have shape [N]"));
                }
                (None, Some(bias))
            }
        };
        if [&lhs, &rhs]
            .into_iter()
            .chain(residual.as_ref())
            .chain(bias.as_ref())
            .any(|input| input.value == output.value)
        {
            return Err(unsupported(
                "Quack adapter does not support output aliasing an input",
            ));
        }
        let api = if pattern.relu {
            QuackApi::GemmAct
        } else if residual.is_some() {
            QuackApi::GemmAdd
        } else {
            QuackApi::Gemm
        };
        Ok(vec![CandidateSpecification::Quack(QuackSpecification {
            target,
            api,
            lhs,
            rhs,
            output,
            residual,
            bias,
        })])
    }
}

fn argument(
    request: &KernelRequest,
    access: &TensorAccess,
    rank: usize,
) -> Result<TensorArgument, ProviderError> {
    let value = request
        .tensors
        .get(&access.value)
        .ok_or_else(|| ProviderError::Failed("missing operand".into()))?;
    if access.shape(&value.shape) != value.shape {
        return Err(unsupported(
            "Quack host adapter does not yet bind per-access views",
        ));
    }
    if value.shape.len() != rank {
        return Err(unsupported(
            "Quack adapter requires operands of the expected rank",
        ));
    }
    Ok(value.clone())
}
