//! BF16 warp MMA with two shared-memory stages on RTX targets.

use super::super::{
    access::{Access, Axis},
    accumulation, unsupported,
};
use crate::emit::native::{
    Kernel, KernelBindings, KernelImplementation, KernelInterface, KernelPort, KernelRequirements,
    RegisterLayout, SpecifiedKernel, ThreadPolicy,
};
use crate::emit::provider::{KernelContext, ProviderError};
use crate::{CudaTargetCapability, DType, Expression, IndexExpr, Storage, TargetCapability};
use std::collections::BTreeMap;

mod render;
#[cfg(test)]
mod tests;

pub(in crate::emit::provider::cute) struct Sm80GemmKernel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) struct Sm80GemmSpecification {
    pub requirements: KernelRequirements,
    pub iteration: String,
    pub lhs: Access,
    pub rhs: Access,
    pub output: Access,
}

impl KernelImplementation for Sm80GemmKernel {
    fn specify(&self, context: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError> {
        if !matches!(
            context.prepared.plan.target(),
            TargetCapability::Cuda(CudaTargetCapability::Sm89 | CudaTargetCapability::Sm120)
        ) {
            return Err(unsupported("SM80 GEMM requires an sm_89 or sm_120 target"));
        }
        let (loop_, rhs) = accumulation(context)?;

        let Expression::Matmul(operands) = rhs else {
            return Err(unsupported("SM80 GEMM requires a matmul accumulation"));
        };

        let [Expression::Load(lhs), Expression::Load(rhs)] = operands.as_ref() else {
            return Err(unsupported("SM80 GEMM requires two tensor loads"));
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

        let lhs = Access::from_plan(context, lhs)?;
        let rhs = Access::from_plan(context, rhs)?;
        let output = Access::from_plan(context, destination)?;

        if !matches!(lhs.storage, Storage::External | Storage::Global)
            || !matches!(rhs.storage, Storage::External | Storage::Global)
            || output.storage == Storage::Shared
        {
            return Err(unsupported(
                "SM80 GEMM requires memory inputs and memory or Register output",
            ));
        }
        if lhs.dtype != DType::Bf16
            || rhs.dtype != DType::Bf16
            || lhs.shape.len() != 2
            || rhs.shape.len() != 2
            || output.shape.len() != 2
        {
            return Err(unsupported(
                "SM80 GEMM requires rank-2 BF16 inputs and a rank-2 output",
            ));
        }

        if lhs.shape[1] != rhs.shape[0]
            || output.shape != [lhs.shape[0], rhs.shape[1]]
            || lhs.axes[0] != output.axes[0]
            || rhs.axes[1] != output.axes[1]
            || lhs.axes[1] != rhs.axes[0]
        {
            return Err(unsupported(
                "SM80 GEMM matrix shapes or accesses do not match",
            ));
        }

        let Axis::Tile {
            variable,
            width,
            clipped: false,
        } = &lhs.axes[1]
        else {
            return Err(unsupported("SM80 GEMM requires an unclipped K tile"));
        };

        if *variable != loop_.domain.variable
            || !width.is_multiple_of(32)
            || !(1..=128).contains(&lhs.width(0))
            || rhs.width(1) != 128
            || matches!(rhs.axes[1], Axis::Tile { clipped: true, .. })
        {
            return Err(unsupported(
                "SM80 GEMM requires M<=128, N=128 and K tiles divisible by 32",
            ));
        }

        if !lhs.shape[1].is_multiple_of(8)
            || !rhs.shape[1].is_multiple_of(8)
            || !aligned_axis(&lhs.axes[1], context)
            || !aligned_axis(&rhs.axes[1], context)
        {
            return Err(unsupported(
                "SM80 GEMM vector copies require 16-byte aligned strides and origins",
            ));
        }

        let mut alignments = vec![(lhs.value, 16)];
        if rhs.value != lhs.value {
            alignments.push((rhs.value, 16));
        }

        if !alignments.iter().any(|(id, _)| *id == output.value) {
            alignments.push((output.value, output.dtype.size_bytes()));
        }

        Ok(SpecifiedKernel::CuTeSm80Gemm(Sm80GemmSpecification {
            iteration: loop_.domain.variable.clone(),
            requirements: KernelRequirements {
                shared_memory_bytes: 32768,
                shared_memory_alignment: 128,
                thread_policy: ThreadPolicy::FullCta { threads: 128 },
                alignments,
            },
            lhs,
            rhs,
            output,
        }))
    }

    fn render(
        &self,
        specification: &SpecifiedKernel,
        bindings: &KernelBindings,
    ) -> Result<Kernel, ProviderError> {
        let SpecifiedKernel::CuTeSm80Gemm(specification) = specification else {
            return Err(ProviderError::Failed(
                "expected SM80 GEMM specification".into(),
            ));
        };

        render::render(specification, bindings)
    }
}

/// Prove vector-copy alignment without evaluating or expanding the loop range.
fn aligned_axis(axis: &Axis, context: &KernelContext<'_, '_>) -> bool {
    match axis {
        Axis::Full => true,
        Axis::Tile {
            variable,
            clipped: false,
            ..
        } => {
            let multiple = |e: &IndexExpr| e.evaluate(&BTreeMap::new()).is_ok_and(|n| n % 8 == 0);
            context
                .loops
                .iter()
                .find(|l| l.domain.variable == *variable)
                .is_some_and(|l| multiple(&l.domain.start) && multiple(&l.domain.step))
        }
        _ => false,
    }
}

impl Sm80GemmSpecification {
    pub(in crate::emit) fn interface(&self) -> KernelInterface {
        KernelInterface {
            inputs: vec![KernelPort::memory(&self.lhs), KernelPort::memory(&self.rhs)],
            outputs: vec![KernelPort {
                access: self.output.clone(),
                register: Some(RegisterLayout::Fixed {
                    representation: "cuda.scalar",
                    distribution: "cute.sm80.mma.128x128.warps2x2.tile32x32x16".into(),
                }),
                projection: (0..self.output.axes.len()).collect(),
            }],
            iteration: Some(self.iteration.clone()),
        }
    }
}
