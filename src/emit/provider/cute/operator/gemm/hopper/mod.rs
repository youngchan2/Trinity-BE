//! BF16 Hopper WGMMA with two shared-memory stages.

use crate::Storage;
use crate::emit::native::{KernelInterface, KernelPort, RegisterLayout};

use super::super::{
    access::{Access, Axis},
    accumulation, unsupported,
};
use crate::emit::native::{
    Kernel, KernelBindings, KernelImplementation, KernelRequirements, SpecifiedKernel, ThreadPolicy,
};
use crate::emit::provider::{KernelContext, ProviderError};
use crate::{CudaTargetCapability, DType, Expression, IndexExpr, TargetCapability};
use std::collections::BTreeMap;

mod render;
#[cfg(test)]
mod tests;

pub(in crate::emit::provider::cute) struct HopperGemmKernel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) struct HopperGemmSpecification {
    pub requirements: KernelRequirements,
    pub iteration: String,
    pub lhs: Access,
    pub rhs: Access,
    pub output: Access,
}

impl KernelImplementation for HopperGemmKernel {
    fn specify(&self, context: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError> {
        if context.prepared.plan.target() != TargetCapability::Cuda(CudaTargetCapability::Hopper) {
            return Err(unsupported("Hopper GEMM requires an sm_90a target"));
        }
        let (loop_, rhs) = accumulation(context)?;

        let Expression::Matmul(operands) = rhs else {
            return Err(unsupported("Hopper GEMM requires a matmul accumulation"));
        };

        let [Expression::Load(lhs), Expression::Load(rhs)] = operands.as_ref() else {
            return Err(unsupported("Hopper GEMM requires two tensor loads"));
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
                "Hopper GEMM requires memory inputs and memory or Register output",
            ));
        }
        if lhs.dtype != DType::Bf16
            || rhs.dtype != DType::Bf16
            || lhs.shape.len() != 2
            || rhs.shape.len() != 2
            || output.shape.len() != 2
        {
            return Err(unsupported(
                "Hopper GEMM requires rank-2 BF16 inputs and a rank-2 output",
            ));
        }

        if lhs.shape[1] != rhs.shape[0]
            || output.shape != [lhs.shape[0], rhs.shape[1]]
            || lhs.axes[0] != output.axes[0]
            || rhs.axes[1] != output.axes[1]
            || lhs.axes[1] != rhs.axes[0]
        {
            return Err(unsupported(
                "Hopper GEMM matrix shapes or accesses do not match",
            ));
        }

        let Axis::Tile {
            variable,
            width,
            clipped: false,
        } = &lhs.axes[1]
        else {
            return Err(unsupported("Hopper GEMM requires an unclipped K tile"));
        };

        if *variable != loop_.domain.variable
            || !width.is_multiple_of(64)
            || !(1..=128).contains(&lhs.width(0))
            || rhs.width(1) != 128
            || matches!(rhs.axes[1], Axis::Tile { clipped: true, .. })
        {
            return Err(unsupported(
                "Hopper GEMM requires M<=128, N=128 and K tiles divisible by 64",
            ));
        }

        if !lhs.shape[1].is_multiple_of(8)
            || !rhs.shape[1].is_multiple_of(8)
            || !aligned_axis(&lhs.axes[1], context)
            || !aligned_axis(&rhs.axes[1], context)
        {
            return Err(unsupported(
                "Hopper GEMM vector copies require 16-byte aligned strides and origins",
            ));
        }

        let mut alignments = vec![(lhs.value, 16)];
        if rhs.value != lhs.value {
            alignments.push((rhs.value, 16));
        }

        if !alignments.iter().any(|(id, _)| *id == output.value) {
            alignments.push((output.value, output.dtype.size_bytes()));
        }

        Ok(SpecifiedKernel::CuTeHopperGemm(HopperGemmSpecification {
            iteration: loop_.domain.variable.clone(),
            requirements: KernelRequirements {
                shared_memory_bytes: 65536,
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
        let SpecifiedKernel::CuTeHopperGemm(specification) = specification else {
            return Err(ProviderError::Failed(
                "expected Hopper GEMM specification".into(),
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

impl HopperGemmSpecification {
    pub(in crate::emit) fn interface(&self) -> KernelInterface {
        KernelInterface {
            inputs: vec![KernelPort::memory(&self.lhs), KernelPort::memory(&self.rhs)],
            outputs: vec![KernelPort {
                access: self.output.clone(),
                register: Some(RegisterLayout::Fixed {
                    representation: "cuda.scalar",
                    distribution: "cute.wgmma.128x128".into(),
                }),
                projection: (0..self.output.axes.len()).collect(),
            }],
            iteration: Some(self.iteration.clone()),
        }
    }
}
