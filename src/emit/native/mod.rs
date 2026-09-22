//! Contracts for composing implementations inside one native device body.
//! Opaque providers do not implement these phase, layout or binding contracts.

use super::provider::cute::{
    HopperGemmSpecification, PointwiseSpecification, ReduceSumSpecification, Sm80GemmSpecification,
};
use super::provider::{KernelContext, KernelProvider, ProviderError};
use crate::ValueInstanceId;
use std::collections::BTreeMap;

pub(super) mod access;
pub(super) mod bindings;
mod code;
mod interface;
pub(super) use code::{KernelCode, RegisterBinding, render_index_expression};
pub(super) use interface::{KernelInterface, KernelPort, RegisterLayout};

/// Optional native composition capability, separate from candidate discovery.
pub(super) trait NativeKernelProvider: KernelProvider {
    fn specify(&self, context: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError>;
    fn render(
        &self,
        specification: &SpecifiedKernel,
        bindings: &KernelBindings,
    ) -> Result<Kernel, ProviderError>;
}

/// Specification and rendering for one implementation path within a provider.
pub(super) trait KernelImplementation {
    /// Returns one specification if this implementation supports the operation.
    fn specify(&self, context: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError>;

    /// Renders this implementation's specification using the resolved bindings.
    fn render(
        &self,
        specification: &SpecifiedKernel,
        bindings: &KernelBindings,
    ) -> Result<Kernel, ProviderError>;
}

/// An implementation specification ready for rendering without consulting the Plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SpecifiedKernel {
    CuTePointwise(PointwiseSpecification),
    CuTeHopperGemm(HopperGemmSpecification),
    CuTeSm80Gemm(Sm80GemmSpecification),
    CuTeReduceSum(ReduceSumSpecification),
}

impl SpecifiedKernel {
    /// Resolves thread-dependent layouts using the enclosing body's CTA size.
    pub fn interface(&self, block_threads: usize) -> KernelInterface {
        match self {
            Self::CuTePointwise(s) => s.interface(block_threads),
            Self::CuTeHopperGemm(s) => s.interface(),
            Self::CuTeSm80Gemm(s) => s.interface(),
            Self::CuTeReduceSum(s) => s.interface(),
        }
    }

    /// The sequential loop required by this implementation, independent of CTA size.
    pub fn iteration(&self) -> Option<&str> {
        match self {
            Self::CuTePointwise(_) => None,
            Self::CuTeHopperGemm(s) => Some(&s.iteration),
            Self::CuTeSm80Gemm(s) => Some(&s.iteration),
            Self::CuTeReduceSum(s) => Some(&s.iteration),
        }
    }

    pub fn requirements(&self) -> &KernelRequirements {
        match self {
            Self::CuTePointwise(specification) => &specification.requirements,
            Self::CuTeHopperGemm(specification) => &specification.requirements,
            Self::CuTeSm80Gemm(specification) => &specification.requirements,
            Self::CuTeReduceSum(specification) => &specification.requirements,
        }
    }

    #[cfg(test)]
    pub fn requirements_mut(&mut self) -> &mut KernelRequirements {
        match self {
            Self::CuTePointwise(s) => &mut s.requirements,
            Self::CuTeHopperGemm(s) => &mut s.requirements,
            Self::CuTeSm80Gemm(s) => &mut s.requirements,
            Self::CuTeReduceSum(s) => &mut s.requirements,
        }
    }
}

/// Thread participation promised by a provider's generated code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ThreadPolicy {
    /// Exactly this many threads enter all phases; CTA barriers are allowed.
    FullCta { threads: usize },
    /// All CTA threads participate at any of these sizes. The list expresses
    /// support, not preference. Emit chooses a size after forming the body.
    /// The provider derives layouts and code from that resolved size; its other
    /// requirements must hold for every supported size.
    Flexible { supported: &'static [usize] },
    /// Only the first `threads` threads participate, in complete warps.
    /// Code uses their zero-based threadIdx.x and must not use CTA-wide barriers.
    /// Synchronization must stay within those participating threads.
    Subgroup { threads: usize },
    /// Scalar continuation in its Register producer's output scope. Thread and
    /// element ownership are inherited; no independent collective is allowed.
    FollowInput,
}

/// Implementation requirements exposed before code generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct KernelRequirements {
    pub thread_policy: ThreadPolicy,
    pub shared_memory_bytes: usize,
    /// Minimum alignment of the supplied implementation scratch.
    pub shared_memory_alignment: usize,
    pub alignments: Vec<(ValueInstanceId, usize)>,
}

impl ThreadPolicy {
    pub fn accepts_block_threads(self, block_threads: usize) -> bool {
        match self {
            Self::FullCta { threads } => block_threads == threads,
            Self::Flexible { supported } => supported.contains(&block_threads),
            Self::Subgroup { threads } => block_threads >= threads,
            Self::FollowInput => true,
        }
    }
}

/// Resolved CTA size and memory, register, and index expressions for code generation.
#[derive(Clone)]
pub(super) struct KernelBindings {
    /// The enclosing body's final CTA size, selected after fusion.
    pub block_threads: usize,
    pub values: BTreeMap<ValueInstanceId, String>,
    /// Input port index to the register element available in the enclosing output scope.
    pub registers: BTreeMap<usize, RegisterBinding>,
    pub indices: BTreeMap<String, String>,
    /// Aligned implementation scratch, separate from tensor storage.
    pub shared_memory: Option<String>,
    /// Identifier prefix unique to this rendered body.
    pub prefix: String,
}

/// Rendered native phases sharing one device scope. Opaque host calls cannot
/// enter this contract. Prologue/epilogue enclose the specified reduction loop.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Kernel {
    pub includes: Vec<&'static str>,
    pub prologue: KernelCode,
    pub mainloop: Option<KernelCode>,
    pub epilogue: KernelCode,
}
