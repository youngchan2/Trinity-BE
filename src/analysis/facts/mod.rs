//! Shared analysis results for one scheduled IR and one set of scalar bindings.
pub mod access;
pub(crate) mod dependencies;
pub mod dtype;
pub(super) mod flow;
pub mod loops;
pub mod metadata;
pub mod scalar;

use crate::analysis::{ScheduledIr, TensorMetadata};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Bindings {
    pub shapes: BTreeMap<String, Vec<usize>>,
    pub symbols: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("IR analysis: {0}")]
pub struct ResolveError(pub String);

pub(super) fn invalid(message: impl Into<String>) -> ResolveError {
    ResolveError(message.into())
}

/// IDs refer to the accompanying ScheduledIr. No backend decisions are stored.
/// This is a projection of the scheduled source, not a second physical schedule.
#[derive(Debug, Clone)]
pub struct ProgramFacts {
    pub metadata: TensorMetadata,
    pub accesses: Vec<access::ResolvedAccess>,
    pub kernels: Vec<super::KernelDataflow>,
}

impl ProgramFacts {
    /// Analyze fully specified input without applying any provider's defaults.
    pub fn analyze(ir: &ScheduledIr, bindings: &mut Bindings) -> Result<Self, ResolveError> {
        let metadata = TensorMetadata::collect(ir, bindings)?;
        Self::resolve(ir, bindings, metadata)
    }

    /// Finish shared analysis after the caller specializes any free IR parameters.
    /// `metadata` must have been collected from the same `ir` and bindings.
    pub fn resolve(
        ir: &ScheduledIr,
        bindings: &mut Bindings,
        mut metadata: TensorMetadata,
    ) -> Result<Self, ResolveError> {
        metadata.resolve_shapes(ir, bindings)?;
        for shape in bindings.shapes.values() {
            scalar::product(shape)?;
        }
        for (i, scope) in ir.scopes().iter().enumerate() {
            if scope.loop_info.is_some() {
                loops::validate(ir, super::ScopeId(i), bindings)?;
            }
        }
        let accesses = ir
            .accesses()
            .iter()
            .map(|a| access::resolve(a, ir, bindings))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            metadata,
            accesses,
            kernels: flow::analyze(ir),
        })
    }
}
