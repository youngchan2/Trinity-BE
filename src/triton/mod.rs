//! Triton fallback for scheduled Trinity IR, including named views and split loops.
//! Analysis supplies the access and scope facts; lowering plans storage, indexing,
//! and launches before codegen writes the Python module.
mod codegen;
mod lowering;
mod plan;
mod shape;

use std::collections::BTreeMap;

use crate::analysis::{AnalysisError, ProgramAnalysis, analyze_text};
pub use lowering::lower;
pub use plan::{
    InitialValue, Initialization, KernelPlan, LocalRead, ProgramMetadata, ProgramPlan, Storage,
    TensorPlan,
};
pub use shape::{AxisAccess, TileAccess};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Options {
    /// Concrete tensor shapes; managed mode infers intermediate shapes from views.
    /// Emitted accesses use the kernel arguments' strides.
    pub shapes: BTreeMap<String, Vec<usize>>,
    /// Concrete IR symbol values used for validation and default tile sizes.
    /// Managed code emits tunable symbols as META_* parameters.
    pub symbols: BTreeMap<String, i64>,
    /// Allocate internal tensors and derive symbolic dimensions from input views.
    /// Automatically enabled for programs containing mloop.
    pub managed: bool,
    /// Optional profile candidates keyed by the original IR parameter name.
    pub tuning: BTreeMap<String, Vec<i64>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Analysis(#[from] AnalysisError),
    #[error("Triton lowering: {0}")]
    Invalid(String),
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// Generate Triton `kernel_N` functions and a named `forward(...)` launch wrapper.
/// Managed mode allocates intermediates and returns outputs; the legacy mode
/// accepts caller-owned tensors and also emits benchmark parameter lists.
pub fn compile(text: &str, options: Options) -> Result<String, Error> {
    let plan = lower(analyze_text(text)?, options)?;
    Ok(plan.emit())
}

pub fn compile_analysis(analysis: ProgramAnalysis, options: Options) -> Result<String, Error> {
    Ok(lower(analysis, options)?.emit())
}
