//! Fp16 Triton fallback using Trinity/backend/codegen's source conventions and
//! benchmark ABI. Rust analysis supplies storage and initialization decisions.
mod emit;
mod plan;
mod shape;

use std::collections::BTreeMap;

use crate::analyzer::{AnalysisError, ProgramAnalysis, analyze_text};
pub use plan::{InitialValue, Initialization, KernelPlan, Storage, TensorPlan, TritonPlan, lower};
pub use shape::{AxisAccess, TileAccess};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Options {
    /// Concrete tensor shapes, including intermediates; emitted accesses use strides.
    pub shapes: BTreeMap<String, Vec<usize>>,
    /// Concrete IR symbols used for validation (e.g. tile_k). Symbolic loop steps
    /// are emitted as BLOCK_* parameters with the reference backend's autotune.
    pub symbols: BTreeMap<String, i64>,
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

/// Generate a Python module exposing the original named `forward(...)` ABI,
/// `TENSOR_PARAMS`, `BLOCK_PARAMS`, and autotuned `kernel_N` functions.
pub fn compile(text: &str, options: Options) -> Result<String, Error> {
    let plan = lower(analyze_text(text)?, options)?;
    Ok(plan.emit())
}

pub fn compile_analysis(analysis: ProgramAnalysis, options: Options) -> Result<String, Error> {
    Ok(lower(analysis, options)?.emit())
}
