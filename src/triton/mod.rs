//! Triton fallback for scheduled Trinity IR, including named views and split loops.
//! Analysis supplies the original IR plus shared view, access and dataflow facts;
//! Triton lowering plans padded blocks, local storage and launches before codegen
//! writes the Python module. The Triton provider adapts supported PhysicalPlan
//! operation requests to this pipeline, retaining typed storage and dot operands.
mod codegen;
mod lowering;
mod plan;
mod shape;

use std::collections::BTreeMap;

use crate::analysis::{AnalysisError, ScheduledIr, analyze_text};
pub use lowering::lower;
pub use plan::{
    InitialValue, Initialization, KernelPlan, LocalRead, ProgramMetadata, Storage, TensorPlan,
    TritonPlan,
};
pub use shape::{AxisAccess, TileAccess};

/// Storage dtype supplied by a typed frontend. Legacy TileLang defaults to FP16.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TensorDType {
    #[default]
    Fp16,
    Bf16,
    Fp32,
}

impl TensorDType {
    pub(crate) fn python(self) -> &'static str {
        match self {
            Self::Fp16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::Fp32 => "float32",
        }
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "FP16",
            Self::Bf16 => "BF16",
            Self::Fp32 => "FP32",
        }
    }
}

impl From<crate::DType> for TensorDType {
    fn from(value: crate::DType) -> Self {
        match value {
            crate::DType::Fp16 => Self::Fp16,
            crate::DType::Bf16 => Self::Bf16,
            crate::DType::Fp32 => Self::Fp32,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Options {
    /// Per-tensor storage types. An absent entry preserves the FP16 source-IR ABI.
    pub dtypes: BTreeMap<String, TensorDType>,
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
    #[error(transparent)]
    Resolution(#[from] crate::analysis::ResolveError),
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

pub fn compile_analysis(analysis: ScheduledIr, options: Options) -> Result<String, Error> {
    Ok(lower(analysis, options)?.emit())
}
