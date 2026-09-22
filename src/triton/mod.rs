//! Triton fallback for scheduled Trinity IR, including named views and split loops.
//! Both source IR and operation candidates enter the common PhysicalPlan before
//! TritonKernelProvider plans padded blocks, local storage, precision and launches.
//! Codegen writes the kernels and their ordered Python launch wrapper.
mod codegen;
pub(crate) mod lowering;
mod plan;
mod shape;

use std::collections::BTreeMap;

use crate::analysis::{AnalysisError, ScheduledIr, analyze_text};
pub use lowering::lower;
pub use plan::{
    InitialValue, Initialization, KernelPlan, LocalRead, ProgramMetadata, Storage, TensorPlan,
    TritonPlan, TuningConfig,
};
pub use shape::{AxisAccess, TileAccess};

/// Storage dtype supplied by a typed frontend. Legacy TileLang defaults to FP16.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
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

impl From<TensorDType> for crate::DType {
    fn from(value: TensorDType) -> Self {
        match value {
            TensorDType::Fp16 => Self::Fp16,
            TensorDType::Bf16 => Self::Bf16,
            TensorDType::Fp32 => Self::Fp32,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    /// Default logical/storage dtype for untyped source IR. Defaults to FP16.
    /// FP32 register computation does not change this contract.
    pub default_dtype: TensorDType,
    /// Explicit logical/storage contracts, overriding default and propagated types.
    pub dtypes: BTreeMap<String, TensorDType>,
    /// Concrete tensor shapes; intermediate shapes are inferred from views.
    /// Emitted accesses use the kernel arguments' strides.
    pub shapes: BTreeMap<String, Vec<usize>>,
    /// Concrete IR symbol values used for validation and default tile sizes.
    /// Tunable symbols are emitted as META_* parameters.
    pub symbols: BTreeMap<String, i64>,
    /// Bounded search policy for kernel launches.
    pub autotune: AutotuneOptions,
    /// Explicit tile candidates keyed by the original IR parameter name.
    /// Unspecified symbolic loop steps get a shape-validated default search.
    pub tuning: BTreeMap<String, Vec<i64>>,
}

/// Compilation budget and hardware choices, independent of IR tile symbols.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AutotuneOptions {
    /// Maximum benchmarked configurations per kernel. One uses the baseline.
    pub max_configs: usize,
    pub num_warps: Vec<u32>,
    pub num_stages: Vec<u32>,
}

impl Default for AutotuneOptions {
    fn default() -> Self {
        Self {
            max_configs: 64,
            num_warps: vec![4, 8, 2],
            num_stages: vec![1, 2, 3],
        }
    }
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
/// The wrapper allocates intermediates and returns outputs. Output buffers may
/// also be supplied by the caller without changing kernel lowering.
pub fn compile(text: &str, options: Options) -> Result<String, Error> {
    let plan = lower(analyze_text(text)?, options)?;
    Ok(plan.emit())
}

pub fn compile_analysis(analysis: ScheduledIr, options: Options) -> Result<String, Error> {
    Ok(lower(analysis, options)?.emit())
}
