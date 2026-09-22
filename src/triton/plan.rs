//! Immutable contract built by lowering and consumed by Triton code generation.
use super::{Options, TileAccess};
use crate::analysis::*;
use std::collections::{BTreeMap, BTreeSet};

pub use crate::analysis::storage::{
    AccessMode as Storage, InitialValue, Initialization, LocalRead, TensorStoragePlan as TensorPlan,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    pub root_scope: ScopeId,
    pub parallel_loops: Vec<ScopeId>,
    /// Grid at the validation/sample bindings supplied in Options.
    pub grid: Vec<usize>,
    /// Actual grid expressions, evaluated at launch with shape/tuning parameters.
    pub grid_extents: Vec<IndexExpr>,
    pub tensors: BTreeMap<TensorId, TensorPlan>,
    /// Per-occurrence binding: an accumulator can be local while later sibling
    /// loops read the same tensor through global memory.
    pub register_accesses: BTreeSet<AccessId>,
    /// A local read refers to an ordered definition and may select a sub-tile.
    pub local_reads: BTreeMap<AccessId, LocalRead>,
}

/// Triton/Python naming and tuning decisions. Logical tensor metadata belongs
/// to ProgramFacts, available through TritonPlan::common.
#[derive(Debug, Clone, Default)]
pub struct ProgramMetadata {
    pub tensor_names: Vec<String>,
    pub candidates: BTreeMap<String, Vec<i64>>,
    /// The one launch responsible for tuning each split parameter.
    pub split_owners: BTreeMap<String, ScopeId>,
}

/// One selected IR program, including all kernels and its Python wrapper.
/// Kernel-local decisions live in each KernelPlan; syntax and options are shared.
#[derive(Debug, Clone)]
pub struct TritonPlan {
    pub(crate) analysis: ScheduledIr,
    pub(crate) common: ProgramFacts,
    pub(crate) options: Options,
    pub(crate) kernels: Vec<KernelPlan>,
    pub(crate) accesses: Vec<TileAccess>,
    pub(crate) expressions: Vec<Expr>,
    pub(crate) globals: BTreeSet<TensorId>,
    pub(crate) metadata: ProgramMetadata,
}

impl TritonPlan {
    pub(crate) fn tensor_dtype(&self, tensor: TensorId) -> super::TensorDType {
        self.options
            .dtypes
            .get(&self.analysis.tensor(tensor).name)
            .copied()
            .unwrap_or_default()
    }
    pub fn analysis(&self) -> &ScheduledIr {
        &self.analysis
    }
    /// Shared logical accesses and dataflow, before Triton padding or storage decisions.
    pub fn common(&self) -> &ProgramFacts {
        &self.common
    }
    pub fn kernels(&self) -> &[KernelPlan] {
        &self.kernels
    }
    pub fn accesses(&self) -> &[TileAccess] {
        &self.accesses
    }
    pub fn options(&self) -> &Options {
        &self.options
    }
    pub fn metadata(&self) -> &ProgramMetadata {
        &self.metadata
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Expr {
    pub kind: ExprKind,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) enum ExprKind {
    Scalar(String),
    Index(IndexExpr),
    Load(AccessId),
    Unary(String, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Reduce(String, usize, Box<Expr>),
    Permute(Vec<usize>, Box<Expr>),
    Transform(String, usize, Box<Expr>),
    Dot(Box<Expr>, Box<Expr>),
    Cast(String, Box<Expr>),
    Concat(usize, Box<Expr>, Box<Expr>),
}
