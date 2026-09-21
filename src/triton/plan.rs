//! Immutable contract built by lowering and consumed by Triton code generation.
use super::{Options, TileAccess};
use crate::analysis::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Register,
    Global,
    /// Cross-sloop tensor supplied by the caller using its declared storage dtype.
    Materialized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialValue {
    Zero,
    Global,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Initialization {
    pub scope: ScopeId,
    pub access: AccessId,
    pub value: InitialValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlan {
    pub storage: Storage,
    pub representative: AccessId,
    pub initialization: Option<Initialization>,
    /// A register's final store executes at the end of this scope.
    pub export_scope: Option<ScopeId>,
    /// Value is visible outside this kernel.
    pub publish: bool,
    /// Additive recurrences, separately from ordinary assignments/epilogues.
    pub accumulators: BTreeSet<StatementId>,
}

impl TensorPlan {
    pub(crate) fn has_global(&self) -> bool {
        self.storage != Storage::Register
            || self.publish
            || self
                .initialization
                .as_ref()
                .is_some_and(|i| i.value == InitialValue::Global)
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRead {
    pub definition: AccessId,
    /// Corresponding storage axes, excluding statically singleton view axes.
    pub axes: Vec<(usize, usize)>,
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
