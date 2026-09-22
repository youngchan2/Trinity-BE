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
    /// ABI output order from the common plan; independent of canonical value IDs.
    pub output_order: Vec<TensorId>,
    pub tensor_names: Vec<String>,
    pub candidates: BTreeMap<String, Vec<i64>>,
    /// The one launch responsible for tuning each split parameter.
    pub split_owners: BTreeMap<String, ScopeId>,
}

/// A validated tile assignment and Triton compilation configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningConfig {
    pub parameters: BTreeMap<String, i64>,
    pub num_warps: u32,
    pub num_stages: u32,
}

/// One selected IR program, including all kernels and its Python wrapper.
/// Kernel-local decisions live in each KernelPlan; syntax and options are shared.
#[derive(Debug, Clone)]
pub struct TritonPlan {
    pub(crate) storage_contracts: BTreeMap<TensorId, crate::Storage>,
    pub(crate) analysis: ScheduledIr,
    pub(crate) common: ProgramFacts,
    pub(crate) options: Options,
    pub(crate) kernels: Vec<KernelPlan>,
    pub(crate) accesses: Vec<TileAccess>,
    pub(crate) expressions: Vec<Expr>,
    pub(crate) dtypes: Vec<super::TensorDType>,
    pub(crate) globals: BTreeSet<TensorId>,
    pub(crate) metadata: ProgramMetadata,
    pub(crate) tuning: Vec<Vec<TuningConfig>>,
}

impl TritonPlan {
    pub fn tuning_configs(&self, kernel: usize) -> &[TuningConfig] {
        &self.tuning[kernel]
    }
    /// Resolved logical/storage type, independent of FP32 register computation.
    pub fn tensor_dtype(&self, tensor: TensorId) -> super::TensorDType {
        self.dtypes[tensor.index()]
    }
    pub fn analysis(&self) -> &ScheduledIr {
        &self.analysis
    }
    /// Shared logical accesses and dataflow, before padding and provider lowering.
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

impl TritonPlan {
    pub(crate) fn canonical_symbol<'a>(&'a self, name: &'a str) -> &'a str {
        self.common
            .metadata
            .aliases
            .get(name)
            .map(String::as_str)
            .unwrap_or(name)
    }
    pub(crate) fn parameters(&self, kernel: &KernelPlan) -> BTreeSet<String> {
        let mut used = BTreeSet::new();
        let ki = self.analysis.scope(kernel.root_scope).kernel;
        for scope in self.analysis.scopes().iter().filter(|s| s.kernel == ki) {
            if let Some(info) = &scope.loop_info {
                for expr in [&info.start, &info.end, &info.step] {
                    used.extend(crate::analysis::scalar::symbols(expr));
                }
            }
        }
        for access in &self.analysis.kernel(ki).accesses {
            let a = self.analysis.access(*access);
            if let Some(shape) = &a.view_shape {
                for expr in shape {
                    used.extend(crate::analysis::scalar::symbols(expr));
                }
            }
            for axis in &self.accesses[access.index()].axes {
                used.extend(crate::analysis::scalar::symbols(&axis.start));
            }
        }
        used.into_iter()
            .map(|s| self.canonical_symbol(&s).to_owned())
            .filter(|s| {
                self.common.metadata.dimensions.contains_key(s)
                    || self.metadata.candidates.contains_key(s)
            })
            .collect()
    }
    pub(crate) fn owned_parameters(&self, kernel: &KernelPlan) -> BTreeSet<String> {
        let ki = self.analysis.scope(kernel.root_scope).kernel;
        self.parameters(kernel)
            .into_iter()
            .filter(|s| {
                self.metadata.candidates.contains_key(s)
                    && self
                        .metadata
                        .split_owners
                        .get(s)
                        .is_none_or(|scope| self.analysis.scope(*scope).kernel == ki)
            })
            .collect()
    }
}
