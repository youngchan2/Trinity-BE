use std::collections::BTreeSet;

use super::{IrNode, SourceSpan};

macro_rules! ids {
    ($($name:ident),+ $(,)?) => {$(
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub(crate) usize);

        impl $name {
            pub fn index(self) -> usize { self.0 }
        }
    )+};
}

ids!(TensorId, KernelId, ScopeId, StatementId, AccessId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorKind {
    Input,
    Output,
    Intermediate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// All declarations observed in the IR, independently of actual access mode.
    pub declarations: BTreeSet<TensorKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

/// Scalar index syntax with lexical loop bindings, without algebraic guessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexExpr {
    Integer(i64),
    Symbol(String),
    LoopVar(ScopeId),
    Apply(String, Vec<IndexExpr>),
}

impl IndexExpr {
    pub fn loop_dependencies(&self) -> BTreeSet<ScopeId> {
        match self {
            Self::LoopVar(id) => BTreeSet::from([*id]),
            Self::Apply(_, args) => args.iter().flat_map(Self::loop_dependencies).collect(),
            _ => BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexDim {
    FullTile,
    Tile { start: IndexExpr, width: IndexExpr },
    Elem(IndexExpr),
    ConstTile { start: IndexExpr, width: IndexExpr },
}

impl IndexDim {
    pub fn loop_dependencies(&self) -> BTreeSet<ScopeId> {
        match self {
            Self::FullTile => BTreeSet::new(),
            Self::Elem(index) => index.loop_dependencies(),
            Self::Tile { start, width } | Self::ConstTile { start, width } => start
                .loop_dependencies()
                .union(&width.loop_dependencies())
                .copied()
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessInfo {
    pub tensor: TensorId,
    pub kind: AccessKind,
    pub statement: StatementId,
    /// The one scope containing this occurrence, not every ancestor loop.
    pub scope: ScopeId,
    pub index: Vec<IndexDim>,
    /// Per-access contiguous view dimensions, in layout order.
    pub view_shape: Option<Vec<IndexExpr>>,
    /// Named axes in layout order. Names are labels, never storage identities.
    pub view_axes: Option<Vec<String>>,
    pub source_span: Option<SourceSpan>,
}

/// Convenience tensor sets derived while recording full AccessInfo entries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReadWrites {
    pub reads: BTreeSet<TensorId>,
    pub writes: BTreeSet<TensorId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Kernel,
    ParallelLoop,
    SequentialLoop,
    /// The parallel binding of an mloop. Its child serial scope binds n_var.
    SplitLoop,
}

impl ScopeKind {
    pub fn is_parallel(self) -> bool {
        matches!(self, Self::ParallelLoop | Self::SplitLoop)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopInfo {
    pub variable: String,
    pub start: IndexExpr,
    pub end: IndexExpr,
    pub step: IndexExpr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeItem {
    Scope(ScopeId),
    Statement(StatementId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeInfo {
    pub kernel: KernelId,
    pub parent: Option<ScopeId>,
    pub kind: ScopeKind,
    pub loop_info: Option<LoopInfo>,
    pub source_span: Option<SourceSpan>,
    pub children: Vec<ScopeItem>,
    /// Direct accesses only. Descendant scopes are queried through the tree.
    pub accesses: Vec<AccessId>,
    pub read_writes: ReadWrites,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueExpr {
    Literal(String),
    Index(IndexExpr),
    Load(AccessId),
    Apply(String, Vec<ValueExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatementInfo {
    pub scope: ScopeId,
    pub source_span: Option<SourceSpan>,
    /// Component selected from a grouped store (zero for ordinary stores).
    pub group_index: usize,
    /// Group-projected expression whose loads refer to the access table.
    pub expression: ValueExpr,
    /// RHS loads in evaluation order, followed by the target write.
    pub accesses: Vec<AccessId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelInfo {
    pub root_scope: ScopeId,
    pub accesses: Vec<AccessId>,
    pub read_writes: ReadWrites,
}

/// Shared scope/access snapshot, collected from source or projected from a plan.
///
/// There are no emitter caches or mutable "currently generating" classifications.
/// Storage, liveness, initialization and precision plans are intentionally not
/// inferred by this first collection pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledIr {
    pub(super) ir: Option<IrNode>,
    pub(super) tensors: Vec<TensorInfo>,
    pub(super) kernels: Vec<KernelInfo>,
    pub(super) scopes: Vec<ScopeInfo>,
    pub(super) statements: Vec<StatementInfo>,
    pub(super) accesses: Vec<AccessInfo>,
}

impl ScheduledIr {
    /// Original syntax, when collected from source. Typed-plan projections have
    /// no syntax tree; code generation consumes the scope/access tables instead.
    pub fn ir(&self) -> Option<&IrNode> {
        self.ir.as_ref()
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn kernels(&self) -> &[KernelInfo] {
        &self.kernels
    }
    pub fn scopes(&self) -> &[ScopeInfo] {
        &self.scopes
    }
    pub fn statements(&self) -> &[StatementInfo] {
        &self.statements
    }
    pub fn accesses(&self) -> &[AccessInfo] {
        &self.accesses
    }
    pub fn tensor(&self, id: TensorId) -> &TensorInfo {
        &self.tensors[id.0]
    }
    pub fn kernel(&self, id: KernelId) -> &KernelInfo {
        &self.kernels[id.0]
    }
    pub fn scope(&self, id: ScopeId) -> &ScopeInfo {
        &self.scopes[id.0]
    }
    pub fn statement(&self, id: StatementId) -> &StatementInfo {
        &self.statements[id.0]
    }
    pub fn access(&self, id: AccessId) -> &AccessInfo {
        &self.accesses[id.0]
    }

    pub fn tensor_id(&self, name: &str) -> Option<TensorId> {
        self.tensors
            .iter()
            .position(|t| t.name == name)
            .map(TensorId)
    }

    pub fn declared_tensors(&self, kind: TensorKind) -> BTreeSet<TensorId> {
        self.tensors
            .iter()
            .enumerate()
            .filter_map(|(id, tensor)| tensor.declarations.contains(&kind).then_some(TensorId(id)))
            .collect()
    }

    pub fn mutated_inputs(&self) -> BTreeSet<TensorId> {
        let inputs = self.declared_tensors(TensorKind::Input);
        self.accesses
            .iter()
            .filter_map(|a| {
                (a.kind == AccessKind::Write && inputs.contains(&a.tensor)).then_some(a.tensor)
            })
            .collect()
    }

    /// Exact lexical scopes of occurrences; an ancestor is never counted again.
    pub fn access_scopes(&self, tensor: TensorId, kind: AccessKind) -> BTreeSet<ScopeId> {
        self.accesses
            .iter()
            .filter_map(|a| (a.tensor == tensor && a.kind == kind).then_some(a.scope))
            .collect()
    }

    pub fn is_within(&self, scope: ScopeId, ancestor: ScopeId) -> bool {
        let mut current = Some(scope);
        while let Some(id) = current {
            if id == ancestor {
                return true;
            }
            current = self.scope(id).parent;
        }
        false
    }
}
