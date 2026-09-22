//! Analysis of an extracted, scheduled Trinity program.
//!
//! The first pass records ordered accesses and their exact lexical scopes. Tensor
//! sets are summaries of those records, not storage or initialization decisions.
//! [`analyze`] accepts a syntax tree; [`analyze_text`] parses an IR source file.
//! Neither pass changes the schedule.
//!
//! [`TensorMetadata::collect`] and [`ProgramFacts::resolve`] add shared shape,
//! access and dataflow facts while retaining the source expressions in
//! [`ScheduledIr`]. They do not choose a provider, register representation,
//! padding, grid or initialization policy.
//!
//! [`storage`] consumes those facts to select common value storage, local read
//! bindings, recurrence initialization and publication points. Providers retain
//! the selected backing storage while choosing their implementation details.

pub mod access;
mod collect;
pub(crate) mod dependencies;
pub mod dtype;
mod facts;
mod flow;
mod ir;
pub mod loops;
pub mod metadata;
mod model;
mod physical;
pub use physical::from_physical;
pub mod scalar;
pub mod storage;

pub use collect::{AnalysisError, analyze, analyze_text};
use facts::invalid;
pub use facts::{Bindings, ProgramFacts, ResolveError};
pub use flow::{EntryValue, KernelDataflow, TensorDataflow};
pub use ir::{IrNode, ParseError, SourceSpan};
pub use metadata::TensorMetadata;
pub use model::{
    AccessId, AccessInfo, AccessKind, IndexDim, IndexExpr, KernelId, KernelInfo, LoopInfo,
    ReadWrites, ScheduledIr, ScopeId, ScopeInfo, ScopeItem, ScopeKind, StatementId, StatementInfo,
    TensorId, TensorInfo, TensorKind, ValueExpr,
};
