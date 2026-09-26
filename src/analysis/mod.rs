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

mod facts;
mod ir;
pub(crate) mod plan;
pub mod regions;
pub mod storage;
pub mod views;

// Preserve the analysis API while grouping its implementations by responsibility.
pub(crate) use facts::dependencies;
pub use facts::flow::{EntryValue, KernelDataflow, TensorDataflow};
use facts::invalid;
pub use facts::metadata::TensorMetadata;
pub use facts::{Bindings, ProgramFacts, ResolveError, access, dtype, loops, metadata, scalar};
pub use ir::ScopeItem;
pub use ir::{
    AccessId, AccessInfo, AccessKind, AnalysisError, IndexDim, IndexExpr, IrNode, KernelId,
    KernelInfo, LoopInfo, ParseError, ReadWrites, ScheduledIr, ScopeId, ScopeInfo, ScopeKind,
    SourceSpan, StatementId, StatementInfo, TensorId, TensorInfo, TensorKind, ValueExpr, analyze,
    analyze_text, from_physical,
};
