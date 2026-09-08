//! Analysis of an extracted, scheduled Trinity program.
//!
//! The first pass records ordered accesses and their exact lexical scopes. Tensor
//! sets are summaries of those records, not storage or initialization decisions.
//! [`analyze`] accepts a syntax tree; [`analyze_text`] is the compatibility entry
//! point for the existing evaluation corpus. Neither pass changes the schedule.

mod collect;
mod ir;
mod model;

pub use collect::{AnalysisError, analyze, analyze_text};
pub use ir::{IrNode, ParseError, SourceSpan};
pub use model::{
    AccessId, AccessInfo, AccessKind, IndexDim, IndexExpr, KernelId, KernelInfo, LoopInfo,
    ProgramAnalysis, ReadWrites, ScopeId, ScopeInfo, ScopeItem, ScopeKind, StatementId,
    StatementInfo, TensorId, TensorInfo, TensorKind, ValueExpr,
};
