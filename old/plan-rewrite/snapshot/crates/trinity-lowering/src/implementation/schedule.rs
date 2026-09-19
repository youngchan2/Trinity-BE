//! Schedule construction belongs to lowering, independently of source emission.

use crate::{Expression, IndexExpr, LoopDomain, LoopKind};
use thiserror::Error;

#[derive(Debug, Clone, Default)]
pub struct OperationSchedule {
    pub dimensions: Vec<(LoopKind, LoopDomain)>,
    pub expression: Option<Expression>,
    pub coordinates: Vec<IndexExpr>,
}

#[derive(Debug, Error)]
pub enum ScheduleError {
    #[error("schedule does not support {0}")]
    Unsupported(String),
    #[error("invalid schedule contract: {0}")]
    Contract(String),
}
