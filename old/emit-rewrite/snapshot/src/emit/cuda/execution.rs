//! Execution-specific placement metadata.
use super::{
    EmitError, Task,
    domain::TaskDomain,
    persistent::{self, PersistentExecution},
    streamed::{self, Grid, StreamedExecution},
};
use crate::PhysicalPlan;

pub(super) enum Placement {
    Streamed(Vec<Grid>),
    Persistent,
}

impl Placement {
    pub(super) fn new(plan: &PhysicalPlan, domains: &[TaskDomain]) -> Result<Self, EmitError> {
        if plan.world_size() == 1 {
            // Check grid arithmetic before body construction and coordinate validation.
            Ok(Self::Streamed(streamed::build_grids(domains)?))
        } else {
            Ok(Self::Persistent)
        }
    }

    pub(super) fn finish(
        self,
        plan: &PhysicalPlan,
        domains: Vec<TaskDomain>,
    ) -> Result<Execution, EmitError> {
        match self {
            Self::Streamed(grids) => streamed::build(domains, grids).map(Execution::Streamed),
            Self::Persistent => persistent::build(plan, &domains).map(Execution::Persistent),
        }
    }
}

/// Only the selected execution mode's metadata is constructed.
#[derive(Debug, Clone)]
pub enum Execution {
    Streamed(StreamedExecution),
    Persistent(PersistentExecution),
}

impl Execution {
    pub fn as_streamed(&self) -> Option<&StreamedExecution> {
        match self {
            Self::Streamed(e) => Some(e),
            _ => None,
        }
    }
    pub fn as_persistent(&self) -> Option<&PersistentExecution> {
        match self {
            Self::Persistent(e) => Some(e),
            _ => None,
        }
    }
    /// Whole-domain tasks for Streamed; rank-local singleton tasks for Persistent.
    pub fn tasks(&self) -> &[Task] {
        match self {
            Self::Streamed(e) => &e.tasks,
            Self::Persistent(e) => &e.tasks,
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        match self {
            Self::Streamed(_) => 0,
            Self::Persistent(e) => e.workspace().bytes,
        }
    }

    pub(super) fn workspace_alignment(&self) -> usize {
        match self {
            Self::Streamed(_) => 1,
            Self::Persistent(_) => persistent::Workspace::ALIGNMENT,
        }
    }

    pub(super) fn reserved_shared_memory_bytes(&self) -> usize {
        match self {
            Self::Streamed(_) => 0,
            Self::Persistent(_) => persistent::SHARED_MEMORY_RESERVE_BYTES,
        }
    }
}
