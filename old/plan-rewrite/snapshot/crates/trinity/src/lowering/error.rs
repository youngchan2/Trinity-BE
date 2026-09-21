//! Errors produced while lowering a logical candidate into physical plans.

use thiserror::Error;

use trinity_lowering::PhysicalInvariantError;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LoweringError {
    #[error(transparent)]
    InvalidPlan(#[from] PhysicalInvariantError),
}
