//! Errors shared by emission.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmitError {
    #[error("CUDA emission does not support {0}")]
    Unsupported(String),
    #[error("invalid CUDA implementation contract: {0}")]
    Contract(String),
    #[error("CUDA template rendering failed: {0}")]
    Template(#[from] minijinja::Error),
}
