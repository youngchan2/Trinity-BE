use thiserror::Error;

use super::Storage;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PhysicalInvariantError {
    #[error("invalid structured program: {0}")]
    InvalidProgram(String),
    #[error("physical plan world size must be greater than zero")]
    InvalidWorldSize,

    #[error("tensor binding names must not be empty")]
    EmptyTensorName,

    #[error("input tensor `{tensor}` is bound more than once")]
    DuplicateInputTensor { tensor: String },

    #[error("value instance {value} referenced by {context} does not exist")]
    InvalidValueId { value: usize, context: &'static str },

    #[error("operation {operation} referenced by a statement does not exist")]
    InvalidOperationId { operation: usize },

    #[error("input value instance {value} also has an operation producer")]
    BoundaryInputHasProducer { value: usize },

    #[error("value instance {value} is consumed without a producer or input binding")]
    MissingProducer { value: usize },

    #[error("boundary value instance {value} must use External storage")]
    InvalidBoundaryStorage { value: usize },

    #[error("External value instance {value} is not an ABI input or output")]
    UnboundExternalValue { value: usize },

    #[error("statement {statement} contains no operations")]
    EmptyStatement { statement: usize },

    #[error("operation {operation} does not belong to a statement")]
    MissingStatementMembership { operation: usize },

    #[error("operation {operation} belongs to multiple statements")]
    DuplicateStatementMembership { operation: usize },

    #[error("statement {statement} contains operation {operation} more than once")]
    DuplicateOperationInStatement { statement: usize, operation: usize },

    #[error("value instance {value} in {storage:?} storage crosses a statement boundary")]
    CrossStatementStorage { value: usize, storage: Storage },
}
