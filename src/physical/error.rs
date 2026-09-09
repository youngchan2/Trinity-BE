use thiserror::Error;

use super::plan::Storage;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PhysicalInvariantError {
    #[error("physical plan world size must be greater than zero")]
    InvalidWorldSize,

    #[error("tensor binding names must not be empty")]
    EmptyTensorName,

    #[error("input tensor `{tensor}` is bound more than once")]
    DuplicateInputTensor { tensor: String },

    #[error("value instance {value} referenced by {context} does not exist")]
    InvalidValueId { value: usize, context: &'static str },

    #[error("operation {operation} referenced by an action does not exist")]
    InvalidOperationId { operation: usize },

    #[error("value instance {value} has multiple producers")]
    DuplicateProducer { value: usize },

    #[error("operations {first} and {second} perform the same physical operation")]
    DuplicateOperation { first: usize, second: usize },

    #[error("input value instance {value} also has an operation producer")]
    BoundaryInputHasProducer { value: usize },

    #[error("value instance {value} is consumed without a producer or input binding")]
    MissingProducer { value: usize },

    #[error("boundary value instance {value} must use External storage")]
    InvalidBoundaryStorage { value: usize },

    #[error("External value instance {value} is not an ABI input or output")]
    UnboundExternalValue { value: usize },

    #[error("operation graph contains a cycle")]
    OperationCycle,

    #[error("action graph contains a cycle")]
    ActionCycle,

    #[error("action {action} contains no operations")]
    EmptyAction { action: usize },

    #[error("operation {operation} does not belong to an action")]
    MissingActionMembership { operation: usize },

    #[error("operation {operation} belongs to multiple actions")]
    DuplicateActionMembership { operation: usize },

    #[error("action {action} contains operation {operation} more than once")]
    DuplicateOperationInAction { action: usize, operation: usize },

    #[error("value instance {value} in {storage:?} storage crosses an action boundary")]
    CrossActionStorage { value: usize, storage: Storage },
}
