//! Special functions in the FP32 pointwise computation.

pub(super) fn relu(operand: &str) -> String {
    // Preserve NaN and signed zero, matching the existing pointwise semantics.
    format!("{operand} < 0.0f ? 0.0f : {operand}")
}

pub(super) fn sigmoid(operand: &str) -> String {
    format!("1.0f / (1.0f + expf(-{operand}))")
}
