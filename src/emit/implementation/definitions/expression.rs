//! Shared geometry helpers for implementation enumeration.

pub(super) const THREADS: usize = 128;

pub(super) fn supported_shape(shape: &[usize]) -> bool {
    matches!(shape.len(), 1 | 2) && !shape.contains(&0)
}
