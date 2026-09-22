//! Contracts inspected before rendering or choosing an implementation.

use super::access::Access;

/// A thread-to-element mapping. Equality requires the same representation and mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) enum RegisterLayout {
    Fixed {
        representation: &'static str,
        distribution: String,
    },
    /// Elementwise code follows its register input's coordinates and active lanes.
    FollowInput { representation: &'static str },
}

impl RegisterLayout {
    pub fn accepts(&self, producer: &Self) -> bool {
        match (self, producer) {
            (
                Self::FollowInput { representation: a },
                Self::Fixed {
                    representation: b, ..
                },
            ) => a == b,
            _ => self == producer,
        }
    }
}

/// One access, rather than one tensor: distinct projections have distinct port indices.
#[derive(Debug, Clone)]
pub(in crate::emit) struct KernelPort {
    pub access: Access,
    /// None means this port requires memory. Register mappings are explicit.
    pub register: Option<RegisterLayout>,
    /// Coordinates projected from the output (e.g. a row broadcast selects axis 0).
    pub projection: Vec<usize>,
}

impl KernelPort {
    pub fn memory(access: &Access) -> Self {
        Self {
            access: access.clone(),
            register: None,
            projection: (0..access.axes.len()).collect(),
        }
    }
}

/// Inputs are consumed in mainloop when `iteration` is present, otherwise in
/// epilogue. Prologue prepares state; outputs become ready in guarded epilogue
/// scopes after the implementation has completed its asynchronous work.
#[derive(Debug, Clone)]
pub(in crate::emit) struct KernelInterface {
    pub inputs: Vec<KernelPort>,
    pub outputs: Vec<KernelPort>,
    /// Prologue/epilogue surround this existing sequential loop; None runs inline.
    pub iteration: Option<String>,
}
