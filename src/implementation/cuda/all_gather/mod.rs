mod nvls;
pub(super) mod peer;
use serde::Serialize;
#[derive(Serialize)]
pub(super) struct GatherContext {
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) source_rows: usize,
    pub(super) source_cols: usize,
    pub(super) target_rows: usize,
    pub(super) target_cols: usize,
    pub(super) axis: usize,
    pub(super) push: bool,
    pub(super) chunk: usize,
}

pub(super) static IMPLEMENTATIONS: &[&dyn crate::AllGatherImplementation] = &[
    &nvls::NVLS_ONE_SHOT_PUSH_NBI,
    &peer::PEER_PUSH,
    &peer::PEER_PULL,
];
