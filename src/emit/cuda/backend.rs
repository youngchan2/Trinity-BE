use super::EmitError;
use crate::{OperationId, PhysicalPlan, ValueInstanceId};

/// A row-major rectangular tensor region on one concrete rank.
#[derive(Debug, Clone, Copy)]
pub struct Region {
    pub value: ValueInstanceId,
    pub rank: usize,
    pub origin: [usize; 2],
    pub extent: [usize; 2],
}

impl Region {
    pub fn new(
        value: ValueInstanceId,
        rank: usize,
        origin: [usize; 2],
        extent: [usize; 2],
    ) -> Self {
        Self {
            value,
            rank,
            origin,
            extent,
        }
    }
}

/// Backend work on one CTA. Persistent execution treats reads as entry
/// conditions and checks stage reads before loading, with stage zero also an
/// entry condition. Its writes become visible when the work's token is
/// published. Streamed execution uses these regions as validation/inspection
/// metadata; kernel boundaries on the stream establish readiness and visibility.
#[derive(Debug, Clone, Default)]
pub struct Work {
    pub coordinate: [usize; 3],
    pub reads: Vec<Region>,
    pub stages: Vec<Vec<Region>>,
    pub writes: Vec<Region>,
    /// All such work shares one ordered collective channel across ranks.
    pub ordered_collective: bool,
}

/// Specialization for one operation. The shared body defines
/// `template<class Runtime> __device__ bool operation_N(Bindings const&,
/// Tile const&, void*, Runtime const&)`. Bindings and tile coordinates contain
/// no scheduler state. Runtime supplies CTA-uniform `await_stage(stage)` and
/// `prefetch_stage(stage, stage_count)` hooks; the latter must not wait for an
/// unavailable input. Persistent communication can also use `rank()`,
/// `peer_ptr(pointer, rank)`, and `fail(code)`. Streamed hooks require every input
/// to be ready at kernel entry and cannot report device-side runtime errors.
/// The typed backend context owns template rendering.
#[derive(Debug, Clone)]
pub struct OperationEmission {
    pub body: String,
    /// Work for each rank, indexed by rank.
    pub work: Vec<Vec<Work>>,
    pub shared_memory_bytes: usize,
    pub symmetric_values: Vec<ValueInstanceId>,
    pub nvls: bool,
}

pub trait CudaImplementation: Sync {
    fn specialize(
        &self,
        plan: &PhysicalPlan,
        operation: OperationId,
    ) -> Result<OperationEmission, EmitError>;
}
