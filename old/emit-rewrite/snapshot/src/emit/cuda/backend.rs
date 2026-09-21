use super::EmitError;
use crate::{
    Expression, IndexExpr, LoopDomain, LoopKind, OperationId, PhysicalPlan, ValueInstanceId,
};

/// Rank-local, row-major rectangular access. Unused axes have extent one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub value: ValueInstanceId,
    pub rank: usize,
    pub origin: [usize; 3],
    pub extent: [usize; 3],
}
impl Region {
    pub fn new<const N: usize>(
        value: ValueInstanceId,
        rank: usize,
        origin: [usize; N],
        extent: [usize; N],
    ) -> Self {
        assert!((1..=3).contains(&N));

        let mut result = Self {
            value,
            rank,
            origin: [0; 3],
            extent: [1; 3],
        };

        result.origin[..N].copy_from_slice(&origin);
        result.extent[..N].copy_from_slice(&extent);
        result
    }
}

/// Accesses of one CTA body, including readiness at each pipeline stage.
#[derive(Debug, Clone, Default)]
pub struct Work {
    pub coordinate: [usize; 3],
    pub reads: Vec<Region>,
    pub stages: Vec<Vec<Region>>,
    pub writes: Vec<Region>,
    pub ordered_collective: bool,
}

/// Logical backend footprint used by common validation, without readiness slots.
#[derive(Debug, Clone, Default)]
pub struct Accesses {
    pub coordinate: [usize; 3],
    pub reads: Vec<Region>,
    pub writes: Vec<Region>,
    pub ordered_collective: bool,
}

impl From<Accesses> for Work {
    fn from(a: Accesses) -> Self {
        Self {
            coordinate: a.coordinate,
            reads: a.reads,
            writes: a.writes,
            ordered_collective: a.ordered_collective,
            stages: Vec::new(),
        }
    }
}

/// A backend's physical input traversal inside an existing accumulation loop.
/// Persistent emission instantiates this pattern; it never changes Task granularity.
pub struct StageAccessPattern {
    pub domain: LoopDomain,
    pub expression: Expression,
    pub whole_compute_inputs: bool,
}

/// A selected implementation's schedule, before physical validation.
/// Dimensions are outermost first; the expression describes one scheduled tile.
#[derive(Debug, Clone, Default)]
pub struct OperationSchedule {
    pub dimensions: Vec<(LoopKind, LoopDomain)>,
    pub expression: Option<Expression>,
    pub coordinates: Vec<IndexExpr>,
}

/// Symbolic device phases supplied by an implementation.
pub type CudaPhaseTemplate = super::Body;

/// A resolved tensor tile in an implementation's sequential accumulation scope.
/// Logical origins remain separate from memory origins for promoted shared tiles.
#[derive(Debug, Clone)]
pub struct AccumulationTile {
    pub value: ValueInstanceId,
    pub dtype: crate::DType,
    pub storage: crate::Storage,
    pub width: Vec<usize>,
    /// Coordinate-dependent valid extents; `width` retains static storage capacity.
    pub valid_width: Vec<String>,
    pub origin: Vec<String>,
    pub memory_shape: Vec<usize>,
    pub memory_origin: Vec<String>,
    /// Global/external address. Shared addresses are connected by Body bindings.
    pub pointer: Option<String>,
}
impl AccumulationTile {
    pub fn offset(&self, coordinates: &[String]) -> String {
        super::indexing::offset(
            &self.memory_shape,
            &self.memory_origin,
            coordinates,
            self.storage,
        )
    }
}

/// Loop and operand bindings prepared by the common emitter. Implementations
/// choose fragment layout, tile restrictions, initialization and pipeline code.
#[derive(Debug, Clone)]
pub struct AccumulationScope {
    pub variable: String,
    pub start: String,
    pub stop: String,
    pub step: i64,
    pub inputs: [AccumulationTile; 2],
    pub output: AccumulationTile,
    pub initialized: bool,
}

/// Implementation-specific traversal at which the common emitter connects
/// dtype-preserving output materialization or pointwise consumers.
pub struct AccumulationBody {
    pub body: super::Body,
    pub output_coordinates: Vec<String>,
    pub output_expression: String,
    pub output_symbol: super::SymbolId,
}

pub trait CudaImplementation: Sync {
    /// Resolve an unscheduled builder operation using the selected attributes.
    fn schedule(
        &self,
        _plan: &PhysicalPlan,
        _operation: OperationId,
    ) -> Result<OperationSchedule, EmitError> {
        Err(EmitError::Unsupported(
            "implementation requires an explicit schedule".into(),
        ))
    }

    /// Validate the selected instance and provide its composable phases/resources.
    /// Scratch requirements must cover accumulation binding as well: the common
    /// emitter reserves this space before placing shared intermediate tiles.
    fn phases(
        &self,
        plan: &PhysicalPlan,
        operation: OperationId,
    ) -> Result<CudaPhaseTemplate, EmitError>;

    /// Bind a sequential accumulation to the selected implementation's fragments.
    fn accumulation(
        &self,
        _plan: &PhysicalPlan,
        _operation: OperationId,
        _scope: &AccumulationScope,
    ) -> Result<AccumulationBody, EmitError> {
        Err(EmitError::Unsupported(
            "selected implementation does not support accumulation".into(),
        ))
    }

    fn scalar_expression(&self, _operator: &str) -> Option<&'static str> {
        None
    }

    /// Logical custom accesses, also checked for streamed emission. Implement this
    /// alongside any custom `work` override; ordinary compute uses its expression.
    fn accesses(
        &self,
        _plan: &PhysicalPlan,
        _operation: OperationId,
        _rank: usize,
        _coordinate: [usize; 3],
    ) -> Result<Option<Accesses>, EmitError> {
        Ok(None)
    }

    fn stage_accesses(
        &self,
        _plan: &PhysicalPlan,
        _operation: OperationId,
        _domain: &LoopDomain,
        _expression: &Expression,
    ) -> Result<Option<StageAccessPattern>, EmitError> {
        Ok(None)
    }

    /// Communication and custom backends describe accesses at bound coordinates.
    /// Ordinary compute accesses are derived from the same expression rendered
    /// by the body context, including accumulation and WGMMA stage boundaries.
    fn work(
        &self,
        plan: &PhysicalPlan,
        operation: OperationId,
        rank: usize,
        coordinate: [usize; 3],
    ) -> Result<Option<Work>, EmitError> {
        Ok(self
            .accesses(plan, operation, rank, coordinate)?
            .map(Into::into))
    }
}
