#![allow(dead_code)]
use trinity_lowering::*;
pub const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Hopper);
pub fn tile(variable: &str, width: usize) -> AccessIndex {
    AccessIndex::Tile {
        variable: variable.into(),
        width,
    }
}
pub fn load(value: ValueInstanceId, indices: impl IntoIterator<Item = AccessIndex>) -> Expression {
    Expression::Load(TensorAccess::new(value, indices))
}
pub fn store(
    value: ValueInstanceId,
    rhs: Expression,
    indices: impl IntoIterator<Item = AccessIndex>,
) -> Expression {
    Expression::Store {
        destination: TensorAccess::new(value, indices),
        value: Box::new(rhs),
    }
}
pub fn loop_node(
    kind: LoopKind,
    var: &str,
    start: i64,
    stop: i64,
    step: i64,
    body: Vec<Statement>,
) -> Statement {
    Statement::Loop(Loop {
        kind,
        domain: LoopDomain {
            variable: var.into(),
            start: IndexExpr::Constant(start),
            stop: IndexExpr::Constant(stop),
            step: IndexExpr::Constant(step),
        },
        body,
    })
}
