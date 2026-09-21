//! Provider-independent expression recognition. No changes to loops, storage,
//! dtype boundaries or operation grouping are performed here.

use crate::{Expression as E, TensorAccess};

pub(super) enum Addend<'a> {
    Residual(&'a TensorAccess),
    ColumnBias(&'a TensorAccess),
}

pub(super) struct GemmEpilogue<'a> {
    pub lhs: &'a TensorAccess,
    pub rhs: &'a TensorAccess,
    pub output: &'a TensorAccess,
    pub addend: Option<Addend<'a>>,
    pub relu: bool,
}

/// Recognize one store of [relu](A@B [+ C or column bias]). Do not fuse separate
/// stores: an intervening store can round to a different dtype or have other users.
pub(super) fn gemm_epilogue(expression: &E) -> Option<GemmEpilogue<'_>> {
    let E::Store { destination, value } = expression else {
        return None;
    };
    let (value, relu) = match value.as_ref() {
        E::Relu(value) => (value.as_ref(), true),
        value => (value, false),
    };
    let (matmul, addend) = match value {
        E::Add(args) => {
            let addend = match &args[1] {
                E::Load(access) => Addend::Residual(access),
                E::Broadcast { value, axis: 0 } => {
                    let E::Load(access) = value.as_ref() else {
                        return None;
                    };
                    Addend::ColumnBias(access)
                }
                _ => return None,
            };
            (&args[0], Some(addend))
        }
        value => (value, None),
    };
    let E::Matmul(operands) = matmul else {
        return None;
    };
    let [E::Load(lhs), E::Load(rhs)] = operands.as_ref() else {
        return None;
    };
    Some(GemmEpilogue {
        lhs,
        rhs,
        output: destination,
        addend,
        relu,
    })
}
