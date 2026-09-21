//! Resolve one selected program without changing its loop or kernel schedule.
mod access;
mod dependencies;
mod expression;
mod loops;
pub(crate) mod metadata;
mod storage;

use super::plan::TritonPlan;
use super::shape::tile;
use super::{Error, Options, invalid};
use crate::analysis::*;
use expression::{broadcast, expression};
use std::collections::BTreeSet;

/// Resolve shapes, storage and initialization into an immutable program plan.
pub fn lower(analysis: ScheduledIr, mut options: Options) -> Result<TritonPlan, Error> {
    let mut bindings = Bindings {
        shapes: std::mem::take(&mut options.shapes),
        symbols: std::mem::take(&mut options.symbols),
    };
    let tensor_metadata = TensorMetadata::collect(&analysis, &mut bindings)?;
    let metadata = metadata::resolve(&analysis, &mut bindings, &tensor_metadata, &mut options)?;
    let common = ProgramFacts::resolve(&analysis, &mut bindings, tensor_metadata)?;
    options.shapes = bindings.shapes.clone();
    options.symbols = bindings.symbols.clone();
    let accesses = common
        .accesses
        .iter()
        .map(tile)
        .collect::<Result<Vec<_>, _>>()?;
    let expressions = analysis
        .statements()
        .iter()
        .map(|s| expression(&s.expression, &accesses, &options))
        .collect::<Result<Vec<_>, _>>()?;
    for (id, statement) in analysis.statements().iter().enumerate() {
        let write = statement.accesses.last().unwrap();
        let expected = &accesses[write.index()].shape;
        if broadcast(&expressions[id].shape, expected)? != *expected {
            return Err(invalid(format!(
                "statement {id}: result {:?} cannot store into {expected:?}",
                expressions[id].shape
            )));
        }
    }
    let mut plan = TritonPlan {
        analysis,
        common,
        options,
        kernels: Vec::new(),
        accesses,
        expressions,
        globals: BTreeSet::new(),
        metadata,
    };
    for ki in 0..plan.analysis.kernels().len() {
        let kernel = storage::plan_kernel(
            ki,
            &plan.analysis,
            &plan.options,
            &mut plan.globals,
            &bindings,
            &plan.common.kernels[ki],
        )?;
        plan.kernels.push(kernel);
    }
    Ok(plan)
}
