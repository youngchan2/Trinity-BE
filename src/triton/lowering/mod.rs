//! Resolve one selected program without changing its loop or kernel schedule.
mod access;
mod dependencies;
mod expression;
mod loops;
pub(crate) mod metadata;
mod storage;

use super::plan::ProgramPlan;
use super::shape::{product, tile};
use super::{Error, Options, invalid};
use crate::analysis::*;
use expression::{broadcast, expression};
use std::collections::BTreeSet;

/// Resolve shapes, storage and initialization into an immutable program plan.
pub fn lower(analysis: ProgramAnalysis, mut options: Options) -> Result<ProgramPlan, Error> {
    let metadata = metadata::resolve(&analysis, &mut options)?;
    for shape in options.shapes.values() {
        product(shape)?;
    }
    let accesses = analysis
        .accesses()
        .iter()
        .map(|a| tile(a, &analysis, &options))
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
    let mut plan = ProgramPlan {
        analysis,
        options,
        kernels: Vec::new(),
        accesses,
        expressions,
        globals: BTreeSet::new(),
        metadata,
    };
    let mut previously_written = BTreeSet::new();
    let mut previous_definitions: Vec<AccessId> = Vec::new();
    for ki in 0..plan.analysis.kernels().len() {
        let kernel = storage::plan_kernel(
            ki,
            &plan.analysis,
            &plan.options,
            &mut plan.globals,
            &mut previously_written,
            &mut previous_definitions,
        )?;
        plan.kernels.push(kernel);
    }
    Ok(plan)
}
