//! Resolve one selected program without changing its loop or kernel schedule.
mod expression;
mod loops;
pub(crate) mod metadata;
pub(crate) mod precision;
mod storage;
mod tuning;

use super::plan::TritonPlan;
use super::shape::tile;
use super::{Error, Options, invalid};
use crate::analysis::*;
use expression::{broadcast, expression};
use std::collections::BTreeMap;

/// Resolve shapes, storage and initialization into an immutable program plan.
pub fn lower(analysis: ScheduledIr, options: Options) -> Result<TritonPlan, Error> {
    Ok(crate::emit::TritonKernelProvider
        .lower_source(analysis, options)?
        .into_plan())
}

pub(crate) fn lower_projected(
    analysis: ScheduledIr,
    options: Options,
    storage_contracts: BTreeMap<TensorId, crate::Storage>,
) -> Result<TritonPlan, Error> {
    let mut plan = lower_impl(analysis, options, storage_contracts)?;
    tuning::resolve(&mut plan)?;
    Ok(plan)
}

fn lower_impl(
    analysis: ScheduledIr,
    mut options: Options,
    storage_contracts: BTreeMap<TensorId, crate::Storage>,
) -> Result<TritonPlan, Error> {
    let mut bindings = Bindings {
        shapes: std::mem::take(&mut options.shapes),
        symbols: std::mem::take(&mut options.symbols),
    };
    let tensor_metadata = TensorMetadata::collect(&analysis, &mut bindings)?;
    let metadata = metadata::resolve(&analysis, &mut bindings, &tensor_metadata, &mut options)?;
    let common = ProgramFacts::resolve(&analysis, &mut bindings, tensor_metadata)?;
    let allocation = crate::analysis::storage::for_values(
        &analysis,
        &bindings,
        &common.kernels,
        &storage_contracts,
    )?;
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
    let dtypes =
        analysis
            .tensors()
            .iter()
            .map(|tensor| {
                options.dtypes.get(&tensor.name).copied().ok_or_else(|| {
                    invalid(format!("missing PhysicalPlan dtype for {}", tensor.name))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    let mut plan = TritonPlan {
        analysis,
        common,
        options,
        kernels: Vec::new(),
        accesses,
        expressions,
        dtypes,
        globals: allocation.globals,
        storage_contracts,
        metadata,
        tuning: Vec::new(),
    };
    for ki in 0..plan.analysis.kernels().len() {
        let kernel = storage::plan_kernel(
            ki,
            &plan.analysis,
            &plan.options,
            &plan.common.kernels[ki],
            &allocation.kernels[ki],
        )?;
        plan.kernels.push(kernel);
    }
    Ok(plan)
}
