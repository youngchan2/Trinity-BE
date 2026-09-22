//! Search existing IR tile symbols, without changing the selected schedule.
//! Re-lowering each tile assignment validates shapes and the storage/ownership
//! decisions that the emitted kernel shares across its constexpr variants.
use super::super::{Error, KernelPlan, TritonPlan, TuningConfig, invalid};
use crate::analysis::{IndexDim, IndexExpr, ScopeId, TensorKind, ValueExpr};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn resolve(plan: &mut TritonPlan) -> Result<(), Error> {
    let policy = &plan.options.autotune;
    if policy.max_configs == 0
        || policy.num_warps.is_empty()
        || policy
            .num_warps
            .iter()
            .any(|w| ![1, 2, 4, 8, 16, 32].contains(w))
        || policy.num_stages.is_empty()
        || policy.num_stages.contains(&0)
    {
        return Err(invalid(
            "invalid autotuning budget, num_warps or num_stages",
        ));
    }
    let mut tuning = Vec::new();
    for (ki, kernel) in plan.kernels.iter().enumerate() {
        let symbols: Vec<_> = plan.owned_parameters(kernel).into_iter().collect();
        let choices: Vec<_> = symbols
            .iter()
            .map(|s| &plan.metadata.candidates[s])
            .collect();
        let count = choices
            .iter()
            .try_fold(1usize, |n, c| n.checked_mul(c.len()))
            .ok_or_else(|| invalid("autotuning search space overflow"))?;
        // Sample a large Cartesian space before doing expensive validation.
        // Include baseline and all single-axis changes before joint samples.
        let limit = if policy.max_configs == 1 { 1 } else { 512 };
        let mut indices = BTreeSet::from([0]);
        if limit > 1 {
            let mut stride = 1;
            for values in &choices {
                for v in 1..values.len() {
                    indices.insert(v * stride);
                }
                stride *= values.len();
            }
            let n = count.min(limit);
            for i in 0..n {
                indices.insert(if n == 1 {
                    0
                } else {
                    (i as u128 * (count - 1) as u128 / (n - 1) as u128) as usize
                });
            }
        }
        let mut rows = Vec::new();
        for mut index in indices {
            let mut row = BTreeMap::new();
            for (symbol, values) in symbols.iter().zip(&choices) {
                row.insert(symbol.clone(), values[index % values.len()]);
                index /= values.len();
            }
            if valid_assignment(plan, ki, kernel, &row) {
                rows.push(row);
            }
        }
        if rows.is_empty() {
            return Err(invalid(format!(
                "kernel {ki}: no tile candidates preserve the IR shape and ownership contract"
            )));
        }
        // Round-robin hardware settings across all tiles so a bounded budget
        // does not consume all its slots on the first tile assignment.
        let has_dot = plan.analysis.statements().iter().any(|s| {
            plan.analysis.scope(s.scope).kernel.index() == ki && contains_dot(&s.expression)
        });
        let stages = if has_dot {
            policy.num_stages.as_slice()
        } else {
            &policy.num_stages[..1]
        };
        let mut configs = Vec::new();
        for stages in stages {
            for warps in &policy.num_warps {
                for row in &rows {
                    configs.push(TuningConfig {
                        parameters: row.clone(),
                        num_warps: *warps,
                        num_stages: *stages,
                    });
                }
            }
        }
        if configs.len() > policy.max_configs {
            let n = policy.max_configs;
            configs = (0..n)
                .map(|i| {
                    configs[if n == 1 {
                        0
                    } else {
                        i * (configs.len() - 1) / (n - 1)
                    }]
                    .clone()
                })
                .collect();
        }
        tuning.push(configs);
    }
    plan.tuning = tuning;
    Ok(())
}

fn valid_assignment(
    plan: &TritonPlan,
    ki: usize,
    kernel: &KernelPlan,
    row: &BTreeMap<String, i64>,
) -> bool {
    // A fixed-size region addressed by a tunable induction variable must not
    // acquire holes/overlap. Such a step is coupled to the fixed IR tile.
    for (i, scope) in plan.analysis.scopes().iter().enumerate() {
        if scope.kernel.index() != ki {
            continue;
        }
        let Some(info) = &scope.loop_info else {
            continue;
        };
        let IndexExpr::Symbol(symbol) = &info.step else {
            continue;
        };
        let Some(value) = row.get(symbol) else {
            continue;
        };
        if *value == plan.options.symbols[symbol] {
            continue;
        }
        for id in &plan.analysis.kernel(scope.kernel).accesses {
            for dim in &plan.analysis.access(*id).index {
                if let IndexDim::Tile { start, width } = dim
                    && start.loop_dependencies().contains(&ScopeId(i))
                    && !crate::analysis::scalar::symbols(width).contains(symbol)
                {
                    return false;
                }
            }
        }
    }
    if row
        .iter()
        .all(|(s, v)| plan.options.symbols.get(s) == Some(v))
    {
        return true;
    }
    let mut options = plan.options.clone();
    options.symbols.extend(row.clone());
    for (alias, canonical) in &plan.common.metadata.aliases {
        if let Some(value) = row.get(canonical) {
            options.symbols.insert(alias.clone(), *value);
        }
    }
    // Split scratch capacity is a symbolic allocation, not the concrete size
    // used by the baseline validation. Re-infer it for the trial split count.
    for (tid, shape) in &plan.common.metadata.shapes {
        let tensor = plan.analysis.tensor(*tid);
        if !tensor.declarations.contains(&TensorKind::Input)
            && !tensor.declarations.contains(&TensorKind::Output)
            && shape
                .iter()
                .flat_map(crate::analysis::scalar::symbols)
                .any(|s| {
                    plan.metadata
                        .split_owners
                        .contains_key(plan.canonical_symbol(&s))
                })
        {
            options.shapes.remove(&plan.analysis.tensor(*tid).name);
        }
    }
    let Ok(candidate) = super::lower_impl(
        plan.analysis.clone(),
        options,
        plan.storage_contracts.clone(),
    ) else {
        return false;
    };
    let other = &candidate.kernels[ki];
    kernel.tensors == other.tensors
        && kernel.register_accesses == other.register_accesses
        && kernel.local_reads == other.local_reads
        && plan.common.metadata.shapes == candidate.common.metadata.shapes
}

fn contains_dot(expr: &ValueExpr) -> bool {
    matches!(expr, ValueExpr::Apply(op, args) if op == "@" || args.iter().any(contains_dot))
}
