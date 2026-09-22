//! Triton/Python names and profile policy; tensor and view facts live in analysis.
use super::super::plan::ProgramMetadata;
use super::super::{Error, Options, invalid};
use crate::analysis::metadata::root;
use crate::analysis::scalar::constant;
use crate::analysis::*;
use std::collections::BTreeSet;

pub(crate) fn identifier(name: &str) -> String {
    let mut value: String = name
        .trim_start_matches('?')
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if value.is_empty() || value.as_bytes()[0].is_ascii_digit() {
        value.insert_str(0, "v_");
    }
    if [
        "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
        "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
        "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
        "try", "while", "with", "yield", "tl", "torch", "triton", "range",
    ]
    .contains(&value.as_str())
    {
        value.insert_str(0, "v_");
    }
    value
}

pub(crate) fn binding_name(name: &str) -> String {
    let name = identifier(name);
    if reserved(&name) {
        format!("iv_{name}")
    } else {
        name
    }
}

fn reserved(name: &str) -> bool {
    [
        "temp_",
        "offset_",
        "mask_",
        "linear_",
        "META_",
        "kernel_",
        "capacity_",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

pub(crate) fn resolve(
    ir: &ScheduledIr,
    bindings: &mut Bindings,
    common: &TensorMetadata,
    options: &mut Options,
) -> Result<ProgramMetadata, Error> {
    let mut result = ProgramMetadata {
        output_order: ir
            .declared_tensors(TensorKind::Output)
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let mut names = BTreeSet::new();
    let loop_names: BTreeSet<_> = ir
        .scopes()
        .iter()
        .filter_map(|s| s.loop_info.as_ref().map(|l| binding_name(&l.variable)))
        .collect();
    for (i, tensor) in ir.tensors().iter().enumerate() {
        let mut name = identifier(&tensor.name);
        if reserved(&name) || loop_names.contains(&name) {
            name.insert_str(0, "tensor_");
        }
        while !names.insert(name.clone()) {
            name.push_str(&format!("_{i}"));
        }
        result.tensor_names.push(name);
    }
    for (i, scope) in ir.scopes().iter().enumerate() {
        if let Some(info) = &scope.loop_info {
            if let IndexExpr::Symbol(s) = &info.step {
                let default = constant(&info.end, &bindings.symbols)
                    .unwrap_or(16)
                    .clamp(1, 16);
                let default = if default < 16 { 1 } else { 16 };
                bindings.symbols.entry(s.clone()).or_insert(default);
                let values = options.tuning.get(s).cloned().unwrap_or_else(|| {
                    let baseline = bindings.symbols[s];
                    let mut values = vec![baseline];
                    let extent = constant(&info.end, &bindings.symbols)
                        .ok()
                        .zip(constant(&info.start, &bindings.symbols).ok())
                        .map(|(end, start)| {
                            ((end - start).max(1) as u64).next_power_of_two() as i64
                        });
                    for value in [16, 32, 64, 128, 256] {
                        if value != baseline && extent.is_none_or(|n| value <= n) {
                            values.push(value);
                        }
                    }
                    values
                });
                result.candidates.insert(s.clone(), values);
            }
            if scope.kind == ScopeKind::SplitLoop
                && let IndexExpr::Symbol(s) = &info.end
            {
                let s = root(&common.aliases, s);
                let fixed = bindings.symbols.get(&s).copied();
                bindings.symbols.entry(s.clone()).or_insert(1);
                if result.split_owners.insert(s.clone(), ScopeId(i)).is_some() {
                    return Err(invalid(format!(
                        "split parameter {s} has multiple mloop controllers; use distinct symbols or fuse the loops before lowering"
                    )));
                }
                result.candidates.insert(
                    s.clone(),
                    options.tuning.get(&s).cloned().unwrap_or_else(|| {
                        fixed.map(|v| vec![v]).unwrap_or_else(|| vec![1, 2, 4, 8])
                    }),
                );
            }
        }
    }
    for (name, values) in &result.candidates {
        if values.is_empty()
            || values.iter().any(|v| {
                *v <= 0 || (!common.splits.contains_key(name) && !(*v as u64).is_power_of_two())
            })
        {
            return Err(invalid(format!("invalid profile candidates for {name}")));
        }
    }
    Ok(result)
}
