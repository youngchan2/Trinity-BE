//! Resolve tensor identities, view dimensions and profile-time scalar parameters.
use super::super::plan::ProgramMetadata;
use super::super::shape::{constant, positive};
use super::super::{Error, Options, invalid};
use crate::analysis::*;
use std::collections::{BTreeMap, BTreeSet};

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

pub(crate) fn symbols(expr: &IndexExpr) -> BTreeSet<String> {
    match expr {
        IndexExpr::Symbol(s) => BTreeSet::from([s.clone()]),
        IndexExpr::Apply(_, args) => args.iter().flat_map(symbols).collect(),
        _ => BTreeSet::new(),
    }
}

fn root(aliases: &BTreeMap<String, String>, s: &str) -> String {
    let mut s = s.to_owned();
    while let Some(next) = aliases.get(&s) {
        if *next == s {
            break;
        }
        s = next.clone();
    }
    s
}

pub(super) fn resolve(
    ir: &ProgramAnalysis,
    options: &mut Options,
) -> Result<ProgramMetadata, Error> {
    options.managed |= ir.scopes().iter().any(|s| s.kind == ScopeKind::SplitLoop);
    let mut result = ProgramMetadata::default();
    let mut names = BTreeSet::new();
    let bindings: BTreeSet<_> = ir
        .scopes()
        .iter()
        .filter_map(|s| s.loop_info.as_ref().map(|l| binding_name(&l.variable)))
        .collect();
    for (i, tensor) in ir.tensors().iter().enumerate() {
        let mut name = identifier(&tensor.name);
        if reserved(&name) || bindings.contains(&name) {
            name.insert_str(0, "tensor_");
        }
        while !names.insert(name.clone()) {
            name.push_str(&format!("_{i}"));
        }
        result.tensor_names.push(name);
        let tid = ir.tensor_id(&tensor.name).unwrap();
        let mut views: Vec<_> = ir
            .accesses()
            .iter()
            .filter(|a| a.tensor == tid && a.view_shape.is_some())
            .collect();
        views.sort_by_key(|a| a.kind != AccessKind::Write);
        if let Some(first) = views.first() {
            let first_shape = first.view_shape.as_ref().unwrap();
            result.shapes.insert(tid, first_shape.clone());
            let normalized: Vec<_> = first_shape
                .iter()
                .filter(|s| **s != IndexExpr::Integer(1))
                .collect();
            for view in &views[1..] {
                let other: Vec<_> = view
                    .view_shape
                    .as_ref()
                    .unwrap()
                    .iter()
                    .filter(|s| **s != IndexExpr::Integer(1))
                    .collect();
                if normalized.len() != other.len() {
                    continue;
                }
                // Only singleton insertion/removal and axis renaming imply this
                // correspondence. Arbitrary equal-size axes are not unified.
                if normalized.iter().zip(&other).all(|(a, b)| {
                    a == b || matches!((a, b), (IndexExpr::Symbol(_), IndexExpr::Symbol(_)))
                }) {
                    for (a, b) in normalized.iter().zip(other) {
                        if let (IndexExpr::Symbol(a), IndexExpr::Symbol(b)) = (a, b) {
                            let a = root(&result.aliases, a);
                            let b = root(&result.aliases, b);
                            if a != b {
                                result.aliases.insert(b, a);
                            }
                        }
                    }
                }
            }
        } else if let Some(shape) = options.shapes.get(&tensor.name) {
            result.shapes.insert(
                tid,
                shape
                    .iter()
                    .map(|n| IndexExpr::Integer(*n as i64))
                    .collect(),
            );
        }
        let physical_view = options.shapes.get(&tensor.name).is_some_and(|shape| {
            views.first().is_some_and(|a| {
                let view = a.view_shape.as_ref().unwrap();
                shape.len() == view.len()
                    && view
                        .iter()
                        .zip(shape)
                        .all(|(e, n)| !matches!(e, IndexExpr::Integer(v) if *v != *n as i64))
            })
        });
        if !physical_view
            && (tensor.declarations.contains(&TensorKind::Input)
                || tensor.declarations.contains(&TensorKind::Output))
            && let Some(shape) = options.shapes.get(&tensor.name)
        {
            result.shapes.insert(
                tid,
                shape
                    .iter()
                    .map(|n| IndexExpr::Integer(*n as i64))
                    .collect(),
            );
        }
        if physical_view
            && tensor.declarations.contains(&TensorKind::Input)
            && let Some(a) = views.first()
        {
            let view = a.view_shape.as_ref().unwrap();
            if let Some(shape) = options.shapes.get(&tensor.name)
                && shape.len() == view.len()
            {
                for (axis, (expr, size)) in view.iter().zip(shape).enumerate() {
                    if let IndexExpr::Symbol(s) = expr {
                        if options
                            .symbols
                            .insert(s.clone(), *size as i64)
                            .is_some_and(|n| n != *size as i64)
                        {
                            return Err(invalid(format!("inconsistent dimension {s}")));
                        }
                        result.dimensions.entry(s.clone()).or_insert((tid, axis));
                    }
                }
            }
        }
    }
    for (i, scope) in ir.scopes().iter().enumerate() {
        if let Some(info) = &scope.loop_info {
            if let IndexExpr::Symbol(s) = &info.step {
                let default = constant(&info.end, options).unwrap_or(16).clamp(1, 16);
                let default = if default < 16 { 1 } else { 16 };
                options.symbols.entry(s.clone()).or_insert(default);
                let values = options
                    .tuning
                    .get(s)
                    .cloned()
                    .unwrap_or_else(|| vec![options.symbols[s]]);
                result.candidates.insert(s.clone(), values);
            }
            if scope.kind == ScopeKind::SplitLoop
                && let IndexExpr::Symbol(s) = &info.end
            {
                let s = root(&result.aliases, s);
                let fixed = options.symbols.get(&s).copied();
                options.symbols.entry(s.clone()).or_insert(1);
                if result.splits.insert(s.clone(), ScopeId(i)).is_some() {
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
    let aliases: Vec<_> = result.aliases.keys().cloned().collect();
    for alias in aliases {
        let canonical = root(&result.aliases, &alias);
        result.aliases.insert(alias.clone(), canonical.clone());
        if let Some(value) = options.symbols.get(&canonical).copied()
            && options
                .symbols
                .insert(alias.clone(), value)
                .is_some_and(|old| old != value)
        {
            return Err(invalid(format!(
                "inconsistent aliases {alias} and {canonical}"
            )));
        }
    }
    for (name, values) in &result.candidates {
        if values.is_empty()
            || values.iter().any(|v| {
                *v <= 0 || (!result.splits.contains_key(name) && !(*v as u64).is_power_of_two())
            })
        {
            return Err(invalid(format!("invalid profile candidates for {name}")));
        }
    }
    for (tid, shape) in &result.shapes {
        let name = &ir.tensor(*tid).name;
        if !options.shapes.contains_key(name) {
            options.shapes.insert(
                name.clone(),
                shape
                    .iter()
                    .map(|e| positive(e, options))
                    .collect::<Result<_, _>>()?,
            );
        }
    }
    Ok(result)
}
