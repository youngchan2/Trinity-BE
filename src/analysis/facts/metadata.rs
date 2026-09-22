//! Tensor/view dimensions and IR symbol relations, independent of a backend.
use super::scalar::positive;
use crate::analysis::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct TensorMetadata {
    pub shapes: BTreeMap<TensorId, Vec<IndexExpr>>,
    /// Aliases of dimension symbols, not aliases of tensor storage.
    pub aliases: BTreeMap<String, String>,
    pub dimensions: BTreeMap<String, (TensorId, usize)>,
    /// All loop controllers using each split parameter. Sharing a parameter is
    /// an IR fact; a provider may impose a narrower tuning ownership policy.
    pub splits: BTreeMap<String, Vec<ScopeId>>,
}

pub(crate) fn root(aliases: &BTreeMap<String, String>, s: &str) -> String {
    let mut s = s.to_owned();
    while let Some(next) = aliases.get(&s) {
        if *next == s {
            break;
        }
        s = next.clone();
    }
    s
}

impl TensorMetadata {
    /// Collect symbolic facts and infer dimension bindings from input shapes.
    /// Unbound schedule parameters are left for the caller to specialize.
    pub fn collect(ir: &ScheduledIr, bindings: &mut Bindings) -> Result<Self, ResolveError> {
        let mut result = Self::default();
        for (i, tensor) in ir.tensors().iter().enumerate() {
            let tid = TensorId(i);
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
            } else if let Some(shape) = bindings.shapes.get(&tensor.name) {
                result.shapes.insert(
                    tid,
                    shape
                        .iter()
                        .map(|n| IndexExpr::Integer(*n as i64))
                        .collect(),
                );
            }
            let physical_view = bindings.shapes.get(&tensor.name).is_some_and(|shape| {
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
                && let Some(shape) = bindings.shapes.get(&tensor.name)
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
                if let Some(shape) = bindings.shapes.get(&tensor.name)
                    && shape.len() == view.len()
                {
                    for (axis, (expr, size)) in view.iter().zip(shape).enumerate() {
                        if let IndexExpr::Symbol(s) = expr {
                            if bindings
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
            if scope.kind == ScopeKind::SplitLoop
                && let Some(LoopInfo {
                    end: IndexExpr::Symbol(s),
                    ..
                }) = &scope.loop_info
            {
                let s = root(&result.aliases, s);
                result.splits.entry(s).or_default().push(ScopeId(i));
            }
        }
        Ok(result)
    }

    /// Resolve dimensions after the caller has supplied remaining IR parameters.
    pub fn resolve_shapes(
        &mut self,
        ir: &ScheduledIr,
        bindings: &mut Bindings,
    ) -> Result<(), ResolveError> {
        let aliases: Vec<_> = self.aliases.keys().cloned().collect();
        for alias in aliases {
            let canonical = root(&self.aliases, &alias);
            self.aliases.insert(alias.clone(), canonical.clone());
            if let Some(value) = bindings.symbols.get(&canonical).copied()
                && bindings
                    .symbols
                    .insert(alias.clone(), value)
                    .is_some_and(|old| old != value)
            {
                return Err(invalid(format!(
                    "inconsistent aliases {alias} and {canonical}"
                )));
            }
        }
        for (tid, shape) in &self.shapes {
            let name = &ir.tensor(*tid).name;
            if !bindings.shapes.contains_key(name) {
                bindings.shapes.insert(
                    name.clone(),
                    shape
                        .iter()
                        .map(|e| positive(e, &bindings.symbols))
                        .collect::<Result<_, _>>()?,
                );
            }
        }
        Ok(())
    }
}
