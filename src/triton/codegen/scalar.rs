//! One spelling for each IR scalar in bounds, shapes, grids and tuning metadata.
use super::super::lowering::metadata::{identifier, symbols};
use super::super::{KernelPlan, ProgramPlan};
use crate::analysis::*;
use std::collections::BTreeSet;

impl ProgramPlan {
    pub(super) fn tensor_name(&self, id: TensorId) -> &str {
        &self.metadata.tensor_names[id.index()]
    }
    pub(super) fn canonical_symbol<'a>(&'a self, name: &'a str) -> &'a str {
        self.metadata
            .aliases
            .get(name)
            .map(String::as_str)
            .unwrap_or(name)
    }
    pub(super) fn parameter(&self, symbol: &str) -> String {
        let symbol = self.canonical_symbol(symbol);
        let name = identifier(symbol);
        let conflict = self
            .metadata
            .dimensions
            .keys()
            .chain(self.metadata.candidates.keys())
            .any(|other| other != symbol && identifier(other) == name);
        if conflict {
            let suffix: String = symbol
                .as_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            format!("META_{name}_{suffix}")
        } else {
            format!("META_{name}")
        }
    }
    pub(super) fn parameters(&self, kernel: &KernelPlan) -> BTreeSet<String> {
        let mut used = BTreeSet::new();
        let ki = self.analysis.scope(kernel.root_scope).kernel;
        for scope in self.analysis.scopes().iter().filter(|s| s.kernel == ki) {
            if let Some(info) = &scope.loop_info {
                for expr in [&info.start, &info.end, &info.step] {
                    used.extend(symbols(expr));
                }
            }
        }
        for access in &self.analysis.kernel(ki).accesses {
            let a = self.analysis.access(*access);
            if let Some(shape) = &a.view_shape {
                for expr in shape {
                    used.extend(symbols(expr));
                }
            }
            for axis in &self.accesses[access.index()].axes {
                used.extend(symbols(&axis.start));
            }
        }
        used.into_iter()
            .map(|s| self.canonical_symbol(&s).to_owned())
            .filter(|s| {
                self.metadata.dimensions.contains_key(s) || self.metadata.candidates.contains_key(s)
            })
            .collect()
    }
    pub(super) fn owned_parameters(&self, kernel: &KernelPlan) -> BTreeSet<String> {
        let ki = self.analysis.scope(kernel.root_scope).kernel;
        self.parameters(kernel)
            .into_iter()
            .filter(|s| {
                self.metadata.candidates.contains_key(s)
                    && self
                        .metadata
                        .splits
                        .get(s)
                        .is_none_or(|scope| self.analysis.scope(*scope).kernel == ki)
            })
            .collect()
    }
    pub(super) fn view_shape(&self, id: AccessId) -> Vec<String> {
        self.analysis
            .access(id)
            .view_shape
            .as_ref()
            .map(|s| s.iter().map(|e| self.index(e)).collect())
            .unwrap_or_else(|| {
                self.accesses[id.index()]
                    .axes
                    .iter()
                    .map(|a| a.extent.to_string())
                    .collect()
            })
    }
}
