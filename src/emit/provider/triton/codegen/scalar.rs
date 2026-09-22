//! One spelling for each IR scalar in bounds, shapes, grids and tuning metadata.
use super::super::TritonPlan;
use super::super::lowering::metadata::identifier;
use crate::analysis::*;

impl TritonPlan {
    pub(super) fn tensor_name(&self, id: TensorId) -> &str {
        &self.metadata.tensor_names[id.index()]
    }
    pub(super) fn parameter(&self, symbol: &str) -> String {
        let symbol = self.canonical_symbol(symbol);
        let name = identifier(symbol);
        let conflict = self
            .common
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
