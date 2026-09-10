//! Mutable source buffers, names and temporary values for one emission.
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn tuple(values: impl IntoIterator<Item = impl ToString>) -> String {
    let values: Vec<_> = values.into_iter().map(|v| v.to_string()).collect();
    format!(
        "({}{})",
        values.join(", "),
        if values.len() == 1 { "," } else { "" }
    )
}

#[derive(Default)]
pub(super) struct CodegenContext {
    pub(super) source: String,
    pub(super) indent: usize,
    pub(super) temp: usize,
    pub(super) offset: usize,
    pub(super) mask: usize,
    pub(super) indices: BTreeSet<String>,
    pub(super) loads: BTreeMap<String, EmittedValue>,
}
impl CodegenContext {
    pub(super) fn line(&mut self, text: impl AsRef<str>) {
        if !text.as_ref().is_empty() {
            self.source.push_str(&"    ".repeat(self.indent));
        }
        self.source.push_str(text.as_ref());
        self.source.push('\n');
    }
    pub(super) fn temporary(&mut self, expression: impl AsRef<str>) -> String {
        let name = format!("temp_{}", self.temp);
        self.temp += 1;
        self.line(format!("{name} = {}", expression.as_ref()));
        name
    }
}
#[derive(Clone)]
pub(super) struct EmittedValue {
    pub(super) code: String,
    pub(super) shape: Vec<String>,
    /// Rectangular validity, one one-dimensional predicate per logical axis.
    /// Contracting an axis removes its predicate; it never reduces mask tensors.
    pub(super) valid: Vec<Option<String>>,
    /// A masked load already supplies zero. Pointwise operations may change it.
    pub(super) zero_invalid: bool,
}
