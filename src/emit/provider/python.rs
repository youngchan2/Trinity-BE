//! A named, independently callable region implementation for Python composition.
//! Providers own the function body; the program emitter owns allocation/order.
use crate::ValueInstanceId;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct PythonKernel {
    pub entrypoint: String,
    pub arguments: Vec<ValueInstanceId>,
    pub source: String,
    pub imports: BTreeSet<String>,
    pub helpers: BTreeMap<String, String>,
}

pub(crate) fn tuple(values: impl IntoIterator<Item = impl ToString>) -> String {
    let values: Vec<_> = values.into_iter().map(|v| v.to_string()).collect();
    format!(
        "({}{})",
        values.join(", "),
        if values.len() == 1 { "," } else { "" }
    )
}
