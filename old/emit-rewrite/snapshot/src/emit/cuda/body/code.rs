//! Symbolic CUDA text, identifier scanning, and substitution.
use std::collections::BTreeMap;

use super::{EmitError, contract};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub(crate) usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub id: SymbolId,
    pub name: String,
    pub parameter: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Part {
    Text(String),
    Symbol(SymbolId),
}

/// Only explicit symbol references are substitutable; CUDA text is opaque.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Code(Vec<Part>);

impl Code {
    pub fn text(text: impl Into<String>) -> Self {
        Self(vec![Part::Text(text.into())])
    }

    pub fn symbol(symbol: SymbolId) -> Self {
        Self(vec![Part::Symbol(symbol)])
    }

    pub fn append(&mut self, other: &Self) {
        self.0.extend(other.0.iter().cloned());
    }

    pub(super) fn template(text: &str, symbols: &[Symbol]) -> Result<Self, EmitError> {
        let mut code = Self::default();
        let mut remaining = text;
        while let Some(start) = remaining.find("${") {
            code.0.push(Part::Text(remaining[..start].into()));
            remaining = &remaining[start + 2..];
            let end = remaining
                .find('}')
                .ok_or_else(|| contract("unclosed symbol slot"))?;
            let name = &remaining[..end];
            let symbol = symbols
                .iter()
                .find(|s| s.name == name)
                .ok_or_else(|| contract(format!("undeclared phase symbol {name}")))?;
            code.0.push(Part::Symbol(symbol.id));
            remaining = &remaining[end + 1..];
        }
        code.0.push(Part::Text(remaining.into()));
        Ok(code)
    }

    pub(crate) fn substitute(&mut self, symbol: SymbolId, replacement: &Self) {
        self.0 = self
            .0
            .iter()
            .flat_map(|part| match part {
                Part::Symbol(id) if *id == symbol => replacement.0.clone(),
                _ => vec![part.clone()],
            })
            .collect();
    }

    pub(super) fn rename(&mut self, names: &BTreeMap<SymbolId, SymbolId>) {
        for part in &mut self.0 {
            if let Part::Symbol(id) = part {
                *id = names[id];
            }
        }
    }

    pub(super) fn render(
        &self,
        symbols: &BTreeMap<SymbolId, &Symbol>,
    ) -> Result<String, EmitError> {
        let mut result = String::new();
        for part in &self.0 {
            match part {
                Part::Text(text) => result.push_str(text),
                Part::Symbol(id) => {
                    let symbol = symbols
                        .get(id)
                        .ok_or_else(|| contract("unknown symbol reference"))?;
                    if symbol.parameter {
                        return Err(contract(format!("unbound phase parameter {}", symbol.name)));
                    }
                    result.push_str(&format!("{}_{}", symbol.name, id.0));
                }
            }
        }
        Ok(result)
    }
}

pub(super) fn tokenize(code: &str, locals: &[&str]) -> Result<String, EmitError> {
    let bytes = code.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        if code[i..].starts_with("${") {
            i += code[i..]
                .find('}')
                .ok_or_else(|| contract("unclosed symbol slot"))?
                + 1;
        } else if matches!(bytes[i], b'\"' | b'\'') {
            let quote = bytes[i];
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                } else if bytes[i] == quote {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
        } else if code[i..].starts_with("//") {
            i += code[i..].find('\n').unwrap_or(bytes.len() - i);
        } else if code[i..].starts_with("/*") {
            i += code[i + 2..].find("*/").map_or(bytes.len() - i, |n| n + 4);
        } else if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let name = &code[start..i];
            // A field or qualified member is not a free local identifier
            // (notably threadIdx.x when the expression also declares x).
            let prefix = code[..start].trim_end();
            let qualified =
                prefix.ends_with('.') || prefix.ends_with("->") || prefix.ends_with("::");
            if locals.contains(&name) && !qualified {
                out.push_str(&format!("${{{name}}}"));
                continue;
            }
        } else {
            i += code[i..].chars().next().unwrap().len_utf8();
        }
        out.push_str(&code[start..i]);
    }
    Ok(out)
}
