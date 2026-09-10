//! Symbolic CUDA fragments. Binding and renaming happen before source rendering.
use std::collections::{BTreeMap, BTreeSet};

use super::EmitError;
use crate::{DType, Storage, ValueInstanceId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub(crate) usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub id: SymbolId,
    pub name: String,
    pub parameter: bool,
}

/// A phase's logical tensor interface. Concrete coordinates belong to the task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub symbol: SymbolId,
    pub value: ValueInstanceId,
    pub dtype: DType,
    pub storage: Storage,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    pub shared_memory_bytes: usize,
    pub symmetric_values: Vec<ValueInstanceId>,
    pub nvls: bool,
}

impl Resources {
    /// Scratch is reusable between sequential scopes after their completion.
    pub(crate) fn sequential(&mut self, other: &Self) {
        self.shared_memory_bytes = self.shared_memory_bytes.max(other.shared_memory_bytes);
        self.symmetric_values.extend(&other.symmetric_values);
        self.symmetric_values.sort();
        self.symmetric_values.dedup();
        self.nvls |= other.nvls;
    }
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

    fn template(text: &str, symbols: &[Symbol]) -> Result<Self, EmitError> {
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

    fn rename(&mut self, names: &BTreeMap<SymbolId, SymbolId>) {
        for part in &mut self.0 {
            if let Part::Symbol(id) = part {
                *id = names[id];
            }
        }
    }

    fn render(&self, symbols: &BTreeMap<SymbolId, &Symbol>) -> Result<String, EmitError> {
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Phase {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub symbols: Vec<Symbol>,
    pub resources: Resources,
    pub code: Code,
}

impl Phase {
    pub(crate) fn append(&mut self, other: &Self) {
        self.code.append(&other.code);
        self.inputs.extend(other.inputs.iter().cloned());
        self.outputs.extend(other.outputs.iter().cloned());
        for symbol in &other.symbols {
            if !self.symbols.iter().any(|s| s.id == symbol.id) {
                self.symbols.push(symbol.clone());
            }
        }
        self.resources.sequential(&other.resources);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Body {
    pub prologue: Phase,
    pub mainloop: Option<Phase>,
    pub epilogue: Phase,
}

impl Body {
    /// Compile named identifiers in a trusted backend template into symbol slots.
    /// Quoted CUDA/assembly strings and comments are never interpreted as symbols.
    pub(crate) fn cuda(code: &str, repeated: bool, locals: &[&str]) -> Result<Self, EmitError> {
        let out = Self::tokenize(code, locals)?;
        Self::template(
            "",
            repeated.then_some(out.as_str()),
            if repeated { "" } else { &out },
            locals,
            &[],
        )
    }

    fn tokenize(code: &str, locals: &[&str]) -> Result<String, EmitError> {
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

    pub(crate) fn cuda_code(&self, code: &str) -> Result<Code, EmitError> {
        let names: Vec<_> = self
            .phases()
            .flat_map(|p| &p.symbols)
            .filter(|s| !s.parameter)
            .map(|s| s.name.as_str())
            .collect();
        self.code(&Self::tokenize(code, &names)?)
    }

    pub fn template(
        prologue: &str,
        mainloop: Option<&str>,
        epilogue: &str,
        locals: &[&str],
        parameters: &[&str],
    ) -> Result<Self, EmitError> {
        let symbols: Vec<_> = locals
            .iter()
            .chain(parameters)
            .enumerate()
            .map(|(id, name)| Symbol {
                id: SymbolId(id),
                name: (*name).into(),
                parameter: id >= locals.len(),
            })
            .collect();
        if symbols
            .iter()
            .map(|s| &s.name)
            .collect::<BTreeSet<_>>()
            .len()
            != symbols.len()
        {
            return Err(contract("duplicate phase symbol"));
        }
        let phase = |text: &str| {
            Ok::<_, EmitError>(Phase {
                symbols: symbols.clone(),
                code: Code::template(text, &symbols)?,
                ..Phase::default()
            })
        };
        Ok(Self {
            prologue: phase(prologue)?,
            mainloop: mainloop.map(phase).transpose()?,
            epilogue: phase(epilogue)?,
        })
    }

    pub fn symbol(&self, name: &str) -> Result<SymbolId, EmitError> {
        self.phases()
            .flat_map(|p| &p.symbols)
            .find(|s| s.name == name)
            .map(|s| s.id)
            .ok_or_else(|| contract(format!("unknown phase symbol {name}")))
    }

    pub(crate) fn declare(&mut self, name: &str, parameter: bool) -> SymbolId {
        if let Ok(id) = self.symbol(name) {
            return id;
        }
        let id = SymbolId(
            self.phases()
                .flat_map(|p| &p.symbols)
                .map(|s| s.id.0 + 1)
                .max()
                .unwrap_or(0),
        );
        let symbol = Symbol {
            id,
            name: name.into(),
            parameter,
        };
        for phase in self.phases_mut() {
            phase.symbols.push(symbol.clone());
        }
        id
    }

    pub fn code(&self, template: &str) -> Result<Code, EmitError> {
        let mut symbols = Vec::new();
        for phase in self.phases() {
            for symbol in &phase.symbols {
                if !symbols.iter().any(|s: &Symbol| s.id == symbol.id) {
                    symbols.push(symbol.clone());
                }
            }
        }
        Code::template(template, &symbols)
    }

    /// Substitute by identity, consistently in every phase and interface.
    pub fn substitute(&mut self, from: SymbolId, to: SymbolId) {
        for phase in self.phases_mut() {
            phase.code.substitute(from, &Code::symbol(to));
            for binding in phase.inputs.iter_mut().chain(&mut phase.outputs) {
                if binding.symbol == from {
                    binding.symbol = to;
                }
            }
        }
    }

    pub fn bind(&mut self, symbol: SymbolId, code: Code) {
        for phase in self.phases_mut() {
            phase.code.substitute(symbol, &code);
        }
    }

    pub(crate) fn bind_text(
        &mut self,
        name: &str,
        text: impl Into<String>,
    ) -> Result<(), EmitError> {
        let id = self.symbol(name)?;
        self.bind(id, Code::text(text));
        Ok(())
    }

    /// Allocate a fresh namespace for this entire body, including phase interfaces.
    pub fn alpha_rename(&mut self, next: &mut usize) {
        let ids: BTreeSet<_> = self
            .phases()
            .flat_map(|p| p.symbols.iter().map(|s| s.id))
            .collect();
        let mut names = BTreeMap::new();
        for id in ids {
            names.insert(id, SymbolId(*next));
            *next += 1;
        }
        for phase in self.phases_mut() {
            phase.code.rename(&names);
            for symbol in &mut phase.symbols {
                symbol.id = names[&symbol.id];
            }
            for binding in phase.inputs.iter_mut().chain(&mut phase.outputs) {
                binding.symbol = names[&binding.symbol];
            }
        }
    }

    pub fn resources(&self) -> Resources {
        let mut resources = Resources::default();
        for phase in self.phases() {
            resources.sequential(&phase.resources);
        }
        resources
    }

    pub fn render(&self) -> Result<String, EmitError> {
        let symbols = self
            .phases()
            .flat_map(|p| p.symbols.iter().map(|s| (s.id, s)))
            .collect();
        let mut code = String::new();
        for phase in self.phases() {
            code.push_str(&phase.code.render(&symbols)?);
        }
        Ok(code)
    }

    pub(crate) fn into_phase(self) -> Phase {
        let mut phase = self.prologue;
        if let Some(mainloop) = self.mainloop {
            phase.append(&mainloop);
        }
        phase.append(&self.epilogue);
        phase
    }

    pub(crate) fn sequence(mut bodies: Vec<Self>) -> Self {
        if bodies.len() == 1 {
            return bodies.remove(0);
        }
        let has_loop = bodies.iter().any(|b| b.mainloop.is_some());
        let mut phase = Phase::default();
        for body in bodies {
            phase.append(&body.into_phase());
        }
        if has_loop {
            Self {
                mainloop: Some(phase),
                ..Self::default()
            }
        } else {
            Self {
                epilogue: phase,
                ..Self::default()
            }
        }
    }

    pub(crate) fn phases(&self) -> impl Iterator<Item = &Phase> {
        std::iter::once(&self.prologue)
            .chain(&self.mainloop)
            .chain(std::iter::once(&self.epilogue))
    }

    fn phases_mut(&mut self) -> impl Iterator<Item = &mut Phase> {
        std::iter::once(&mut self.prologue)
            .chain(&mut self.mainloop)
            .chain(std::iter::once(&mut self.epilogue))
    }
}

fn contract(message: impl Into<String>) -> EmitError {
    EmitError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renaming_is_consistent_across_all_three_phases_and_bindings() {
        let mut a = Body::template(
            "float ${acc}=0;",
            Some("${acc}+=${input};"),
            "use(${acc});",
            &["acc"],
            &["input"],
        )
        .unwrap();
        let id = a.symbol("acc").unwrap();
        let binding = Binding {
            symbol: id,
            value: ValueInstanceId::from_index(0),
            dtype: DType::Fp32,
            storage: Storage::Register,
        };
        a.prologue.outputs.push(binding.clone());
        a.mainloop.as_mut().unwrap().inputs.push(binding.clone());
        a.epilogue.inputs.push(binding);
        let mut b = a.clone();
        let mut next = 0;
        a.alpha_rename(&mut next);
        b.alpha_rename(&mut next);
        assert_ne!(a.symbol("acc").unwrap(), b.symbol("acc").unwrap());
        for body in [&mut a, &mut b] {
            body.bind_text("input", "1.0f").unwrap();
            let id = body.symbol("acc").unwrap();
            assert_eq!(body.prologue.outputs[0].symbol, id);
            assert_eq!(body.mainloop.as_ref().unwrap().inputs[0].symbol, id);
            assert_eq!(body.epilogue.inputs[0].symbol, id);
            assert_eq!(
                body.render()
                    .unwrap()
                    .matches(&format!("acc_{}", id.0))
                    .count(),
                3
            );
        }
    }

    #[test]
    fn substitution_changes_identity_without_replacing_cuda_text() {
        let mut body = Body::template(
            "float ${producer}=1;",
            None,
            "consume(${consumer}); /* consumer producer */",
            &["producer"],
            &["consumer"],
        )
        .unwrap();
        assert!(body.render().is_err());
        body.substitute(
            body.symbol("consumer").unwrap(),
            body.symbol("producer").unwrap(),
        );
        assert_eq!(
            body.render().unwrap(),
            "float producer_0=1;consume(producer_0); /* consumer producer */"
        );
        assert!(Body::template("${missing}", None, "", &[], &[]).is_err());
    }

    #[test]
    fn cuda_identifier_scanning_keeps_assembly_literals_comments_and_prefixes() {
        let body = Body::cuda(
            "int x=0; int xy=x; asm(\"x\"); /* x */ // x\nuse(${x});",
            false,
            &["x"],
        )
        .unwrap();
        assert_eq!(
            body.render().unwrap(),
            "int x_0=0; int xy=x_0; asm(\"x\"); /* x */ // x\nuse(x_0);"
        );
        assert!(body.mainloop.is_none());
        let body = Body::cuda("x + threadIdx.x + p->x + ns::x", false, &["x"]).unwrap();
        assert_eq!(body.render().unwrap(), "x_0 + threadIdx.x + p->x + ns::x");
    }
}
