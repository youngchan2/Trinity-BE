//! CUDA body model, composition, and construction.

mod builder;
mod code;
mod program;

pub use code::{Code, Symbol, SymbolId};
pub(super) use program::{BodyDefinition, ProgramBodies, build_bodies};
use std::collections::{BTreeMap, BTreeSet};

use super::EmitError;
use crate::{DType, Storage, ValueInstanceId};

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
        Self::cuda_with_parameters(code, repeated, locals, &[])
    }

    pub(crate) fn cuda_with_parameters(
        code: &str,
        repeated: bool,
        locals: &[&str],
        parameters: &[&str],
    ) -> Result<Self, EmitError> {
        let out = code::tokenize(code, locals)?;
        Self::template(
            "",
            repeated.then_some(out.as_str()),
            if repeated { "" } else { &out },
            locals,
            parameters,
        )
    }

    pub(crate) fn cuda_code(&self, code: &str) -> Result<Code, EmitError> {
        let names: Vec<_> = self
            .phases()
            .flat_map(|p| &p.symbols)
            .filter(|s| !s.parameter)
            .map(|s| s.name.as_str())
            .collect();
        self.code(&code::tokenize(code, &names)?)
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
