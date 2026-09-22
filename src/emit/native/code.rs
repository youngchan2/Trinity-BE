//! Native code with lexical output slots; no parsing of generated CUDA is required.

use super::KernelBindings;
use crate::IndexExpr;
use crate::emit::provider::ProviderError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) struct RegisterBinding {
    /// An element already converted to the producing operation's declared dtype.
    pub value: String,
    /// Absolute tensor coordinates of this element, on the current thread.
    pub coordinates: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::emit) enum KernelCode {
    Text(String),
    Sequence(Vec<Self>),
    Scope {
        before: String,
        body: Box<Self>,
        after: String,
    },
    /// Continuations run while the producer's local variables are alive.
    /// The enclosing scope already guards invalid elements and inactive lanes.
    Output {
        port: usize,
        binding: RegisterBinding,
    },
}

impl From<String> for KernelCode {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl KernelCode {
    /// Render an unconnected body. Output slots are empty unless a continuation is supplied.
    #[cfg(test)]
    pub fn source(&self) -> String {
        self.connect(&mut |_, _| Ok(String::new()))
            .expect("empty continuation")
    }

    pub fn connect(
        &self,
        output: &mut impl FnMut(usize, &RegisterBinding) -> Result<String, ProviderError>,
    ) -> Result<String, ProviderError> {
        Ok(match self {
            Self::Text(text) => text.clone(),
            Self::Sequence(nodes) => {
                let mut text = String::new();
                for node in nodes {
                    text.push_str(&node.connect(output)?);
                }
                text
            }
            Self::Scope {
                before,
                body,
                after,
            } => format!("{before}{}{after}", body.connect(output)?),
            Self::Output { port, binding } => output(*port, binding)?,
        })
    }
}

pub(in crate::emit) fn render_index_expression(
    expression: &IndexExpr,
    bindings: &KernelBindings,
) -> Result<String, ProviderError> {
    let (a, operator, b) = match expression {
        IndexExpr::Constant(value) => return Ok(format!("int64_t({value})")),
        IndexExpr::Variable(name) => {
            return bindings
                .indices
                .get(name)
                .map(|v| format!("({v})"))
                .ok_or_else(|| ProviderError::Failed(format!("missing loop binding {name}")));
        }
        IndexExpr::Add(a, b) => (a, "+", b),
        IndexExpr::Sub(a, b) => (a, "-", b),
        IndexExpr::Mul(a, b) => (a, "*", b),
        IndexExpr::Div(a, b) => (a, "/", b),
    };
    Ok(format!(
        "({} {operator} {})",
        render_index_expression(a, bindings)?,
        render_index_expression(b, bindings)?
    ))
}
