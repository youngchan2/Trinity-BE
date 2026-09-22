use super::OperationId;
use std::collections::BTreeMap;

/// One node in the ordered physical program, including nested loop statements.
///
/// CUDA lowering determines body and task boundaries from this structure; a
/// statement does not itself define a kernel launch or one task.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Statement {
    /// One scheduled source kernel region; its internal sequence is indivisible
    /// when offered as an opaque implementation candidate.
    Region(Vec<Statement>),
    Loop(Loop),
    Operation(OperationId),
}

impl Statement {
    pub fn operations(&self) -> Vec<OperationId> {
        match self {
            Self::Region(body) => body.iter().flat_map(Self::operations).collect(),
            Self::Operation(id) => vec![*id],
            Self::Loop(loop_) => loop_.body.iter().flat_map(Self::operations).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum IndexExpr {
    Constant(i64),
    Variable(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
}

impl IndexExpr {
    pub fn evaluate(&self, bindings: &BTreeMap<String, i64>) -> Result<i64, String> {
        let binary = |a: &Self, b: &Self, f: fn(i64, i64) -> Option<i64>| {
            f(a.evaluate(bindings)?, b.evaluate(bindings)?)
                .ok_or_else(|| "invalid/overflowing loop bound".into())
        };

        match self {
            Self::Constant(n) => Ok(*n),
            Self::Variable(v) => bindings
                .get(v)
                .copied()
                .ok_or_else(|| format!("unbound loop variable {v}")),
            Self::Add(a, b) => binary(a, b, i64::checked_add),
            Self::Sub(a, b) => binary(a, b, i64::checked_sub),
            Self::Mul(a, b) => binary(a, b, i64::checked_mul),
            Self::Div(a, b) => binary(a, b, i64::checked_div),
        }
    }

    pub(crate) fn rename(&mut self, names: &BTreeMap<String, String>) {
        match self {
            Self::Variable(v) => {
                if let Some(new) = names.get(v) {
                    *v = new.clone();
                }
            }

            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) | Self::Div(a, b) => {
                a.rename(names);
                b.rename(names);
            }
            Self::Constant(_) => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoopKind {
    /// Parallel split binding of a normalized mloop.
    Split,
    Parallel,
    Sequential,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct LoopDomain {
    pub variable: String,
    pub start: IndexExpr,
    pub stop: IndexExpr,
    pub step: IndexExpr,
}

impl LoopDomain {
    pub(crate) fn bounds(
        &self,
        bindings: &BTreeMap<String, i64>,
    ) -> Result<(i64, i64, i64), String> {
        let start = self.start.evaluate(bindings)?;
        let stop = self.stop.evaluate(bindings)?;
        let step = self.step.evaluate(bindings)?;

        if start < 0 || stop <= start || step <= 0 || (stop - start) % step != 0 {
            return Err(format!(
                "loop {} requires a nonempty, nonnegative, divisible range; got {start}..{stop} step {step}",
                self.variable
            ));
        }

        Ok((start, stop, step))
    }

    pub fn values(&self, bindings: &BTreeMap<String, i64>) -> Result<Vec<i64>, String> {
        let (start, stop, step) = self.bounds(bindings)?;
        Ok((start..stop)
            .step_by(usize::try_from(step).map_err(|_| "loop step exceeds usize")?)
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Loop {
    pub kind: LoopKind,
    pub domain: LoopDomain,
    pub body: Vec<Statement>,
}
