use super::Statement;
use std::collections::BTreeMap;

/// An S-expression tree describing an operation in the physical plan.
///
/// Atoms hold operator names, identifiers, or literal values as strings.
/// By convention, a list starts with an operator atom followed by its
/// arguments, which may themselves be nested expressions. These expressions
/// describe computations, memory accesses, views, and indices.
///
/// For example, `(tile i 32)` is represented as
/// `List(vec![Atom("tile".into()), Atom("i".into()), Atom("32".into())])`.
///
/// This type stores structure only: it does not enforce operator names,
/// argument counts, or literal types. Consumers interpret and validate
/// expressions according to the operators they support.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expression {
    /// A leaf containing an operator name, identifier, or literal value.
    Atom(String),
    /// An ordered sequence of expressions, conventionally an operator and its arguments.
    List(Vec<Expression>),
}

impl Expression {
    pub fn atom(&self) -> Option<&str> {
        match self {
            Self::Atom(s) => Some(s),
            _ => None,
        }
    }

    pub fn list(&self) -> Option<&[Self]> {
        match self {
            Self::List(xs) => Some(xs),
            _ => None,
        }
    }

    pub fn operator(&self) -> Option<&str> {
        self.list()?.first()?.atom()
    }

    pub(crate) fn rename_indices(&mut self, names: &BTreeMap<String, String>) {
        if let Self::List(xs) = self {
            if matches!(
                xs.first().and_then(Self::atom),
                Some("tile" | "clipped_tile" | "elem")
            ) && let Some(Self::Atom(name)) = xs.get_mut(1)
                && let Some(new) = names.get(name)
            {
                *name = new.clone();
            }
            for x in xs {
                x.rename_indices(names);
            }
        }
    }

    /// Axis labels are local to a view. Normalize them by layout position while
    /// retaining axis order, index expressions and tile widths.
    pub(crate) fn normalize_axes(&mut self) {
        if let Self::List(xs) = self {
            if matches!(xs.first().and_then(Self::atom), Some("load" | "store")) {
                let mut axes = BTreeMap::new();
                if let Some(Self::List(view)) = xs.get_mut(1)
                    && let Some(Self::List(layout)) = view.get_mut(2)
                {
                    for (i, axis) in layout.iter_mut().skip(1).enumerate() {
                        if let Self::List(axis) = axis
                            && let Some(Self::Atom(name)) = axis.get_mut(1)
                        {
                            let canonical = format!("axis{i}");
                            axes.insert(name.clone(), canonical.clone());
                            *name = canonical;
                        }
                    }
                }

                if let Some(Self::List(index)) = xs.last_mut() {
                    for slot in index.iter_mut().skip(1) {
                        if let Self::List(slot) = slot
                            && let Some(Self::Atom(name)) = slot.get_mut(1)
                        {
                            if let Some(canonical) = axes.get(name) {
                                *name = canonical.clone();
                            } else {
                                *name = format!("unbound_axis:{name}");
                            }
                        }
                    }
                }
            }
            for x in xs {
                x.normalize_axes();
            }
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

/// Recognize a reduction update whose destination is invariant in this serial scope.
pub(crate) fn accumulation_rhs<'a>(
    store: &'a Expression,
    variable: &str,
) -> Option<&'a Expression> {
    let a = store.list()?;
    if store.operator() != Some("store") || a.len() != 4 {
        return None;
    }

    let plus = a[2].list()?;
    if a[2].operator() != Some("+") || plus.len() != 3 {
        return None;
    }

    let load = plus[1].list()?;
    if plus[1].operator() != Some("load") || load.len() != 3 || load[1] != a[1] || load[2] != a[3] {
        return None;
    }

    fn uses(e: &Expression, var: &str) -> bool {
        (matches!(e.operator(), Some("tile" | "clipped_tile" | "elem"))
            && e.list().and_then(|a| a.get(1)).and_then(Expression::atom) == Some(var))
            || e.list().is_some_and(|xs| xs.iter().any(|e| uses(e, var)))
    }

    (!uses(&a[3], variable)).then_some(&plus[2])
}
