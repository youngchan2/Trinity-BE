//! Typed computation and memory accesses owned by the physical plan.

use super::{IndexExpr, ValueInstanceId};
use std::collections::BTreeMap;

/// A scalar literal. Floating-point bits preserve signed zero and NaN payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Constant {
    Integer(i64),
    Float32(u32),
    Float64(u64),
}

/// A fixed tile width or a configuration parameter, resolved before native emit.
/// This is not a loop-coordinate expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TileWidth {
    Constant(usize),
    Symbol(String),
}

impl TileWidth {
    pub fn resolve(&self, bindings: &BTreeMap<String, i64>) -> Result<usize, String> {
        let width = match self {
            Self::Constant(width) => *width,
            Self::Symbol(name) => usize::try_from(
                *bindings
                    .get(name)
                    .ok_or_else(|| format!("unbound tile width {name}"))?,
            )
            .map_err(|_| format!("tile width {name} must be positive"))?,
        };
        if width == 0 || width > i64::MAX as usize {
            return Err("tile width must be a positive i64 extent".into());
        }
        Ok(width)
    }
}

impl From<usize> for TileWidth {
    fn from(width: usize) -> Self {
        Self::Constant(width)
    }
}
impl From<String> for TileWidth {
    fn from(name: String) -> Self {
        Self::Symbol(name)
    }
}
impl From<&str> for TileWidth {
    fn from(name: &str) -> Self {
        Self::Symbol(name.into())
    }
}

pub(super) fn is_symbol(name: &str) -> bool {
    let name = name.strip_prefix('?').unwrap_or(name);
    name.as_bytes()
        .first()
        .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// An access along one tensor axis. Tile width is independent of the loop step.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessIndex {
    /// A literal/arithmetic slice origin, rather than a loop-coordinate tile.
    Slice {
        start: IndexExpr,
        width: TileWidth,
    },
    /// A scalar index expression; plain loop-variable ordinals use Elem.
    Element(IndexExpr),
    FullTile,
    Tile {
        variable: String,
        width: TileWidth,
    },
    ClippedTile {
        variable: String,
        width: TileWidth,
    },
    /// Selects the element at the loop coordinate divided by its step.
    Elem(String),
}

impl AccessIndex {
    pub(crate) fn variable(&self) -> Option<&str> {
        match self {
            Self::FullTile | Self::Slice { .. } | Self::Element(_) => None,
            Self::Tile { variable, .. }
            | Self::ClippedTile { variable, .. }
            | Self::Elem(variable) => Some(variable),
        }
    }

    fn rename(&mut self, names: &BTreeMap<String, String>) {
        match self {
            Self::Slice { start, .. } | Self::Element(start) => {
                start.rename(names);
                return;
            }
            _ => {}
        }
        let variable = match self {
            Self::FullTile | Self::Slice { .. } | Self::Element(_) => return,
            Self::Tile { variable, .. }
            | Self::ClippedTile { variable, .. }
            | Self::Elem(variable) => variable,
        };
        if let Some(name) = names.get(variable) {
            *variable = name.clone();
        }
    }
}

/// A value reference and its accesses in view-axis order.
/// Dtype, allocation shape and storage belong to the referenced ValueInstance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorAccess {
    pub value: ValueInstanceId,
    /// Contiguous row-major reinterpretation of the same storage; no data movement.
    /// None uses ValueInstance::shape(). Views must preserve the element count.
    pub view_shape: Option<Box<[usize]>>,
    /// Parametric view before specialization (e.g. split scratch dimensions).
    pub view_dimensions: Option<Box<[IndexExpr]>>,
    pub indices: Box<[AccessIndex]>,
}

impl TensorAccess {
    pub fn new(value: ValueInstanceId, indices: impl IntoIterator<Item = AccessIndex>) -> Self {
        Self {
            value,
            view_shape: None,
            view_dimensions: None,
            indices: indices.into_iter().collect(),
        }
    }

    pub fn with_view_shape(mut self, shape: impl IntoIterator<Item = usize>) -> Self {
        self.view_shape = Some(shape.into_iter().collect());
        self
    }

    pub fn shape<'a>(&'a self, value_shape: &'a [usize]) -> &'a [usize] {
        self.view_shape.as_deref().unwrap_or(value_shape)
    }

    pub(super) fn validate_view(&self, value_shape: &[usize]) -> Result<(), String> {
        if self.indices.len() != self.shape(value_shape).len() {
            return Err("access rank differs from view or value".into());
        }
        if let Some(view) = &self.view_shape
            && (view.is_empty() || element_count(view)? != element_count(value_shape)?)
        {
            return Err("view element count differs from value".into());
        }
        Ok(())
    }
}

/// Pure extended-IR operators. Arguments retain source order, including axes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueOp {
    Exp,
    Erf,
    Abs,
    Transpose,
    Permute,
    ReduceMax,
    ReduceMin,
    Squeeze,
    Concat,
    LessEqual,
    Maximum,
    Minimum,
    Cast(String),
    ReduceSum,
    Broadcast,
    Unsqueeze,
}

pub(super) fn element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |n, &size| {
        n.checked_mul(size)
            .filter(|&n| n > 0 && n <= i64::MAX as usize)
            .ok_or_else(|| "view extent/product must be positive and fit i64".into())
    })
}

/// Computation and memory effects with resolved operands and constants.
/// An Operation contains a Store or AllGather; their computation children are pure.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expression {
    Index(IndexExpr),
    Apply {
        op: ValueOp,
        args: Box<[Self]>,
    },
    Constant(Constant),
    Load(TensorAccess),
    Store {
        destination: TensorAccess,
        value: Box<Self>,
    },
    Add(Box<[Self; 2]>),
    Sub(Box<[Self; 2]>),
    Mul(Box<[Self; 2]>),
    Div(Box<[Self; 2]>),
    Matmul(Box<[Self; 2]>),
    Sqr(Box<Self>),
    Sqrt(Box<Self>),
    Sigmoid(Box<Self>),
    Relu(Box<Self>),
    ReduceSum {
        value: Box<Self>,
        axis: usize,
    },
    Broadcast {
        value: Box<Self>,
        axis: usize,
    },
    Unsqueeze {
        value: Box<Self>,
        axis: usize,
    },
    AllGather {
        source: TensorAccess,
        destination: TensorAccess,
        axis: usize,
    },
}

impl Expression {
    pub(crate) fn children(&self) -> &[Self] {
        match self {
            Self::Apply { args, .. } => args,
            Self::Add(values)
            | Self::Sub(values)
            | Self::Mul(values)
            | Self::Div(values)
            | Self::Matmul(values) => values.as_slice(),
            Self::Store { value, .. }
            | Self::Sqr(value)
            | Self::Sqrt(value)
            | Self::Sigmoid(value)
            | Self::Relu(value)
            | Self::ReduceSum { value, .. }
            | Self::Broadcast { value, .. }
            | Self::Unsqueeze { value, .. } => std::slice::from_ref(value),
            Self::Index(_) | Self::Constant(_) | Self::Load(_) | Self::AllGather { .. } => &[],
        }
    }

    pub(crate) fn children_mut(&mut self) -> &mut [Self] {
        match self {
            Self::Apply { args, .. } => args,
            Self::Add(values)
            | Self::Sub(values)
            | Self::Mul(values)
            | Self::Div(values)
            | Self::Matmul(values) => values.as_mut_slice(),
            Self::Store { value, .. }
            | Self::Sqr(value)
            | Self::Sqrt(value)
            | Self::Sigmoid(value)
            | Self::Relu(value)
            | Self::ReduceSum { value, .. }
            | Self::Broadcast { value, .. }
            | Self::Unsqueeze { value, .. } => std::slice::from_mut(value),
            Self::Index(_) | Self::Constant(_) | Self::Load(_) | Self::AllGather { .. } => &mut [],
        }
    }

    /// Memory accesses in expression-tree order, with a store's destination first.
    pub fn accesses(&self) -> Vec<&TensorAccess> {
        let mut accesses = match self {
            Self::Load(access)
            | Self::Store {
                destination: access,
                ..
            } => vec![access],
            Self::AllGather {
                source,
                destination,
                ..
            } => vec![source, destination],
            _ => vec![],
        };
        accesses.extend(self.children().iter().flat_map(Self::accesses));
        accesses
    }

    /// Distinct read values in expression order, before applying accumulation semantics.
    pub(crate) fn reads(&self) -> Vec<ValueInstanceId> {
        let mut values = match self {
            Self::Load(access) | Self::AllGather { source: access, .. } => vec![access.value],
            _ => vec![],
        };

        for value in self.children().iter().flat_map(Self::reads) {
            if !values.contains(&value) {
                values.push(value);
            }
        }
        values
    }

    pub(crate) fn is_pure(&self) -> bool {
        !matches!(self, Self::Store { .. } | Self::AllGather { .. })
            && self.children().iter().all(Self::is_pure)
    }

    pub(super) fn map_accesses(&mut self, visit: &mut impl FnMut(&mut TensorAccess)) {
        match self {
            Self::Load(access)
            | Self::Store {
                destination: access,
                ..
            } => visit(access),
            Self::AllGather {
                source,
                destination,
                ..
            } => {
                visit(source);
                visit(destination);
            }
            _ => {}
        }
        for child in self.children_mut() {
            child.map_accesses(visit);
        }
    }

    pub(crate) fn remap_values(&mut self, ids: &[usize]) {
        self.map_accesses(&mut |access| {
            access.value = ValueInstanceId::from_index(ids[access.value.index()])
        });
    }

    pub(crate) fn rename_indices(&mut self, names: &BTreeMap<String, String>) {
        fn indices(expr: &mut Expression, names: &BTreeMap<String, String>) {
            if let Expression::Index(i) = expr {
                i.rename(names);
            }
            for child in expr.children_mut() {
                indices(child, names);
            }
        }
        indices(self, names);
        self.map_accesses(&mut |access| {
            if let Some(shape) = &mut access.view_dimensions {
                for dim in shape {
                    dim.rename(names);
                }
            }
            for index in &mut access.indices {
                index.rename(names);
            }
        });
    }
}

/// Recognizes the existing implicit-zero reduction convention within a serial loop.
pub(crate) fn accumulation_rhs<'a>(
    store: &'a Expression,
    variable: &str,
) -> Option<&'a Expression> {
    // The root must store the updated value.
    let Expression::Store { destination, value } = store else {
        return None;
    };

    // The stored value must be a sum: load(destination) + rhs.
    let Expression::Add(values) = value.as_ref() else {
        return None;
    };

    // The left operand must be a load; its address is checked below.
    let Expression::Load(source) = &values[0] else {
        return None;
    };

    (source == destination
        && !destination
            .indices
            .iter()
            .any(|index| index.variable() == Some(variable)))
    .then_some(&values[1])
}
