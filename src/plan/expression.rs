//! Typed computation and memory accesses owned by the physical plan.

use super::ValueInstanceId;
use std::collections::BTreeMap;

/// A scalar literal. Floating-point bits preserve signed zero and NaN payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Constant {
    Integer(i64),
    Float32(u32),
}

/// An access along one tensor axis. Tile width is independent of the loop step.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessIndex {
    FullTile,
    Tile {
        variable: String,
        width: usize,
    },
    ClippedTile {
        variable: String,
        width: usize,
    },
    /// Selects the element at the loop coordinate divided by its step.
    Elem(String),
}

impl AccessIndex {
    pub(crate) fn variable(&self) -> Option<&str> {
        match self {
            Self::FullTile => None,
            Self::Tile { variable, .. }
            | Self::ClippedTile { variable, .. }
            | Self::Elem(variable) => Some(variable),
        }
    }

    fn rename(&mut self, names: &BTreeMap<String, String>) {
        let variable = match self {
            Self::FullTile => return,
            Self::Tile { variable, .. }
            | Self::ClippedTile { variable, .. }
            | Self::Elem(variable) => variable,
        };
        if let Some(name) = names.get(variable) {
            *variable = name.clone();
        }
    }
}

/// A value reference and its accesses in tensor-axis order.
/// Dtype, shape and storage belong to the referenced ValueInstance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorAccess {
    pub value: ValueInstanceId,
    pub indices: Box<[AccessIndex]>,
}

impl TensorAccess {
    pub fn new(value: ValueInstanceId, indices: impl IntoIterator<Item = AccessIndex>) -> Self {
        Self {
            value,
            indices: indices.into_iter().collect(),
        }
    }
}

/// Computation and memory effects with resolved operands and constants.
/// An Operation contains a Store or AllGather; their computation children are pure.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expression {
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
            Self::Constant(_) | Self::Load(_) | Self::AllGather { .. } => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Self] {
        match self {
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
            Self::Constant(_) | Self::Load(_) | Self::AllGather { .. } => &mut [],
        }
    }

    pub(crate) fn accesses(&self) -> Vec<&TensorAccess> {
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

    fn map_accesses(&mut self, visit: &mut impl FnMut(&mut TensorAccess)) {
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
        self.map_accesses(&mut |access| {
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
