//! Native port/access representation derived from common TensorAccess contracts.

use crate::emit::provider::{KernelContext, ProviderError};

fn unsupported(message: &str) -> ProviderError {
    ProviderError::Unsupported(message.into())
}
use crate::{AccessIndex, DType, IndexExpr, Storage, TensorAccess, TileWidth, ValueInstanceId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) struct Access {
    pub value: ValueInstanceId,
    pub dtype: DType,
    pub storage: Storage,
    pub value_shape: Vec<usize>,
    /// Logical shape of this access, used for contiguous addressing and tiles.
    pub shape: Vec<usize>,
    pub axes: Vec<Axis>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) enum Axis {
    Full,
    Tile {
        variable: String,
        width: usize,
        clipped: bool,
    },
    Element {
        variable: String,
        step: IndexExpr,
    },
}

impl Access {
    pub(in crate::emit) fn from_plan(
        context: &KernelContext<'_, '_>,
        access: &TensorAccess,
    ) -> Result<Self, ProviderError> {
        let value = context
            .prepared
            .plan
            .value_instance(access.value)
            .ok_or_else(|| ProviderError::Failed("missing value".into()))?;

        let shape = access.shape(value.shape());
        if shape
            .iter()
            .any(|&size| size == 0 || size > i64::MAX as usize)
        {
            return Err(unsupported("kernel requires positive i64 extents"));
        }

        let axes = access
            .indices
            .iter()
            .map(|index| {
                Ok(match index {
                    AccessIndex::Slice { .. } | AccessIndex::Element(_) => {
                        return Err(unsupported(
                            "CuTe template requires loop-coordinate access indices",
                        ));
                    }
                    AccessIndex::FullTile => Axis::Full,
                    AccessIndex::Tile { variable, width }
                    | AccessIndex::ClippedTile { variable, width } => {
                        let width = width
                            .resolve(&Default::default())
                            .map_err(ProviderError::Unsupported)?;

                        Axis::Tile {
                            variable: variable.clone(),
                            width,
                            clipped: matches!(index, AccessIndex::ClippedTile { .. }),
                        }
                    }
                    AccessIndex::Elem(variable) => {
                        let loop_ = context
                            .loops
                            .iter()
                            .find(|loop_| loop_.domain.variable == *variable)
                            .ok_or_else(|| unsupported("index needs an enclosing loop"))?;

                        Axis::Element {
                            variable: variable.clone(),
                            step: loop_.domain.step.clone(),
                        }
                    }
                })
            })
            .collect::<Result<_, ProviderError>>()?;

        Ok(Self {
            value: access.value,
            dtype: value.dtype(),
            storage: value.storage(),
            value_shape: value.shape().to_vec(),
            shape: shape.to_vec(),
            axes,
        })
    }
}

impl Access {
    pub(in crate::emit) fn width(&self, axis: usize) -> usize {
        match self.axes[axis] {
            Axis::Full => self.shape[axis],
            Axis::Tile { width, .. } => width,
            Axis::Element { .. } => 1,
        }
    }

    pub(in crate::emit) fn matches(&self, source: &TensorAccess) -> bool {
        self.value == source.value
            && self.shape == source.shape(&self.value_shape)
            && self.axes.len() == source.indices.len()
            && self
                .axes
                .iter()
                .zip(&source.indices)
                .all(|(axis, index)| match (axis, index) {
                    (Axis::Full, AccessIndex::FullTile) => true,
                    (
                        Axis::Tile {
                            variable,
                            width,
                            clipped,
                        },
                        AccessIndex::Tile {
                            variable: v,
                            width: w,
                        },
                    ) => !clipped && variable == v && *w == TileWidth::Constant(*width),
                    (
                        Axis::Tile {
                            variable,
                            width,
                            clipped,
                        },
                        AccessIndex::ClippedTile {
                            variable: v,
                            width: w,
                        },
                    ) => *clipped && variable == v && *w == TileWidth::Constant(*width),
                    (Axis::Element { variable, .. }, AccessIndex::Elem(v)) => variable == v,
                    _ => false,
                })
    }
}
