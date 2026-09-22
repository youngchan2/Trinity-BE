//! Tile coordinates, pointer offsets, address masks and global load/store output.
use super::super::shape::constant;
use super::super::{KernelPlan, TritonPlan};
use super::context::{CodegenContext, EmittedValue};
use crate::analysis::*;

impl TritonPlan {
    pub(super) fn index(&self, expr: &IndexExpr) -> String {
        match expr {
            IndexExpr::LoopVar(id) => self.loop_name(*id),
            IndexExpr::Apply(op, args) => format!(
                "({} {} {})",
                self.index(&args[0]),
                if op == "/" { "//" } else { op },
                self.index(&args[1])
            ),
            IndexExpr::Symbol(s)
                if (self
                    .common
                    .metadata
                    .dimensions
                    .contains_key(self.canonical_symbol(s))
                    || self
                        .metadata
                        .candidates
                        .contains_key(self.canonical_symbol(s))) =>
            {
                self.parameter(s)
            }
            _ => constant(expr, &self.options).unwrap().to_string(),
        }
    }

    fn width(&self, dim: &IndexDim, fallback: usize) -> String {
        if let IndexDim::Tile { width, .. } | IndexDim::ConstTile { width, .. } = dim {
            let value = self.index(width);
            return if self.power_of_two_width(width) {
                value
            } else {
                format!("triton.next_power_of_2({value})")
            };
        }
        fallback.to_string()
    }

    fn power_of_two_width(&self, width: &IndexExpr) -> bool {
        match width {
            IndexExpr::Integer(n) => *n > 0 && (*n as u64).is_power_of_two(),
            IndexExpr::Symbol(s) => self
                .metadata
                .candidates
                .get(self.canonical_symbol(s))
                .is_some_and(|values| {
                    values
                        .iter()
                        .all(|n| *n > 0 && (*n as u64).is_power_of_two())
                }),
            _ => false,
        }
    }

    pub(super) fn tile_shape(&self, id: AccessId) -> Vec<String> {
        self.analysis
            .access(id)
            .index
            .iter()
            .zip(&self.accesses[id.index()].shape)
            .enumerate()
            .map(|(axis, (dim, size))| {
                if matches!(dim, IndexDim::FullTile)
                    && self
                        .analysis
                        .access(id)
                        .view_shape
                        .as_ref()
                        .is_some_and(|s| !matches!(s[axis], IndexExpr::Integer(_)))
                {
                    format!("triton.next_power_of_2({})", self.view_shape(id)[axis])
                } else {
                    self.width(dim, *size)
                }
            })
            .collect()
    }

    pub(super) fn coordinate(&self, id: AccessId, axis: usize) -> String {
        let access = self.analysis.access(id);
        let tile = &self.accesses[id.index()];
        match &access.index[axis] {
            IndexDim::FullTile => format!("tl.arange(0, {})", self.tile_shape(id)[axis]),
            IndexDim::Elem(IndexExpr::LoopVar(scope)) => format!(
                "({} // {}) + tl.arange(0, 1)",
                self.loop_name(*scope),
                self.block(*scope)
            ),
            _ => format!(
                "{} + tl.arange(0, {})",
                self.index(&tile.axes[axis].start),
                self.width(&access.index[axis], tile.shape[axis])
            ),
        }
    }

    pub(super) fn slice(rank: usize, axis: usize) -> String {
        (0..rank)
            .map(|i| if i == axis { ":" } else { "None" })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn mask(&self, id: AccessId, w: &mut CodegenContext) -> String {
        let rank = self.analysis.access(id).index.len();
        // Conditions carry their own logical axis; do not squeeze elem axes.
        let conditions: Vec<_> = (0..rank)
            .filter_map(|axis| {
                self.dynamic_condition(id, axis).map(|s| {
                    if rank > 1 {
                        format!("({s})[{}]", Self::slice(rank, axis))
                    } else {
                        s
                    }
                })
            })
            .collect();
        if conditions.is_empty() {
            return "True".into();
        }
        let name = format!("mask_{}", w.mask);
        w.mask += 1;
        w.line(format!("{name} = {}", conditions.join(" & ")));
        name
    }

    fn address(
        &self,
        id: AccessId,
        kernel: &KernelPlan,
        w: &mut CodegenContext,
    ) -> (String, String) {
        let access = self.analysis.access(id);
        let tensor = self.tensor_name(access.tensor);
        let kernel_shape = self.kernel_shape(kernel, access.tensor);
        let rank = access.index.len();
        let same_shape = self.accesses[id.index()]
            .axes
            .iter()
            .map(|a| a.extent)
            .eq(kernel_shape.iter().copied());
        if !same_shape
            && (0..rank).any(|axis| {
                let singleton = access
                    .view_shape
                    .as_ref()
                    .is_some_and(|shape| shape[axis] == IndexExpr::Integer(1));
                !singleton && self.stride_axis(id, kernel, axis).is_none()
            })
        {
            // Reinterpret a logical row-major view through the actual strides
            // of the kernel argument. No contiguous physical-stride assumption.
            let product = |shape: &[String]| {
                if shape.is_empty() {
                    "1".into()
                } else {
                    shape.join(" * ")
                }
            };
            let shape = self.view_shape(id);
            let linear = (0..rank)
                .map(|axis| {
                    let coordinate = self.coordinate(id, axis);
                    let coordinate = if rank > 1 {
                        format!("({coordinate})[{}]", Self::slice(rank, axis))
                    } else {
                        coordinate
                    };
                    format!("({coordinate}) * ({})", product(&shape[axis + 1..]))
                })
                .collect::<Vec<_>>()
                .join(" + ");
            let linear_name = format!("linear_{}", w.offset);
            w.line(format!("{linear_name} = {linear}"));
            let base = self.view_shape(self.kernel_access(kernel, access.tensor));
            let offset = (0..base.len())
                .map(|axis| {
                    format!(
                        "(({linear_name} // ({})) % ({})) * {tensor}_stride{axis}",
                        product(&base[axis + 1..]),
                        base[axis]
                    )
                })
                .collect::<Vec<_>>()
                .join(" + ");
            let name = format!("offset_{}", w.offset);
            w.offset += 1;
            w.line(format!("{name} = {offset}"));
            return (name, self.mask(id, w));
        }
        let offset = (0..rank)
            .map(|axis| {
                let stride = if self.accesses[id.index()]
                    .axes
                    .iter()
                    .map(|a| a.extent)
                    .eq(kernel_shape.iter().copied())
                {
                    format!("{tensor}_stride{axis}")
                } else if let Some(mapped) = self.stride_axis(id, kernel, axis) {
                    format!("{tensor}_stride{mapped}")
                } else {
                    self.accesses[id.index()].axes[axis].stride.to_string()
                };
                let coordinate = self.coordinate(id, axis);
                if rank == 1 {
                    format!("({coordinate}) * {stride}")
                } else {
                    format!("({coordinate})[{}] * {stride}", Self::slice(rank, axis))
                }
            })
            .collect::<Vec<_>>()
            .join(" + ");
        let name = format!("offset_{}", w.offset);
        w.offset += 1;
        w.line(format!("{name} = {offset}"));
        let mask = self.mask(id, w);
        (name, mask)
    }

    pub(super) fn validity(&self, id: AccessId) -> Vec<Option<String>> {
        (0..self.accesses[id.index()].axes.len())
            .map(|axis| self.dynamic_condition(id, axis))
            .collect()
    }

    pub(super) fn load(
        &self,
        id: AccessId,
        kernel: &KernelPlan,
        w: &mut CodegenContext,
    ) -> EmittedValue {
        let access = self.analysis.access(id);
        if w.pending_stores.contains(&access.tensor) {
            w.synchronize_stores();
        }
        let (offset, mask) = self.address(id, kernel, w);
        let code = w.temporary(format!(
            "tl.load({}_ptr + {offset}, mask={mask}, other=0.0).to(tl.float32)",
            self.tensor_name(access.tensor)
        ));
        EmittedValue {
            code,
            shape: self.tile_shape(id),
            valid: self.validity(id),
            zero_invalid: true,
        }
    }

    pub(super) fn store(
        &self,
        id: AccessId,
        value: &str,
        kernel: &KernelPlan,
        w: &mut CodegenContext,
    ) {
        let (offset, mask) = self.address(id, kernel, w);
        let value = if value.parse::<f64>().is_ok() {
            format!("tl.full((), {value}, tl.float32)")
        } else {
            format!("({value})")
        };
        w.line(format!(
            "tl.store({}_ptr + {offset}, {value}.to(tl.{}), mask={mask})",
            self.tensor_name(self.analysis.access(id).tensor),
            self.tensor_dtype(self.analysis.access(id).tensor).python()
        ));
        w.pending_stores.insert(self.analysis.access(id).tensor);
    }

    fn dynamic_condition(&self, id: AccessId, axis: usize) -> Option<String> {
        let dim = &self.analysis.access(id).index[axis];
        let info = &self.accesses[id.index()].axes[axis];
        let extent = self.view_shape(id)[axis].clone();
        if matches!(dim, IndexDim::FullTile)
            && extent.parse::<usize>().is_ok_and(usize::is_power_of_two)
        {
            return None;
        }
        let coordinate = self.coordinate(id, axis);
        let mut predicate = format!("(({coordinate}) >= 0) & (({coordinate}) < {extent})");
        if let Some(end) = &info.loop_end {
            let end = self.index(end);
            if end != extent {
                predicate.push_str(&format!(" & (({coordinate}) < {end})"));
            }
        }
        if let IndexDim::Tile { width, .. } | IndexDim::ConstTile { width, .. } = dim
            && !self.power_of_two_width(width)
        {
            predicate.push_str(&format!(
                " & (tl.arange(0, {}) < {})",
                self.tile_shape(id)[axis],
                self.index(width)
            ));
        }
        Some(predicate)
    }

    fn stride_axis(&self, id: AccessId, kernel: &KernelPlan, axis: usize) -> Option<usize> {
        let a = self.analysis.access(id);
        let representative = self.kernel_access(kernel, a.tensor);
        let b = self.analysis.access(representative);
        let stable_axes = |access: AccessId, info: &AccessInfo| -> Vec<usize> {
            (0..info.index.len())
                .filter(|i| {
                    info.view_shape
                        .as_ref()
                        .map(|s| s[*i] != IndexExpr::Integer(1))
                        .unwrap_or(self.accesses[access.index()].axes[*i].extent != 1)
                })
                .collect()
        };
        let aa = stable_axes(id, a);
        let bb = stable_axes(representative, b);
        if aa.len() != bb.len()
            || aa.iter().zip(&bb).any(|(x, y)| {
                self.accesses[id.index()].axes[*x].extent
                    != self.accesses[representative.index()].axes[*y].extent
            })
        {
            return None;
        }
        aa.iter().position(|i| *i == axis).map(|i| bb[i])
    }
}
