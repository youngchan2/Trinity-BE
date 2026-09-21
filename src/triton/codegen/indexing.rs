//! Tile coordinates, pointer offsets, address masks and global load/store output.
use super::super::shape::{constant, loop_range};
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
                if self.options.managed
                    && (self
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
            IndexExpr::Symbol(s) if !self.options.managed => {
                if let Some((i, _)) =
                    self.analysis
                        .scopes()
                        .iter()
                        .enumerate()
                        .find(|(_, scope)| {
                            scope
                                .loop_info
                                .as_ref()
                                .is_some_and(|info| info.step == IndexExpr::Symbol(s.clone()))
                        })
                {
                    self.block(ScopeId(i))
                } else {
                    constant(expr, &self.options).unwrap().to_string()
                }
            }
            _ => constant(expr, &self.options).unwrap().to_string(),
        }
    }

    fn width(&self, dim: &IndexDim, fallback: usize) -> String {
        if self.options.managed
            && let IndexDim::Tile { width, .. } | IndexDim::ConstTile { width, .. } = dim
        {
            let value = self.index(width);
            return if self.power_of_two_width(width) {
                value
            } else {
                format!("triton.next_power_of_2({value})")
            };
        }
        match dim {
            IndexDim::Tile {
                start: IndexExpr::LoopVar(id),
                width,
            } if *width == self.analysis.scope(*id).loop_info.as_ref().unwrap().step
                && !matches!(width, IndexExpr::Integer(n) if !(*n as usize).is_power_of_two()) =>
            {
                self.block(*id)
            }
            _ => fallback.to_string(),
        }
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
                if self.options.managed
                    && matches!(dim, IndexDim::FullTile)
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
        if self.dynamic_access(id) {
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
            return name;
        }
        let access = self.analysis.access(id);
        let tile = &self.accesses[id.index()];
        let mut masks = Vec::new();
        for (axis, dim) in access.index.iter().enumerate() {
            let info = &tile.axes[axis];
            if matches!(dim, IndexDim::FullTile) && info.extent.is_power_of_two() {
                continue;
            }
            if matches!(dim, IndexDim::ConstTile { .. })
                && info.width.is_power_of_two()
                && constant(&info.start, &self.options)
                    .is_ok_and(|start| start >= 0 && start as usize + info.width <= info.extent)
            {
                continue;
            }
            let coordinate = if let IndexDim::Tile {
                start: IndexExpr::LoopVar(scope),
                ..
            } = dim
            {
                let width = self.width(dim, tile.shape[axis]);
                let suffix = if width == self.block(*scope) {
                    String::new()
                } else {
                    format!("_{width}")
                };
                let name = format!("{}_indices{suffix}", self.loop_name(*scope));
                if w.indices.insert(name.clone()) {
                    w.line(format!("{name} = {}", self.coordinate(id, axis)));
                }
                name
            } else if let IndexDim::Elem(IndexExpr::LoopVar(scope)) = dim {
                let name = format!("elem_{}_indices", self.loop_name(*scope));
                if w.indices.insert(name.clone()) {
                    w.line(format!("{name} = {}", self.coordinate(id, axis)));
                }
                name
            } else {
                format!("({})", self.coordinate(id, axis))
            };
            let end = if self.padded(id) {
                info.loop_end
                    .as_ref()
                    .map(|e| constant(e, &self.options).unwrap() as usize)
                    .unwrap_or(info.extent)
                    .min(info.extent)
            } else {
                info.extent
            };
            let mut condition = format!("({coordinate} < {end})");
            if !info.width.is_power_of_two() {
                condition.push_str(&format!(
                    " & (tl.arange(0, {}) < {})",
                    tile.shape[axis], info.width
                ));
            }
            if access.index.len() > 1 {
                condition = format!("({condition})[{}]", Self::slice(access.index.len(), axis));
            }
            masks.push(condition);
        }
        if masks.is_empty() {
            return "True".into();
        }
        let name = format!("mask_{}", w.mask);
        w.mask += 1;
        w.line(format!("{name} = {}", masks.join(" & ")));
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
        if self.options.managed
            && !same_shape
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

    fn padded(&self, id: AccessId) -> bool {
        (0..self.analysis.access(id).index.len()).any(|axis| self.axis_padded(id, axis))
    }

    fn axis_padded(&self, id: AccessId, axis: usize) -> bool {
        if self.dynamic_access(id) {
            return self.dynamic_condition(id, axis).is_some();
        }
        let access = self.analysis.access(id);
        let tile = &self.accesses[id.index()];
        let dim = &access.index[axis];
        let info = &tile.axes[axis];
        let width = if let IndexDim::Tile {
            start: IndexExpr::LoopVar(scope),
            width,
        } = dim
        {
            if self.tunable(*scope)
                && *width == self.analysis.scope(*scope).loop_info.as_ref().unwrap().step
            {
                128.min(loop_range(&self.analysis, *scope, &self.options).unwrap().1 as usize)
                    .next_power_of_two()
            } else {
                info.width
            }
        } else {
            info.width
        };
        !info.width.is_power_of_two()
            || !info.extent.is_multiple_of(width)
            || info.loop_end.as_ref().is_some_and(|e| {
                !(constant(e, &self.options).unwrap() as usize).is_multiple_of(width)
            })
    }

    pub(super) fn validity(&self, id: AccessId) -> Vec<Option<String>> {
        if self.dynamic_access(id) {
            return (0..self.accesses[id.index()].axes.len())
                .map(|axis| self.dynamic_condition(id, axis))
                .collect();
        }
        let tile = &self.accesses[id.index()];
        tile.axes
            .iter()
            .enumerate()
            .map(|(axis, info)| {
                if !self.axis_padded(id, axis) {
                    return None;
                }
                let end = info
                    .loop_end
                    .as_ref()
                    .map(|e| constant(e, &self.options).unwrap() as usize)
                    .unwrap_or(info.extent)
                    .min(info.extent);
                let mut predicate = format!("({} < {end})", self.coordinate(id, axis));
                if !info.width.is_power_of_two() {
                    predicate.push_str(&format!(
                        " & (tl.arange(0, {}) < {})",
                        tile.shape[axis], info.width
                    ));
                }
                Some(predicate)
            })
            .collect()
    }

    pub(super) fn load(
        &self,
        id: AccessId,
        kernel: &KernelPlan,
        w: &mut CodegenContext,
    ) -> EmittedValue {
        let access = self.analysis.access(id);
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
    }

    fn dynamic_access(&self, id: AccessId) -> bool {
        self.options.managed
            || self.accesses[id.index()].axes.iter().any(|a| {
                a.loop_end
                    .as_ref()
                    .is_some_and(|e| !e.loop_dependencies().is_empty())
            })
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
