//! Read an earlier register definition through a compatible view or sub-tile.
use super::super::{KernelPlan, ProgramPlan};
use super::context::{CodegenContext, EmittedValue, tuple};
use crate::analysis::*;

impl ProgramPlan {
    pub(super) fn local_load(
        &self,
        id: AccessId,
        kernel: &KernelPlan,
        w: &mut CodegenContext,
    ) -> EmittedValue {
        let access = self.analysis.access(id);
        let mut code = self.tensor_name(access.tensor).to_owned();
        if let Some(binding) = kernel.local_reads.get(&id) {
            let definition = self.analysis.access(binding.definition);
            if definition.index != access.index || definition.view_shape != access.view_shape {
                let source_shape = self.tile_shape(binding.definition);
                let target_shape = self.tile_shape(id);
                let mut shape: Vec<_> = binding
                    .axes
                    .iter()
                    .map(|(a, _)| source_shape[*a].clone())
                    .collect();
                code = w.temporary(format!("tl.reshape({code}, {})", tuple(&shape)));
                for (axis, (source, target)) in binding.axes.iter().enumerate() {
                    if definition.index[*source] == access.index[*target] {
                        continue;
                    }
                    let start = self.index(&self.accesses[id.index()].axes[*target].start);
                    let origin =
                        self.index(&self.accesses[binding.definition.index()].axes[*source].start);
                    let extent = shape[axis].clone();
                    shape[axis] = target_shape[*target].clone();
                    let index = if shape[axis] == "1" {
                        format!(
                            "tl.full({}, ({start}) - ({origin}), tl.int32)",
                            tuple(&shape)
                        )
                    } else {
                        format!(
                            "tl.broadcast_to((({start}) - ({origin}) + tl.arange(0, {}))[{}], {})",
                            shape[axis],
                            Self::slice(shape.len(), axis),
                            tuple(&shape)
                        )
                    };
                    // Padded lanes are represented by validity predicates. Keep
                    // their gather indices in bounds before those predicates act.
                    let index =
                        w.temporary(format!("tl.minimum(tl.maximum({index}, 0), {extent} - 1)"));
                    code = w.temporary(format!("tl.gather({code}, {index}, axis={axis})"));
                }
                code = w.temporary(format!("tl.reshape({code}, {})", tuple(&target_shape)));
            }
        }
        let valid = self.validity(id);
        let mut zero_invalid = false;
        if self.options.managed
            && !kernel.tensors[&access.tensor]
                .accumulators
                .contains(&access.statement)
        {
            let rank = valid.len();
            let predicates: Vec<_> = valid
                .iter()
                .enumerate()
                .filter_map(|(axis, p)| {
                    p.as_ref().map(|p| {
                        if rank > 1 {
                            format!("({p})[{}]", Self::slice(rank, axis))
                        } else {
                            p.clone()
                        }
                    })
                })
                .collect();
            if !predicates.is_empty() {
                code = w.temporary(format!("tl.where({}, {code}, 0.0)", predicates.join(" & ")));
                zero_invalid = true;
            }
        }
        if self.options.managed {
            code = format!("({code}).to(tl.float32)");
        }
        EmittedValue {
            code,
            shape: self.tile_shape(id),
            valid,
            zero_invalid,
        }
    }
}
