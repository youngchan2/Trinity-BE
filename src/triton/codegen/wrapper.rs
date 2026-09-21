//! Autotune configurations, benchmark metadata and the named forward ABI.
use super::super::shape::{loop_range, positive};
use super::super::{KernelPlan, TritonPlan};
use super::context::{CodegenContext, tuple};
use std::collections::BTreeSet;

impl TritonPlan {
    pub(super) fn autotune(&self, kernel: &KernelPlan, w: &mut CodegenContext) {
        if self.options.managed {
            self.managed_autotune(kernel, w);
            return;
        }
        let loops: Vec<_> = self
            .loops(kernel)
            .into_iter()
            .filter(|id| self.tunable(*id))
            .collect();
        if loops.is_empty() {
            return;
        }
        let mut configs: Vec<Vec<usize>> = vec![vec![]];
        for id in &loops {
            let size = loop_range(&self.analysis, *id, &self.options)
                .map(|(start, end, _)| (end - start) as usize)
                .unwrap_or_else(|_| {
                    positive(
                        &self.analysis.scope(*id).loop_info.as_ref().unwrap().step,
                        &self.options,
                    )
                    .unwrap()
                });
            let choices = if size < 16 {
                vec![1, 2, 4, 8]
            } else {
                vec![16, 32, 64, 128]
            };
            configs = configs
                .into_iter()
                .flat_map(|prefix| {
                    choices.iter().filter(|v| **v <= size).map(move |v| {
                        let mut row = prefix.clone();
                        row.push(*v);
                        row
                    })
                })
                .collect();
            configs.truncate(16);
        }
        w.line("@triton.autotune(\n    configs = [");
        for (i, config) in configs.iter().enumerate() {
            let values = loops
                .iter()
                .zip(config)
                .map(|(id, v)| format!("'{}': {v}", self.block(*id)))
                .collect::<Vec<_>>()
                .join(", ");
            w.line(format!(
                "        triton.Config({{{values}}}){}",
                if i + 1 < configs.len() { "," } else { "" }
            ));
        }
        w.line("    ], key=[]\n)");
    }

    pub(super) fn wrapper(&self, w: &mut CodegenContext) {
        if self.options.managed {
            self.managed_wrapper(w);
            return;
        }
        let mut tensors: Vec<_> = self.globals.iter().copied().collect();
        tensors.sort_by_key(|t| &self.analysis.tensor(*t).name);
        let names: Vec<_> = tensors
            .iter()
            .map(|t| self.tensor_name(*t).to_owned())
            .collect();
        let blocks: BTreeSet<_> = self
            .kernels
            .iter()
            .flat_map(|k| self.loops(k))
            .filter(|id| self.tunable(*id))
            .map(|id| format!("block_{}", self.loop_name(id)))
            .collect();
        w.line("# Metadata for benchmark.py");
        w.line(format!(
            "TENSOR_PARAMS = [{}]",
            names
                .iter()
                .map(|s| format!("'{s}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        w.line(format!(
            "BLOCK_PARAMS = [{}]\n",
            blocks
                .iter()
                .map(|s| format!("'{s}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        let mut params = names;
        params.extend(blocks.iter().map(|s| format!("{s}=16")));
        w.line(format!("def forward({}):", params.join(", ")));
        w.indent = 1;
        w.line("\"\"\"Wrapper function that executes all kernels sequentially.\"\"\"");
        for (ki, kernel) in self.kernels.iter().enumerate() {
            let mut arguments = Vec::new();
            for tensor in self
                .tensor_order(kernel)
                .into_iter()
                .filter(|t| kernel.tensors[t].has_global())
            {
                let name = self.tensor_name(tensor);
                let shape = self.kernel_shape(kernel, tensor);
                let arg = if shape != self.options.shapes[&self.analysis.tensor(tensor).name] {
                    let arg = format!("{name}_view_{ki}");
                    w.line(format!("{arg} = {name}.view{}", tuple(&shape)));
                    arg
                } else {
                    name.to_owned()
                };
                arguments.push(arg.clone());
                for axis in 0..shape.len() {
                    arguments.push(format!("{arg}.stride({axis})"));
                }
            }
            let dynamic = kernel.parallel_loops.iter().any(|id| self.tunable(*id));
            let grid: Vec<_> = kernel
                .parallel_loops
                .iter()
                .map(|id| {
                    let (start, end, step) =
                        loop_range(&self.analysis, *id, &self.options).unwrap();
                    let block = if self.tunable(*id) {
                        format!("meta[\"{}\"]", self.block(*id))
                    } else {
                        step.to_string()
                    };
                    format!("({end} - {start} + {block} - 1) // {block}")
                })
                .collect();
            let grid = if grid.is_empty() {
                "(1,)".into()
            } else {
                tuple(grid)
            };
            w.line(format!(
                "kernel_{ki}[{}{grid}](",
                if dynamic { "lambda meta: " } else { "" }
            ));
            w.indent += 1;
            for arg in arguments {
                w.line(format!("{arg},"));
            }
            for id in self.loops(kernel) {
                let block = self.block(id);
                if self.tunable(id) {
                    w.line(format!("# {block} is automatically set by autotune"));
                } else {
                    w.line(format!(
                        "{block}={},",
                        positive(
                            &self.analysis.scope(id).loop_info.as_ref().unwrap().step,
                            &self.options
                        )
                        .unwrap()
                    ));
                }
            }
            w.indent -= 1;
            w.line(")\n");
        }
        w.line("# Return output tensors if needed");
        w.line("# This depends on your specific use case");
        w.line("pass");
        w.indent = 0;
    }
}
