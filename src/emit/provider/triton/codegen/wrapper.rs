//! Program ABI, allocation and ordered kernel launches.
use super::super::TritonPlan;
use super::context::{CodegenContext, tuple};
use crate::analysis::*;

impl TritonPlan {
    pub(super) fn wrapper(&self, w: &mut CodegenContext) {
        let inputs: Vec<_> = self
            .analysis
            .declared_tensors(TensorKind::Input)
            .into_iter()
            .collect();
        let outputs = &self.metadata.output_order;
        let external: Vec<_> = inputs
            .iter()
            .chain(outputs.iter().filter(|t| !inputs.contains(t)))
            .copied()
            .collect();
        w.line(format!(
            "TENSOR_PARAMS = [{}]",
            external
                .iter()
                .map(|t| format!("'{}'", self.tensor_name(*t)))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        w.line("BLOCK_PARAMS = []\n");
        let mut params: Vec<_> = inputs
            .iter()
            .map(|t| self.tensor_name(*t).to_owned())
            .collect();
        params.extend(
            outputs
                .iter()
                .filter(|t| !inputs.contains(t))
                .map(|t| format!("{}=None", self.tensor_name(*t))),
        );
        w.line(format!("def forward({}):", params.join(", ")));
        w.indent = 1;
        w.line("\"\"\"Run the selected IR regions; internal tensors are allocated here.\"\"\"");
        let device = inputs.first().map(|t| format!("{}.device",self.tensor_name(*t)))
            .unwrap_or_else(|| outputs.first().map(|t| {
                let n=self.tensor_name(*t);
                format!("({n}.device if {n} is not None else torch.device('cuda', torch.cuda.current_device()))")
            }).unwrap_or("torch.device('cuda', torch.cuda.current_device())".into()));
        for (symbol, (tensor, axis)) in &self.common.metadata.dimensions {
            let parameter = self.parameter(symbol);
            w.line(format!(
                "{parameter} = {}.shape[{axis}]",
                self.tensor_name(*tensor)
            ));
            w.line(format!("if {parameter} <= 0:"));
            w.indent += 1;
            w.line(format!("raise ValueError('{parameter} must be positive')"));
            w.indent -= 1;
        }
        for tid in &external {
            let name = self.tensor_name(*tid);
            let dtype = self.tensor_dtype(*tid).python();
            let label = self.tensor_dtype(*tid).label();
            let shape = tuple(
                self.common.metadata.shapes[tid]
                    .iter()
                    .map(|e| self.index(e)),
            );
            if outputs.contains(tid) {
                w.line(format!("if {name} is None:"));
                w.indent += 1;
                w.line(format!(
                    "{name} = torch.empty({shape}, device={device}, dtype=torch.{dtype})"
                ));
                w.indent -= 1;
            }
            w.line(format!("if tuple({name}.shape) != {shape} or {name}.dtype != torch.{dtype} or {name}.device != {device}:"));
            w.indent += 1;
            w.line(format!("raise ValueError('{name}: expected {label} tensor with shape {shape} on the input device')"));
            w.indent -= 1;
        }
        let dimensions = self
            .common
            .metadata
            .dimensions
            .keys()
            .map(|s| {
                let p = self.parameter(s);
                format!("'{p}': {p}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        for (symbol, scope) in &self.metadata.split_owners {
            let ki = self.analysis.scope(*scope).kernel.index();
            let p = self.parameter(symbol);
            w.line(format!(
                "legal_{ki} = _prune_kernel_{ki}(KERNEL_{ki}_CONFIGS, {{{dimensions}}})"
            ));
            w.line(format!("if not legal_{ki}:"));
            w.indent += 1;
            w.line(format!(
                "raise ValueError('kernel {ki}: no legal profile configuration')"
            ));
            w.indent -= 1;
            w.line(format!(
                "capacity_{p} = max(config.kwargs['{p}'] for config in legal_{ki})"
            ));
        }
        for tid in self.globals.iter().filter(|t| !external.contains(t)) {
            let shape = tuple(
                self.common.metadata.shapes[tid]
                    .iter()
                    .map(|e| self.allocation_expr(e)),
            );
            w.line(format!(
                "{} = torch.empty({shape}, device={device}, dtype=torch.{})",
                self.tensor_name(*tid),
                self.tensor_dtype(*tid).python()
            ));
        }
        for (ki, kernel) in self.kernels.iter().enumerate() {
            self.kernel_launch(ki, kernel, w);
        }
        w.line(format!(
            "return {}",
            match outputs.len() {
                0 => "None".into(),
                1 => self.tensor_name(outputs[0]).to_owned(),
                _ => tuple(outputs.iter().map(|t| self.tensor_name(*t))),
            }
        ));
        w.indent = 0;
    }
    pub(super) fn kernel_launch(
        &self,
        ki: usize,
        kernel: &super::super::KernelPlan,
        w: &mut CodegenContext,
    ) {
        let mut arguments = Vec::new();
        for tensor in self
            .tensor_order(kernel)
            .into_iter()
            .filter(|t| kernel.tensors[t].has_global())
        {
            let name = self.tensor_name(tensor);
            let representative = self.kernel_access(kernel, tensor);
            let view = self
                .analysis
                .access(representative)
                .view_shape
                .as_ref()
                .unwrap_or(&self.common.metadata.shapes[&tensor]);
            let shape: Vec<_> = view.iter().map(|e| self.allocation_expr(e)).collect();
            let base: Vec<_> = self.common.metadata.shapes[&tensor]
                .iter()
                .map(|e| self.allocation_expr(e))
                .collect();
            let arg = if shape != base {
                let arg = format!("{name}_view_{ki}");
                w.line(format!("{arg} = {name}.view{}", tuple(&shape)));
                arg
            } else {
                name.to_owned()
            };
            arguments.push(arg.clone());
            arguments.extend((0..view.len()).map(|axis| format!("{arg}.stride({axis})")));
        }
        let grid: Vec<_> = kernel
            .grid_extents
            .iter()
            .map(|expr| self.grid_expr(expr, kernel))
            .collect();
        let grid = if grid.is_empty() {
            "(1,)".into()
        } else {
            tuple(grid)
        };
        w.line(format!("kernel_{ki}[lambda meta: {grid}]("));
        w.indent += 1;
        for arg in arguments {
            w.line(format!("{arg},"));
        }
        let owned = self.owned_parameters(kernel);
        for symbol in self.parameters(kernel).difference(&owned) {
            let p = self.parameter(symbol);
            w.line(format!("{p}={p},"));
        }
        w.indent -= 1;
        w.line(")");
        for symbol in owned
            .iter()
            .filter(|s| self.metadata.split_owners.contains_key(*s))
        {
            let p = self.parameter(symbol);
            w.line(format!("{p} = kernel_{ki}.best_config.kwargs['{p}']"));
        }
    }
}
