//! Independently callable original regions for provider comparison. No operation
//! is extracted from a region and no cross-region split parameter is guessed.
use super::{CodegenContext, TritonPlan};
impl TritonPlan {
    pub fn emit_region(&self, index: usize) -> Result<String, String> {
        let kernel = self.kernels.get(index).ok_or("missing Triton region")?;
        if !self.metadata.split_owners.is_empty() {
            return Err(
                "region selection does not yet carry cross-kernel split tuning parameters".into(),
            );
        }
        let mut w = CodegenContext::default();
        w.line("import triton\nimport triton.language as tl\nimport torch\n");
        self.launch_prelude(&mut w);
        self.kernel(index, kernel, &mut w);
        w.line("def run(values):");
        w.indent = 1;
        let mut device = None;
        for tensor in self
            .tensor_order(kernel)
            .into_iter()
            .filter(|t| kernel.tensors[t].has_global())
        {
            let name = self.tensor_name(tensor);
            let id = tensor.index();
            let shape = &self.options.shapes[&self.analysis.tensor(tensor).name];
            let dtype = self.tensor_dtype(tensor).python();
            w.line(format!("{name} = values[{id}]"));
            w.line(format!("if list({name}.shape) != {shape:?} or {name}.dtype != torch.{dtype} or not {name}.is_contiguous() or not {name}.is_cuda:"));
            w.indent += 1;
            w.line("raise ValueError('Triton region argument differs from the fixed-shape plan')");
            w.indent -= 1;
            if let Some(first) = &device {
                w.line(format!("if {name}.device != {first}.device:"));
                w.indent += 1;
                w.line("raise ValueError('Triton region arguments must share a device')");
                w.indent -= 1;
            } else {
                device = Some(name.to_string());
            }
        }
        for (symbol, value) in &self.options.symbols {
            w.line(format!("{} = {value}", self.parameter(symbol)));
        }
        for (symbol, (tensor, axis)) in &self.common.metadata.dimensions {
            let value = self.options.shapes[&self.analysis.tensor(*tensor).name][*axis];
            w.line(format!("{} = {value}", self.parameter(symbol)));
        }
        if let Some(device) = device {
            w.line(format!("with torch.cuda.device({device}.device):"));
            w.indent += 1;
            self.kernel_launch(index, kernel, &mut w);
            w.indent -= 1;
        } else {
            return Err("region selection requires a device memory argument".into());
        }
        w.line("return None");
        w.indent = 0;
        Ok(w.source)
    }
}
