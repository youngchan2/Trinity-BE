//! Assemble one Python module using Trinity's source conventions and benchmark ABI.
mod context;
mod indexing;
mod kernel;
mod launch;
mod local;
mod ops;
mod scalar;
mod wrapper;

use super::ProgramPlan;
use context::CodegenContext;

impl ProgramPlan {
    /// Emit every kernel and the named forward wrapper into one Python source string.
    pub fn emit(&self) -> String {
        let mut w = CodegenContext::default();
        w.line("import triton\nimport triton.language as tl\nimport torch\n");
        if self.options.managed {
            self.managed_prelude(&mut w);
        }
        for (ki, kernel) in self.kernels.iter().enumerate() {
            self.kernel(ki, kernel, &mut w);
        }
        self.wrapper(&mut w);
        w.source
    }
}
