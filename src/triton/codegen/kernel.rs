//! Kernel signatures, program mapping, ordered scopes and init/export placement.
use super::super::{InitialValue, KernelPlan, ProgramPlan};
use super::context::{CodegenContext, tuple};
use crate::analysis::*;
use std::collections::BTreeSet;

impl ProgramPlan {
    pub(super) fn kernel(&self, ki: usize, kernel: &KernelPlan, w: &mut CodegenContext) {
        self.autotune(kernel, w);
        w.line("@triton.jit");
        w.line(format!("def kernel_{ki}("));
        w.indent = 1;
        let mut params = Vec::new();
        for tensor in self
            .tensor_order(kernel)
            .into_iter()
            .filter(|t| kernel.tensors[t].has_global())
        {
            let name = self.tensor_name(tensor);
            params.push(format!("{name}_ptr"));
            for axis in 0..self.kernel_shape(kernel, tensor).len() {
                params.push(format!("{name}_stride{axis}: tl.constexpr"));
            }
        }
        if self.options.managed {
            params.extend(
                self.parameters(kernel)
                    .iter()
                    .map(|s| format!("{}: tl.constexpr", self.parameter(s))),
            );
        } else {
            for id in self.loops(kernel) {
                params.push(format!("{}: tl.constexpr", self.block(id)));
            }
        }
        for (i, param) in params.iter().enumerate() {
            w.line(format!(
                "{param}{}",
                if i + 1 < params.len() { "," } else { "" }
            ));
        }
        w.indent = 0;
        w.line("):");
        w.indent = 1;
        w.temp = 0;
        w.offset = 0;
        w.indices.clear();
        w.loads.clear();
        if self.options.managed {
            self.managed_assertions(kernel, w);
        }
        if self.analysis.scope(kernel.root_scope).children.is_empty() {
            w.line("pass");
        }
        self.scope(kernel.root_scope, kernel, w);
        w.indent = 0;
        w.line("\n");
    }

    pub(super) fn tensor_order(&self, kernel: &KernelPlan) -> Vec<TensorId> {
        let mut ids: Vec<_> = kernel.tensors.keys().copied().collect();
        ids.sort_by_key(|id| &self.analysis.tensor(*id).name);
        ids
    }

    pub(super) fn loop_name(&self, id: ScopeId) -> String {
        let s = self.analysis.scope(id);
        let info = s.loop_info.as_ref().unwrap();
        let name = super::super::lowering::metadata::binding_name(&info.variable);
        let conflict = self.analysis.scopes().iter().enumerate().any(|(i, other)| {
            i < id.index()
                && other.kernel == s.kernel
                && other.loop_info.as_ref().is_some_and(|l| {
                    super::super::lowering::metadata::binding_name(&l.variable) == name
                        && (l.variable != info.variable
                            || l.step != info.step
                            || self.analysis.is_within(id, ScopeId(i)))
                })
        });
        if conflict {
            format!("{}_{}", name, id.index())
        } else {
            name
        }
    }

    pub(super) fn block(&self, id: ScopeId) -> String {
        if self.options.managed {
            return self.index(&self.analysis.scope(id).loop_info.as_ref().unwrap().step);
        }
        format!("BLOCK_{}", self.loop_name(id).to_uppercase())
    }

    pub(super) fn loops(&self, kernel: &KernelPlan) -> Vec<ScopeId> {
        let mut seen = BTreeSet::new();
        self.analysis
            .scopes()
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let id = ScopeId(i);
                if s.kernel == self.analysis.scope(kernel.root_scope).kernel
                    && s.loop_info.is_some()
                    && seen.insert(self.block(id))
                {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub(super) fn tunable(&self, id: ScopeId) -> bool {
        matches!(
            self.analysis.scope(id).loop_info.as_ref().unwrap().step,
            IndexExpr::Symbol(_)
        )
    }

    pub(super) fn kernel_shape(&self, kernel: &KernelPlan, tensor: TensorId) -> Vec<usize> {
        self.accesses[self.kernel_access(kernel, tensor).index()]
            .axes
            .iter()
            .map(|a| a.extent)
            .collect()
    }

    pub(super) fn kernel_access(&self, kernel: &KernelPlan, tensor: TensorId) -> AccessId {
        let ki = self.analysis.scope(kernel.root_scope).kernel;
        self.analysis
            .accesses()
            .iter()
            .enumerate()
            .filter(|(_, a)| a.tensor == tensor && self.analysis.scope(a.scope).kernel == ki)
            .max_by_key(|(_, a)| a.index.len())
            .map(|(i, _)| AccessId(i))
            .unwrap()
    }

    fn initializations(
        &self,
        id: ScopeId,
        kernel: &KernelPlan,
        zero: bool,
        w: &mut CodegenContext,
    ) {
        for tensor in self.tensor_order(kernel) {
            let tp = &kernel.tensors[&tensor];
            if let Some(init) = &tp.initialization
                && init.scope == id
                && (init.value == InitialValue::Zero) == zero
            {
                let name = self.tensor_name(tensor);
                if zero {
                    w.line(format!(
                        "{name} = tl.zeros({}, dtype=tl.float32)",
                        tuple(self.tile_shape(init.access))
                    ));
                } else {
                    let value = self.load(init.access, kernel, w);
                    w.line(format!("{name} = {}", value.code));
                }
            }
        }
    }

    fn scope(&self, id: ScopeId, kernel: &KernelPlan, w: &mut CodegenContext) {
        let scope = self.analysis.scope(id);
        let previous_indices = w.indices.clone();
        if scope.kind == ScopeKind::SequentialLoop && scope.children.is_empty() {
            w.line("# Skipped empty sloop with dummy body");
            return;
        }
        if scope.kind.is_parallel() {
            self.initializations(id, kernel, true, w);
        }
        if scope.kind != ScopeKind::Kernel {
            let info = scope.loop_info.as_ref().unwrap();
            let start = self.index(&info.start);
            let end = self.index(&info.end);
            let variable = self.loop_name(id);
            let block = self.block(id);
            if scope.kind.is_parallel() {
                let axis = kernel.parallel_loops.iter().position(|s| *s == id).unwrap();
                w.line(format!(
                    "# Parallel loop {variable} from {start} to {end} with tile size {block}"
                ));
                w.line(format!("# Executed across grid dimension {axis}"));
                w.line(format!(
                    "{variable} = {start} + tl.program_id({axis}) * {block}"
                ));
            } else {
                w.line(format!(
                    "# Sequential loop {variable} from {start} to {end} with tile size {block}"
                ));
                w.line(format!("for {variable} in range({start}, {end}, {block}):"));
                w.indent += 1;
                w.indices.clear();
                w.loads.clear();
                if scope.children.is_empty() {
                    w.line("pass");
                }
            }
        }
        if !scope.kind.is_parallel() {
            self.initializations(id, kernel, true, w);
        }
        self.initializations(id, kernel, false, w);
        for item in &scope.children {
            match *item {
                ScopeItem::Scope(child) => self.scope(child, kernel, w),
                ScopeItem::Statement(statement) => {
                    let info = self.analysis.statement(statement);
                    let access = *info.accesses.last().unwrap();
                    let tensor = self.analysis.access(access).tensor;
                    let value = self.expression(&self.expressions[statement.index()], kernel, w);
                    // Stores have their own address mask. Register lanes only
                    // need neutralization when a later operation reduces them.
                    let mut code = value.code;
                    if value.shape.is_empty() {
                        code = format!("tl.full((), {code}, tl.float32)");
                    }
                    if kernel.register_accesses.contains(&access) {
                        let target_shape = self.tile_shape(access);
                        if self.options.managed
                            || self.expressions[statement.index()].shape
                                != self.accesses[access.index()].shape
                        {
                            code = format!("tl.broadcast_to({code}, {})", tuple(target_shape));
                        }
                        if self.options.managed
                            && !kernel.tensors[&tensor].accumulators.contains(&statement)
                        {
                            code = format!("({code}).to(tl.float16)");
                        }
                        w.line(format!("{} = {code}", self.tensor_name(tensor)));
                    } else {
                        self.store(access, &code, kernel, w);
                    }
                    w.loads
                        .retain(|key, _| !key.starts_with(&format!("{}:", tensor.index())));
                }
            }
        }
        for tensor in self.tensor_order(kernel) {
            let tp = &kernel.tensors[&tensor];
            if tp.export_scope == Some(id) {
                self.store(tp.representative, self.tensor_name(tensor), kernel, w);
            }
        }
        if scope.kind == ScopeKind::SequentialLoop {
            w.indent -= 1;
            w.indices = previous_indices;
            w.loads.clear();
        }
    }
}
