//! Kernel launches: scratch capacity, legal profiles and cross-kernel parameters.
use super::super::{KernelPlan, TritonPlan};
use super::context::CodegenContext;
use crate::analysis::*;

impl TritonPlan {
    fn profile_expr(&self, expr: &IndexExpr) -> String {
        match expr {
            IndexExpr::Symbol(s)
                if self
                    .common
                    .metadata
                    .dimensions
                    .contains_key(self.canonical_symbol(s))
                    || self
                        .metadata
                        .candidates
                        .contains_key(self.canonical_symbol(s)) =>
            {
                format!("args['{}']", self.parameter(s))
            }
            IndexExpr::Apply(op, a) => format!(
                "({} {} {})",
                self.profile_expr(&a[0]),
                if op == "/" { "//" } else { op },
                self.profile_expr(&a[1])
            ),
            _ => self.index(expr),
        }
    }

    fn split_contract(&self, split: ScopeId) -> Option<(IndexExpr, &IndexExpr)> {
        let ScopeItem::Scope(serial) = *self.analysis.scope(split).children.first()? else {
            return None;
        };
        let info = self.analysis.scope(serial).loop_info.as_ref()?;
        let IndexExpr::Apply(add, a) = &info.start else {
            return None;
        };
        let IndexExpr::Apply(mul, b) = a.get(1)? else {
            return None;
        };
        if add != "+" || mul != "*" || b.first()? != &IndexExpr::LoopVar(split) {
            return None;
        }
        let chunk = b.get(1)?;
        let extent = match chunk {
            IndexExpr::Apply(op, args) if op == "/" || op == "//" => args.first()?.clone(),
            // A concrete PhysicalPlan may have folded extent / splits already.
            // Preserve its fixed chunk size instead of assuming an AST shape.
            _ => IndexExpr::Apply(
                "*".into(),
                vec![
                    chunk.clone(),
                    self.analysis.scope(split).loop_info.as_ref()?.end.clone(),
                ],
            ),
        };
        Some((extent, &info.step))
    }

    pub(super) fn launch_prelude(&self, w: &mut CodegenContext) {
        self.launch_prelude_for(&(0..self.kernels.len()).collect::<Vec<_>>(), w);
    }

    pub(super) fn launch_prelude_for(&self, indices: &[usize], w: &mut CodegenContext) {
        for &ki in indices {
            let kernel = &self.kernels[ki];
            w.line(format!("KERNEL_{ki}_CONFIGS = ["));
            w.indent = 1;
            for config in &self.tuning[ki] {
                let values = config
                    .parameters
                    .iter()
                    .map(|(s, v)| format!("'{}': {v}", self.parameter(s)))
                    .collect::<Vec<_>>()
                    .join(", ");
                w.line(format!(
                    "triton.Config({{{values}}}, num_warps={}, num_stages={}),",
                    config.num_warps, config.num_stages
                ));
            }
            w.indent = 0;
            w.line("]\n");
            w.line(format!(
                "def _prune_kernel_{ki}(configs, named_args, **kwargs):"
            ));
            w.indent = 1;
            w.line("legal = []");
            w.line("for config in configs:");
            w.indent = 2;
            w.line("args = dict(named_args)");
            w.line("args.update(kwargs)");
            w.line("args.update(config.kwargs)");
            let mut conditions = Vec::new();
            for (si, scope) in self
                .analysis
                .scopes()
                .iter()
                .enumerate()
                .filter(|(_, s)| s.kernel == self.analysis.scope(kernel.root_scope).kernel)
            {
                let Some(info) = &scope.loop_info else {
                    continue;
                };
                let id = ScopeId(si);
                if scope.kind == ScopeKind::SplitLoop
                    && let Some((extent, step)) = self.split_contract(id)
                {
                    let ns = self.profile_expr(&info.end);
                    conditions.push(format!(
                        "({ns} == 1 or {} % ({ns} * {}) == 0)",
                        self.profile_expr(&extent),
                        self.profile_expr(step)
                    ));
                } else if matches!(info.step, IndexExpr::Symbol(_))
                    && info.start.loop_dependencies().is_empty()
                    && info.end.loop_dependencies().is_empty()
                {
                    conditions.push(format!(
                        "{} <= triton.next_power_of_2({} - {})",
                        self.profile_expr(&info.step),
                        self.profile_expr(&info.end),
                        self.profile_expr(&info.start)
                    ));
                }
            }
            w.line(format!(
                "if {}:",
                if conditions.is_empty() {
                    "True".into()
                } else {
                    conditions.join(" and ")
                }
            ));
            w.indent = 3;
            w.line("legal.append(config)");
            w.indent = 1;
            w.line("if not legal:");
            w.indent = 2;
            w.line(format!("raise ValueError('kernel {ki}: no legal autotuning configuration for these shapes')"));
            w.indent = 1;
            w.line("return legal");
            w.indent = 0;
            w.line("");
        }
    }

    pub(super) fn autotune(&self, kernel: &KernelPlan, w: &mut CodegenContext) {
        let ki = self.analysis.scope(kernel.root_scope).kernel.index();
        let owned = self.owned_parameters(kernel);
        let mut keys: Vec<_> = self
            .parameters(kernel)
            .difference(&owned)
            .map(|s| format!("'{}'", self.parameter(s)))
            .collect();
        for tensor in self
            .tensor_order(kernel)
            .into_iter()
            .filter(|t| kernel.tensors[t].has_global())
        {
            keys.extend(
                (0..self.kernel_shape(kernel, tensor).len())
                    .map(|axis| format!("'{}_stride{axis}'", self.tensor_name(tensor))),
            );
        }
        // Any globally read AND written buffer can carry state between benchmark
        // invocations (inputs, cache updates, or an earlier kernel's scratch).
        let rw = &self
            .analysis
            .kernel(self.analysis.scope(kernel.root_scope).kernel)
            .read_writes;
        let restore = rw
            .reads
            .intersection(&rw.writes)
            .filter(|t| {
                let tensor = &kernel.tensors[t];
                tensor.has_global()
                    && (tensor
                        .initialization
                        .as_ref()
                        .is_some_and(|init| init.value == super::super::InitialValue::Global)
                        || self
                            .analysis
                            .kernel(self.analysis.scope(kernel.root_scope).kernel)
                            .accesses
                            .iter()
                            .any(|id| {
                                let access = self.analysis.access(*id);
                                access.tensor == **t
                                    && access.kind == AccessKind::Read
                                    && !kernel.register_accesses.contains(id)
                            }))
            })
            .map(|t| format!("'{}_ptr'", self.tensor_name(*t)))
            .collect::<Vec<_>>()
            .join(", ");
        w.line(format!("@triton.autotune(configs=KERNEL_{ki}_CONFIGS, key=[{}], restore_value=[{restore}], prune_configs_by={{'early_config_prune': _prune_kernel_{ki}}})", keys.join(", ")));
    }

    pub(super) fn launch_assertions(&self, kernel: &KernelPlan, w: &mut CodegenContext) {
        let ki = self.analysis.scope(kernel.root_scope).kernel;
        let mut constraints = std::collections::BTreeSet::new();
        for id in &self.analysis.kernel(ki).accesses {
            let a = self.analysis.access(*id);
            if !kernel.register_accesses.contains(id) {
                let view = self.view_shape(*id).join(" * ");
                let base = self
                    .view_shape(self.kernel_access(kernel, a.tensor))
                    .join(" * ");
                if view != base && constraints.insert((view.clone(), base.clone())) {
                    w.line(format!("tl.static_assert(({view}) == ({base}), 'view changes the storage element count')"));
                }
            }
        }
        for symbol in self
            .parameters(kernel)
            .iter()
            .filter(|s| self.metadata.candidates.contains_key(*s))
        {
            let name = self.parameter(symbol);
            w.line(format!("tl.static_assert({name} > 0)"));
            if !self.metadata.split_owners.contains_key(symbol) {
                w.line(format!("tl.static_assert(({name} & ({name} - 1)) == 0)"));
            }
        }
        for split in kernel
            .parallel_loops
            .iter()
            .filter(|s| self.analysis.scope(**s).kind == ScopeKind::SplitLoop)
        {
            let Some((extent, step)) = self.split_contract(*split) else {
                continue;
            };
            let ns = self.index(&self.analysis.scope(*split).loop_info.as_ref().unwrap().end);
            w.line(format!("tl.static_assert({ns} == 1 or {} % ({ns} * {}) == 0, 'mloop split chunks must contain whole serial tiles')", self.index(&extent), self.index(step)));
        }
    }

    pub(super) fn allocation_expr(&self, expr: &IndexExpr) -> String {
        match expr {
            IndexExpr::Symbol(s)
                if self
                    .common
                    .metadata
                    .splits
                    .contains_key(self.canonical_symbol(s)) =>
            {
                format!("capacity_{}", self.parameter(s))
            }
            IndexExpr::Apply(op, a) => format!(
                "({} {} {})",
                self.allocation_expr(&a[0]),
                if op == "/" { "//" } else { op },
                self.allocation_expr(&a[1])
            ),
            _ => self.index(expr),
        }
    }

    pub(super) fn grid_expr(&self, expr: &IndexExpr, kernel: &KernelPlan) -> String {
        match expr {
            IndexExpr::Symbol(s)
                if self
                    .owned_parameters(kernel)
                    .contains(self.canonical_symbol(s)) =>
            {
                format!("meta['{}']", self.parameter(s))
            }
            IndexExpr::Apply(op, a) => format!(
                "({} {} {})",
                self.grid_expr(&a[0], kernel),
                if op == "/" { "//" } else { op },
                self.grid_expr(&a[1], kernel)
            ),
            _ => self.index(expr),
        }
    }
}
