//! Managed launches: scratch capacity, legal profiles and cross-kernel parameters.
use super::super::{KernelPlan, TritonPlan};
use super::context::{CodegenContext, tuple};
use crate::analysis::*;
use std::collections::BTreeMap;

impl TritonPlan {
    fn configurations(&self, kernel: &KernelPlan) -> Vec<BTreeMap<String, i64>> {
        let mut configs = vec![BTreeMap::new()];
        for symbol in self.owned_parameters(kernel) {
            configs = configs
                .into_iter()
                .flat_map(|prefix| {
                    self.metadata.candidates[&symbol].iter().map({
                        let symbol = symbol.clone();
                        move |value| {
                            let mut row = prefix.clone();
                            row.insert(symbol.clone(), *value);
                            row
                        }
                    })
                })
                .collect();
        }
        configs
    }

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

    fn split_contract(&self, split: ScopeId) -> (&IndexExpr, &IndexExpr) {
        let ScopeItem::Scope(serial) = self.analysis.scope(split).children[0] else {
            unreachable!()
        };
        let info = self.analysis.scope(serial).loop_info.as_ref().unwrap();
        let IndexExpr::Apply(_, a) = &info.start else {
            unreachable!()
        };
        let IndexExpr::Apply(_, b) = &a[1] else {
            unreachable!()
        };
        let IndexExpr::Apply(_, chunk) = &b[1] else {
            unreachable!()
        };
        (&chunk[0], &info.step)
    }

    pub(super) fn managed_prelude(&self, w: &mut CodegenContext) {
        for (ki, kernel) in self.kernels.iter().enumerate() {
            w.line(format!("KERNEL_{ki}_CONFIGS = ["));
            w.indent = 1;
            for row in self.configurations(kernel) {
                let values = row
                    .iter()
                    .map(|(s, v)| format!("'{}': {v}", self.parameter(s)))
                    .collect::<Vec<_>>()
                    .join(", ");
                w.line(format!(
                    "triton.Config({{{values}}}, num_warps=4, num_stages=1),"
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
            for id in &kernel.parallel_loops {
                let scope = self.analysis.scope(*id);
                let info = scope.loop_info.as_ref().unwrap();
                if scope.kind == ScopeKind::SplitLoop {
                    let (extent, step) = self.split_contract(*id);
                    let ns = self.profile_expr(&info.end);
                    conditions.push(format!(
                        "({ns} == 1 or {} % ({ns} * {}) == 0)",
                        self.profile_expr(extent),
                        self.profile_expr(step)
                    ));
                } else if matches!(info.step, IndexExpr::Symbol(_)) {
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
            w.line("return legal");
            w.indent = 0;
            w.line("");
        }
    }

    pub(super) fn managed_autotune(&self, kernel: &KernelPlan, w: &mut CodegenContext) {
        if self.owned_parameters(kernel).is_empty() {
            return;
        }
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
        w.line(format!("@triton.autotune(configs=KERNEL_{ki}_CONFIGS, key=[{}], prune_configs_by={{'early_config_prune': _prune_kernel_{ki}}})", keys.join(", ")));
    }

    pub(super) fn managed_assertions(&self, kernel: &KernelPlan, w: &mut CodegenContext) {
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
            let (extent, step) = self.split_contract(*split);
            let ns = self.index(&self.analysis.scope(*split).loop_info.as_ref().unwrap().end);
            w.line(format!("tl.static_assert({ns} == 1 or {} % ({ns} * {}) == 0, 'mloop split chunks must contain whole serial tiles')", self.index(extent), self.index(step)));
        }
    }

    fn allocation_expr(&self, expr: &IndexExpr) -> String {
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

    fn grid_expr(&self, expr: &IndexExpr, kernel: &KernelPlan) -> String {
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

    pub(super) fn managed_wrapper(&self, w: &mut CodegenContext) {
        let inputs: Vec<_> = self
            .analysis
            .declared_tensors(TensorKind::Input)
            .into_iter()
            .collect();
        let outputs: Vec<_> = self
            .analysis
            .declared_tensors(TensorKind::Output)
            .into_iter()
            .collect();
        let external: Vec<_> = inputs.iter().chain(&outputs).copied().collect();
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
}
