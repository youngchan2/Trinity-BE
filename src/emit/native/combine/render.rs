//! Rendering of decided phase placement and register connections.
use super::{CombinedBody, CombinedPlan};
use crate::emit::native::render_index_expression;
use crate::emit::native::{Kernel, KernelBindings, RegisterBinding, ThreadPolicy};
use crate::emit::provider::ProviderError;
use crate::{LoopDomain, OperationId};
use std::collections::{BTreeMap, BTreeSet};

pub(in crate::emit) struct RenderedBody {
    pub includes: Vec<&'static str>,
    pub code: String,
}

pub(super) fn render_body(
    plan: &CombinedPlan<'_>,
    body: &CombinedBody,
    bindings: &KernelBindings,
) -> Result<RenderedBody, ProviderError> {
    let mut includes = BTreeSet::new();
    let mut code = String::new();
    for invocation in &body.roots {
        // A uniform CTA barrier makes global writes visible and releases scratch.
        if invocation.barrier_before {
            code.push_str("__syncthreads();\n");
        }
        let steps: Vec<_> = std::iter::once(invocation.operation)
            .chain(invocation.followers.iter().copied())
            .collect();
        let pipeline = render_pipeline(
            plan,
            body,
            &steps,
            invocation.iteration.as_ref(),
            bindings,
            &BTreeMap::new(),
            &mut includes,
        )?;
        let requirements = plan.kernels[&invocation.operation]
            .specification
            .requirements();
        if let ThreadPolicy::Subgroup { threads } = requirements.thread_policy
            && threads < body.requirements.block_threads
        {
            // The complete invocation and its Register continuations share this
            // predicate. CTA barriers between roots remain outside the predicate.
            code.push_str(&format!(
                "if (threadIdx.x < {threads}) {{\n{pipeline}\n}}\n"
            ));
        } else {
            code.push_str(&pipeline);
        }
    }
    Ok(RenderedBody {
        includes: includes.into_iter().collect(),
        code,
    })
}

fn failed(message: impl Into<String>) -> ProviderError {
    ProviderError::Failed(message.into())
}

fn render_pipeline(
    plan: &CombinedPlan<'_>,
    body: &CombinedBody,
    steps: &[OperationId],
    iteration: Option<&LoopDomain>,
    bindings: &KernelBindings,
    available: &BTreeMap<(OperationId, usize), RegisterBinding>,
    includes: &mut BTreeSet<&'static str>,
) -> Result<String, ProviderError> {
    let Some((&operation, remaining)) = steps.split_first() else {
        return Ok(String::new());
    };
    let kernel = &plan.kernels[&operation];
    let specification = &kernel.specification;
    let mut local = bindings.clone();
    local.block_threads = body.requirements.block_threads;
    local.prefix = format!("{}_op{}", bindings.prefix, operation.index());
    local.registers.clear();
    for connection in body.connections.iter().filter(|c| c.consumer == operation) {
        let value = available
            .get(&(connection.producer, connection.output))
            .ok_or_else(|| failed("Register connection escaped its lexical scope"))?;
        local.registers.insert(connection.input, value.clone());
    }
    if let Some(domain) = iteration {
        local.indices.insert(
            domain.variable.clone(),
            format!("{}_{}", local.prefix, domain.variable),
        );
    }
    let Kernel {
        includes: headers,
        prologue,
        mainloop,
        epilogue,
    } = kernel.provider.render(specification, &local)?;
    includes.extend(headers);
    let reject_output =
        &mut |_, _: &RegisterBinding| Err(failed("output slot must occur in epilogue"));
    let prologue = prologue.connect(reject_output)?;
    let mainloop = mainloop
        .map(|code| code.connect(reject_output))
        .transpose()?;
    let interface = specification.interface(body.requirements.block_threads);
    let mut exports = BTreeSet::new();
    let epilogue = epilogue.connect(&mut |port, value| {
        let output = interface
            .outputs
            .get(port)
            .ok_or_else(|| failed("undeclared output slot"))?;
        if value.coordinates.len() != output.access.axes.len() || !exports.insert(port) {
            return Err(failed("invalid or duplicate output slot"));
        }
        let mut available = available.clone();
        available.insert((operation, port), value.clone());
        render_pipeline(plan, body, remaining, None, bindings, &available, includes)
    })?;
    if exports.len() != interface.outputs.len() {
        return Err(failed("missing declared output slot"));
    }
    let middle = match (iteration, mainloop) {
        (Some(domain), Some(code)) => {
            let variable = &local.indices[&domain.variable];
            format!(
                "for (int64_t {variable} = {}; {variable} < {}; {variable} += {}) {{\n{code}\n}}\n",
                render_index_expression(&domain.start, bindings)?,
                render_index_expression(&domain.stop, bindings)?,
                render_index_expression(&domain.step, bindings)?
            )
        }
        (None, None) => String::new(),
        _ => {
            return Err(failed(
                "rendered phases differ from the declared iteration contract",
            ));
        }
    };
    Ok(format!("{{\n{prologue}\n{middle}\n{epilogue}\n}}\n"))
}
