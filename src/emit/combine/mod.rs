//! Composition within provider boundaries using access, phase, and thread contracts.

use super::EmitError;
use super::collect::{SelectedKernel, SelectedKernels};
use super::execution::ExecutionModel;
use super::prepare::PreparedPlan;
use super::provider::{KernelInterface, RegisterLayout, ThreadPolicy};
use crate::{LoopDomain, LoopKind, OperationId, PhysicalPlan, Statement, Storage, ValueInstanceId};
use std::collections::BTreeMap;

mod render;

#[cfg(test)]
pub(in crate::emit) mod tests;

pub(super) struct CombinedPlan<'provider> {
    pub execution: ExecutionModel,
    pub kernels: BTreeMap<OperationId, SelectedKernel<'provider>>,
    pub statements: Vec<CombinedStatement>,
}

#[derive(Debug)]
pub(super) enum CombinedStatement {
    Loop {
        kind: LoopKind,
        domain: LoopDomain,
        body: Vec<Self>,
    },
    Body(CombinedBody),
}

#[derive(Debug)]
pub(super) struct CombinedBody {
    /// Each root owns its output continuations and their lexical register lifetime.
    pub roots: Vec<Invocation>,
    pub requirements: BodyRequirements,
    pub connections: Vec<Connection>,
}

/// Execution requirements resolved for a complete body, not an individual kernel.
#[derive(Debug)]
pub(super) struct BodyRequirements {
    pub block_threads: usize,
    pub shared_memory_bytes: usize,
    pub shared_memory_alignment: usize,
    pub alignments: Vec<(ValueInstanceId, usize)>,
}

#[derive(Debug)]
pub(super) struct Invocation {
    pub operation: OperationId,
    pub iteration: Option<LoopDomain>,
    pub followers: Vec<OperationId>,
    /// Uniform CTA synchronization before memory consumption or scratch reuse.
    pub barrier_before: bool,
}

#[derive(Debug)]
pub(super) struct Connection {
    pub producer: OperationId,
    pub output: usize,
    pub consumer: OperationId,
    pub input: usize,
}

pub(super) fn combine<'p>(
    prepared: &PreparedPlan<'_>,
    execution: ExecutionModel,
    selected: SelectedKernels<'p>,
) -> Result<CombinedPlan<'p>, EmitError> {
    let SelectedKernels { kernels } = selected;

    validate_requirements(prepared.plan, &kernels)
        .map_err(|reason| EmitError::Combination { reason })?;

    let statements = compose_sequence(prepared.plan, prepared.plan.statements(), &kernels)
        .map_err(|reason| EmitError::Combination { reason })?;

    Ok(CombinedPlan {
        execution,
        kernels,
        statements,
    })
}

fn validate_requirements(
    plan: &PhysicalPlan,
    kernels: &BTreeMap<OperationId, SelectedKernel<'_>>,
) -> Result<(), String> {
    for (&id, kernel) in kernels {
        let requirements = kernel.specification.requirements();
        if !requirements.shared_memory_alignment.is_power_of_two() {
            return Err("invalid native execution requirements".into());
        }
        match requirements.thread_policy {
            ThreadPolicy::FullCta { threads } | ThreadPolicy::Subgroup { threads }
                if !(1..=1024).contains(&threads) =>
            {
                return Err("invalid native execution requirements".into());
            }
            ThreadPolicy::Subgroup { threads } if !threads.is_multiple_of(32) => {
                return Err("Subgroup execution requires complete warps".into());
            }
            ThreadPolicy::Flexible { supported }
                if supported.is_empty()
                    || supported
                        .iter()
                        .any(|threads| !(1..=1024).contains(threads)) =>
            {
                return Err("invalid flexible CTA thread sizes".into());
            }
            _ => {}
        }

        let crate::TargetCapability::Cuda(target) = plan.target();
        if requirements.shared_memory_bytes > target.max_shared_memory_per_cta() {
            return Err(format!(
                "operation {}: shared memory exceeds target capacity",
                id.index()
            ));
        }
    }

    Ok(())
}

fn validate_interface(
    plan: &PhysicalPlan,
    id: OperationId,
    interface: &KernelInterface,
) -> Result<(), String> {
    for port in interface.inputs.iter().chain(&interface.outputs) {
        let value = plan
            .value_instance(port.access.value)
            .ok_or("missing port value")?;

        if value.dtype() != port.access.dtype
            || value.shape() != port.access.value_shape
            || value.storage() != port.access.storage
        {
            return Err("interface value metadata differs from the Plan".into());
        }
        if !plan
            .operation(id)
            .ok_or("missing interface operation")?
            .expression()
            .accesses()
            .iter()
            .any(|access| port.access.matches(access))
        {
            return Err("interface access/view differs from the Plan".into());
        }
        if value.storage() == Storage::Shared {
            return Err("Shared tensor transport is not supported yet".into());
        }
        if value.storage() == Storage::Register && port.register.is_none() {
            return Err(format!(
                "operation {}: port requires memory for a Register value",
                id.index()
            ));
        }
    }
    Ok(())
}

fn compose_sequence(
    plan: &PhysicalPlan,
    statements: &[Statement],
    kernels: &BTreeMap<OperationId, SelectedKernel<'_>>,
) -> Result<Vec<CombinedStatement>, String> {
    let mut result = Vec::new();
    let mut pending = Vec::new();

    for statement in statements {
        match statement {
            Statement::Region(body) => {
                flush(plan, &mut pending, &mut result, kernels)?;
                result.extend(compose_sequence(plan, body, kernels)?);
            }
            Statement::Operation(id) => {
                if kernels[id].specification.iteration().is_some() {
                    return Err(
                        "accumulating implementation needs its enclosing sequential loop".into(),
                    );
                }

                pending.push(Invocation {
                    operation: *id,
                    iteration: None,
                    followers: vec![],
                    barrier_before: false,
                });
            }
            Statement::Loop(loop_) => {
                let accumulation = match loop_.body.as_slice() {
                    [Statement::Operation(id)]
                        if loop_.kind == LoopKind::Sequential
                            && kernels[id].specification.iteration()
                                == Some(loop_.domain.variable.as_str()) =>
                    {
                        Some(*id)
                    }
                    _ => None,
                };

                if let Some(operation) = accumulation {
                    pending.push(Invocation {
                        operation,
                        iteration: Some(loop_.domain.clone()),
                        followers: vec![],
                        barrier_before: false,
                    });
                } else {
                    flush(plan, &mut pending, &mut result, kernels)?;
                    result.push(CombinedStatement::Loop {
                        kind: loop_.kind,
                        domain: loop_.domain.clone(),
                        body: compose_sequence(plan, &loop_.body, kernels)?,
                    });
                }
            }
        }
    }

    flush(plan, &mut pending, &mut result, kernels)?;
    Ok(result)
}

fn flush(
    plan: &PhysicalPlan,
    pending: &mut Vec<Invocation>,
    result: &mut Vec<CombinedStatement>,
    kernels: &BTreeMap<OperationId, SelectedKernel<'_>>,
) -> Result<(), String> {
    let mut group: Vec<Invocation> = Vec::new();
    let mut supported: Vec<usize> = (1..=1024).collect();
    for invocation in pending.drain(..) {
        let policy = kernels[&invocation.operation]
            .specification
            .requirements()
            .thread_policy;
        let common: Vec<_> = supported
            .iter()
            .copied()
            .filter(|&threads| policy.accepts_block_threads(threads))
            .collect();
        let provider_changed = group.first().is_some_and(|first| {
            kernels[&first.operation].provider_index
                != kernels[&invocation.operation].provider_index
        });
        // Fusion retains all feasible sizes. It never commits to a launch size.
        if provider_changed || common.is_empty() {
            finish_group(plan, &mut group, &supported, result, kernels)?;
            supported = (1..=1024)
                .filter(|&threads| policy.accepts_block_threads(threads))
                .collect();
        } else {
            supported = common;
        }
        group.push(invocation);
    }
    finish_group(plan, &mut group, &supported, result, kernels)
}

fn choose_block_threads(supported: &[usize]) -> usize {
    // Body-wide policy, independent of operation and support-list order.
    // Keep 128 when supported; otherwise use the smallest feasible size.
    if supported.contains(&128) {
        128
    } else {
        supported[0]
    }
}

fn finish_group(
    plan: &PhysicalPlan,
    group: &mut Vec<Invocation>,
    supported: &[usize],
    result: &mut Vec<CombinedStatement>,
    kernels: &BTreeMap<OperationId, SelectedKernel<'_>>,
) -> Result<(), String> {
    if group.is_empty() {
        return Ok(());
    }
    let block_threads = choose_block_threads(supported);
    result.push(CombinedStatement::Body(compose_body(
        plan,
        std::mem::take(group),
        kernels,
        block_threads,
    )?));
    Ok(())
}

struct RegisterProducer {
    operation: OperationId,
    output: usize,
    root: usize,
    port: super::provider::KernelPort,
    layout: RegisterLayout,
}

fn compose_body(
    plan: &PhysicalPlan,
    invocations: Vec<Invocation>,
    kernels: &BTreeMap<OperationId, SelectedKernel<'_>>,
    block_threads: usize,
) -> Result<CombinedBody, String> {
    let mut roots: Vec<Invocation> = Vec::new();
    let mut connections = Vec::new();
    let mut available: BTreeMap<ValueInstanceId, RegisterProducer> = BTreeMap::new();
    let mut requirements = BodyRequirements {
        block_threads,
        shared_memory_bytes: 0,
        shared_memory_alignment: 1,
        alignments: vec![],
    };
    for mut invocation in invocations {
        let id = invocation.operation;
        let specification = &kernels[&id].specification;
        let interface = specification.interface(block_threads);
        validate_interface(plan, id, &interface)?;
        let resources = specification.requirements();
        let mut root = None;
        let mut inherited = None;
        for (input, port) in interface
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.access.storage == Storage::Register)
        {
            let producer = available.get(&port.access.value).ok_or_else(|| {
                format!(
                    "operation {}: Register input has no producer in this execution scope",
                    id.index()
                )
            })?;
            if producer.port.access != port.access || producer.port.projection != port.projection {
                return Err("Register access or projection differs from producer".into());
            }
            if !port
                .register
                .as_ref()
                .is_some_and(|layout| layout.accepts(&producer.layout))
            {
                return Err("Register thread layouts are incompatible".into());
            }
            if root.is_some_and(|r| r != producer.root) || producer.root + 1 != roots.len() {
                return Err("Register lifetime would escape its producer's output scope".into());
            }
            root = Some(producer.root);
            inherited = Some(producer.layout.clone());
            connections.push(Connection {
                producer: producer.operation,
                output: producer.output,
                consumer: id,
                input,
            });
        }
        let root = if let Some(root) = root {
            if invocation.iteration.is_some()
                || resources.thread_policy != ThreadPolicy::FollowInput
                || resources.shared_memory_bytes != 0
            {
                return Err(
                    "collective or repeated code cannot enter a per-element output scope".into(),
                );
            }
            // Moving consumers into a streaming fragment loop must not turn a
            // whole-tile memory dependency into a dependency between different elements.
            let previous =
                std::iter::once(roots[root].operation).chain(roots[root].followers.iter().copied());
            for prior in previous {
                check_memory_order(
                    &kernels[&prior].specification.interface(block_threads),
                    &interface,
                )?;
            }
            roots[root].followers.push(id);
            root
        } else {
            if resources.thread_policy == ThreadPolicy::FollowInput {
                return Err("FollowInput requires a Register producer in this output scope".into());
            }
            for previous in &roots {
                for prior in
                    std::iter::once(previous.operation).chain(previous.followers.iter().copied())
                {
                    check_memory_coverage(
                        &kernels[&prior].specification.interface(block_threads),
                        &interface,
                    )?;
                }
            }
            invocation.barrier_before = !roots.is_empty();
            roots.push(invocation);
            roots.len() - 1
        };
        for (output, port) in interface
            .outputs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.access.storage == Storage::Register)
        {
            let layout = match port
                .register
                .as_ref()
                .ok_or("Register output has no layout")?
            {
                RegisterLayout::FollowInput { .. } => inherited
                    .clone()
                    .ok_or("output layout needs a Register input")?,
                fixed => fixed.clone(),
            };
            available.insert(
                port.access.value,
                RegisterProducer {
                    operation: id,
                    output,
                    root,
                    port: port.clone(),
                    layout,
                },
            );
        }
        requirements.shared_memory_bytes = requirements
            .shared_memory_bytes
            .max(resources.shared_memory_bytes);
        requirements.shared_memory_alignment = requirements
            .shared_memory_alignment
            .max(resources.shared_memory_alignment);
        for &(value, alignment) in &resources.alignments {
            if interface
                .inputs
                .iter()
                .chain(&interface.outputs)
                .any(|p| p.access.value == value && p.access.storage == Storage::Register)
            {
                continue;
            }
            if let Some((_, previous)) = requirements
                .alignments
                .iter_mut()
                .find(|(v, _)| *v == value)
            {
                *previous = (*previous).max(alignment);
            } else {
                requirements.alignments.push((value, alignment));
            }
        }
    }
    Ok(CombinedBody {
        roots,
        requirements,
        connections,
    })
}

fn check_memory_order(prior: &KernelInterface, next: &KernelInterface) -> Result<(), String> {
    let hazards = prior
        .outputs
        .iter()
        .flat_map(|a| next.inputs.iter().chain(&next.outputs).map(move |b| (a, b)))
        .chain(
            prior
                .inputs
                .iter()
                .flat_map(|a| next.outputs.iter().map(move |b| (a, b))),
        );
    for (a, b) in hazards {
        if a.access.storage != Storage::Register
            && a.access.value == b.access.value
            && (a.access != b.access || a.projection != b.projection)
        {
            return Err("fragment interleaving would change memory access order".into());
        }
    }
    Ok(())
}

impl CombinedPlan<'_> {
    pub(super) fn render_body(
        &self,
        body: &CombinedBody,
        bindings: &super::provider::KernelBindings,
    ) -> Result<render::RenderedBody, super::provider::ProviderError> {
        render::render_body(self, body, bindings)
    }
}

/// A producer in this CTA must cover memory consumed by subsequent roots.
/// Bounds validity remains the responsibility of the input generator.
fn check_memory_coverage(prior: &KernelInterface, next: &KernelInterface) -> Result<(), String> {
    use super::provider::access::Axis;
    for output in &prior.outputs {
        for input in &next.inputs {
            if output.access.storage == Storage::Register
                || output.access.value != input.access.value
            {
                continue;
            }
            let covers = output
                .access
                .axes
                .iter()
                .zip(&input.access.axes)
                .all(|(a, b)| match (a, b) {
                    (Axis::Full, _) => true,
                    (
                        Axis::Tile {
                            variable: a,
                            width: aw,
                            clipped: ac,
                        },
                        Axis::Tile {
                            variable: b,
                            width: bw,
                            clipped: bc,
                        },
                    ) => a == b && aw >= bw && ac == bc,
                    _ => a == b,
                });
            if !covers {
                return Err(
                    "memory dependency requires a different CTA or an unproven execution region"
                        .into(),
                );
            }
        }
    }
    Ok(())
}
