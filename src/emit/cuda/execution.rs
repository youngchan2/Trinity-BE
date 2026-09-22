use super::{invalid, unsupported};
use crate::compile::CudaRequirements;
use crate::emit::native::bindings::BufferBindings;
use crate::emit::{
    EmitError,
    combine::{CombinedBody, CombinedPlan, CombinedStatement},
    execution::ExecutionModel,
    prepare::PreparedPlan,
};
use crate::{LoopDomain, LoopKind, TargetCapability};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub(super) struct Domain {
    pub name: String,
    pub start: i64,
    pub stop: i64,
    pub step: i64,
    pub count: u64,
}

impl Domain {
    pub fn new(domain: &LoopDomain) -> Result<Self, EmitError> {
        let constant = |e: &crate::IndexExpr| {
            e.evaluate(&BTreeMap::new()).map_err(|e| {
                unsupported(format!(
                    "loop {} requires constant bounds: {e}",
                    domain.variable
                ))
            })
        };

        let (start, stop, step) = (
            constant(&domain.start)?,
            constant(&domain.stop)?,
            constant(&domain.step)?,
        );

        if start < 0 || stop <= start || step <= 0 {
            return Err(unsupported(format!(
                "loop {} requires a nonnegative, nonempty range and positive step",
                domain.variable
            )));
        }

        let span = stop - start;
        let count = ((span - 1) / step + 1) as u64;

        // The emitted for-loop also computes the successor of its last iteration.
        start
            .checked_add(
                ((count - 1) as i64)
                    .checked_mul(step)
                    .ok_or_else(|| invalid("loop coordinate overflow"))?,
            )
            .and_then(|last| last.checked_add(step))
            .ok_or_else(|| invalid("loop increment overflows int64"))?;
        Ok(Self {
            name: domain.variable.clone(),
            start,
            stop,
            step,
            count,
        })
    }
}

pub(super) enum DeviceStatement<'a> {
    Body(&'a CombinedBody),
    Sequential { domain: Domain, body: Vec<Self> },
}

pub(super) struct Kernel<'a> {
    pub domain: Vec<Domain>,
    pub body: Vec<DeviceStatement<'a>>,
    pub blocks: u64,
    pub threads: usize,
    pub shared: usize,
    pub alignment: usize,
}

#[derive(Serialize)]
pub(super) struct Launch {
    pub kernel: usize,
    pub base: u64,
    pub blocks: u32,
}

pub(super) struct Execution<'a> {
    pub kernels: Vec<Kernel<'a>>,
    pub launches: Vec<Launch>,
    pub requirements: CudaRequirements,
}

pub(super) const MAX_GRID_X: u64 = i32::MAX as u64;

pub(super) fn launches(kernel: usize, blocks: u64) -> Vec<Launch> {
    let mut out = Vec::new();
    let mut base = 0;
    while base < blocks {
        let count = (blocks - base).min(MAX_GRID_X);
        out.push(Launch {
            kernel,
            base,
            blocks: count as u32,
        });
        base += count;
    }
    out
}

pub(super) fn build<'a>(
    prepared: &PreparedPlan<'_>,
    buffers: &BufferBindings,
    combined: &'a CombinedPlan<'_>,
) -> Result<Execution<'a>, EmitError> {
    if combined.execution != ExecutionModel::CudaStreamed {
        return Err(unsupported("only Streamed execution is implemented"));
    }

    let mut kernels = Vec::new();

    partition(&combined.statements, &[], &mut kernels)?;

    let TargetCapability::Cuda(target) = prepared.plan.target();

    let mut requirements = buffers.requirements.clone();

    for kernel in &kernels {
        if kernel.shared > target.max_shared_memory_per_cta() {
            return Err(invalid("kernel shared memory exceeds target capacity"));
        }

        visit_bodies(&kernel.body, &mut |body| {
            for &(value, alignment) in &body.requirements.alignments {
                if alignment == 0 || !alignment.is_power_of_two() {
                    return Err(invalid("invalid buffer alignment"));
                }
                let slot = buffers
                    .slot(value)
                    .ok_or_else(|| invalid("memory requirement has no binding"))?;
                requirements[slot].alignment = requirements[slot].alignment.max(alignment);
            }

            Ok(())
        })?;
    }

    let launches = kernels
        .iter()
        .enumerate()
        .flat_map(|(i, k)| launches(i, k.blocks))
        .collect();

    let requirements = CudaRequirements {
        target,
        world_size: 1,
        buffers: requirements,
        workspace_bytes: 0,
        workspace_alignment: 1,
        workspace_symmetric: false,
        cooperative_launch: false,
        shared_memory_bytes: kernels.iter().map(|k| k.shared).max().unwrap_or(0),
        block_threads: kernels.iter().map(|k| k.threads).max().unwrap_or(128),
        nvshmem: false,
        nvls: false,
    };

    Ok(Execution {
        kernels,
        launches,
        requirements,
    })
}

fn device(statements: &[CombinedStatement]) -> Result<Vec<DeviceStatement<'_>>, EmitError> {
    statements
        .iter()
        .map(|s| match s {
            CombinedStatement::Body(b) => {
                for invocation in &b.roots {
                    if let Some(d) = &invocation.iteration {
                        Domain::new(d)?;
                    }
                }
                Ok(DeviceStatement::Body(b))
            }
            CombinedStatement::Loop {
                kind: LoopKind::Sequential,
                domain,
                body,
            } => Ok(DeviceStatement::Sequential {
                domain: Domain::new(domain)?,
                body: device(body)?,
            }),
            _ => Err(unsupported("parallel loop inside a sequential loop")),
        })
        .collect()
}

fn partition<'a>(
    statements: &'a [CombinedStatement],
    domain: &[Domain],
    kernels: &mut Vec<Kernel<'a>>,
) -> Result<(), EmitError> {
    for statement in statements {
        if let CombinedStatement::Loop {
            kind: LoopKind::Parallel,
            domain: d,
            body,
        } = statement
        {
            let mut nested = domain.to_vec();
            nested.push(Domain::new(d)?);
            partition(body, &nested, kernels)?;
        } else {
            let body = device(std::slice::from_ref(statement))?;
            let mut threads = None;
            let mut shared = 0;
            let mut alignment = 1;

            visit_bodies(&body, &mut |b| {
                let r = &b.requirements;
                if threads.is_some_and(|n| n != r.block_threads) {
                    return Err(unsupported(
                        "sequential bodies require matching CTA thread counts",
                    ));
                }
                threads = Some(r.block_threads);
                shared = shared.max(r.shared_memory_bytes);
                alignment = alignment.max(r.shared_memory_alignment);
                Ok(())
            })?;

            if !alignment.is_power_of_two() {
                return Err(invalid("invalid scratch alignment"));
            }

            let blocks = domain
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(d.count))
                .filter(|&n| n <= i64::MAX as u64)
                .ok_or_else(|| invalid("grid size overflows int64"))?;

            kernels.push(Kernel {
                domain: domain.to_vec(),
                body,
                blocks,
                threads: threads.unwrap_or(128),
                shared,
                alignment,
            });
        }
    }
    Ok(())
}

pub(super) fn visit_bodies(
    nodes: &[DeviceStatement<'_>],
    f: &mut impl FnMut(&CombinedBody) -> Result<(), EmitError>,
) -> Result<(), EmitError> {
    for node in nodes {
        match node {
            DeviceStatement::Body(b) => f(b)?,
            DeviceStatement::Sequential { body, .. } => visit_bodies(body, f)?,
        }
    }

    Ok(())
}

pub(super) fn coordinates(kernel: &Kernel<'_>, mut block: u64) -> BTreeMap<String, i64> {
    let mut out = BTreeMap::new();

    for d in kernel.domain.iter().rev() {
        let i = block % d.count;
        block /= d.count;
        out.insert(d.name.clone(), d.start + i as i64 * d.step);
    }

    out
}
