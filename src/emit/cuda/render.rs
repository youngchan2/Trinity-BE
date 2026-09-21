use super::execution::{DeviceStatement, Execution, Launch};
use crate::emit::{
    EmitError, combine::CombinedPlan, prepare::PreparedPlan, provider::KernelBindings,
};
use crate::{CudaSource, CudaTargetCapability};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Serialize)]
struct Axis {
    start: i64,
    step: i64,
    count: u64,
    divisor: u64,
}

#[derive(Serialize)]
struct Kernel {
    code: String,
    axes: Vec<Axis>,
    coordinates: usize,
    threads: usize,
    shared: usize,
    alignment: usize,
}

#[derive(Serialize)]
struct Context<'a> {
    abi: &'static str,
    metadata: String,
    includes: BTreeSet<&'static str>,
    kernels: Vec<Kernel>,
    launches: &'a [Launch],
    buffers: &'a [crate::compile::BufferBindingRequirement],
    major: u32,
    minor: u32,
}

fn failed(e: impl std::fmt::Display) -> EmitError {
    EmitError::Render {
        message: e.to_string(),
    }
}

fn device(
    nodes: &[DeviceStatement<'_>],
    plan: &CombinedPlan<'_>,
    bindings: &KernelBindings,
    next: &mut usize,
    includes: &mut BTreeSet<&'static str>,
) -> Result<String, EmitError> {
    let mut code = String::new();

    for node in nodes {
        let id = *next;
        *next += 1;

        match node {
            DeviceStatement::Body(body) => {
                let mut bindings = bindings.clone();
                bindings.prefix = format!("{}_b{id}", bindings.prefix);

                let rendered = plan.render_body(body, &bindings).map_err(failed)?;
                includes.extend(rendered.includes);

                code.push_str(&rendered.code);
                // Also covers scratch reuse and memory handoff between bodies
                // and between iterations of an enclosing ordinary serial loop.
                code.push_str("\n__syncthreads();\n");
            }
            DeviceStatement::Sequential { domain, body } => {
                let name = format!("serial_{id}");
                let mut bindings = bindings.clone();

                bindings.indices.insert(domain.name.clone(), name.clone());

                let inner = device(body, plan, &bindings, next, includes)?;
                code.push_str(&format!(
                    "for (int64_t {name} = {}LL; {name} < {}LL; {name} += {}LL) {{\n{inner}\n}}\n",
                    domain.start, domain.stop, domain.step
                ));
            }
        }
    }
    Ok(code)
}

pub(super) fn render(
    prepared: &PreparedPlan<'_>,
    plan: &CombinedPlan<'_>,
    execution: &Execution<'_>,
) -> Result<CudaSource, EmitError> {
    let mut includes = BTreeSet::new();
    let mut kernels = Vec::new();
    let values = prepared
        .plan
        .value_instances()
        .filter_map(|(id, value)| {
            prepared.bindings.slot(id).map(|slot| {
                let dtype = match value.dtype() {
                    crate::DType::Bf16 => "cutlass::bfloat16_t",
                    crate::DType::Fp32 => "float",
                };

                (
                    id,
                    format!("static_cast<{dtype}*>(bindings.values[{slot}])"),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();

    for (id, k) in execution.kernels.iter().enumerate() {
        let bindings = KernelBindings {
            block_threads: k.threads,
            values: values.clone(),
            registers: BTreeMap::new(),
            indices: k
                .domain
                .iter()
                .enumerate()
                .map(|(i, d)| (d.name.clone(), format!("coordinates[{i}]")))
                .collect(),
            shared_memory: Some("scratch".into()),
            prefix: format!("kernel{id}"),
        };

        let code = device(&k.body, plan, &bindings, &mut 0, &mut includes)?;
        let axes = k
            .domain
            .iter()
            .enumerate()
            .map(|(i, d)| Axis {
                start: d.start,
                step: d.step,
                count: d.count,
                divisor: k.domain[i + 1..].iter().map(|d| d.count).product(),
            })
            .collect();

        kernels.push(Kernel {
            code,
            axes,
            coordinates: k.domain.len().max(1),
            threads: k.threads,
            shared: k.shared,
            alignment: k.alignment.max(16),
        });
    }

    let (major, minor) = match execution.requirements.target {
        CudaTargetCapability::Hopper => (9, 0),
        CudaTargetCapability::Sm89 => (8, 9),
        CudaTargetCapability::Sm120 => (12, 0),
    };

    let metadata = serde_json::to_string(&execution.requirements).map_err(failed)?;
    let context = Context {
        abi: include_str!("../../native/abi.h"),
        metadata: serde_json::to_string(&metadata).map_err(failed)?,
        includes,
        kernels,
        launches: &execution.launches,
        buffers: &execution.requirements.buffers,
        major,
        minor,
    };

    let mut env = minijinja::Environment::new();

    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env.add_template("program", include_str!("templates/program.cu.j2"))
        .map_err(failed)?;
    env.add_template("host", include_str!("templates/host.cu.j2"))
        .map_err(failed)?;

    let code = env
        .get_template("program")
        .map_err(failed)?
        .render(context)
        .map_err(failed)?;

    Ok(CudaSource {
        code,
        requirements: execution.requirements.clone(),
    })
}
