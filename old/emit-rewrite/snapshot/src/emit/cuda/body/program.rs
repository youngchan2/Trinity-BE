//! CUDA body composition, reuse, and resource collection.
use super::super::{EmitError, domain::TaskDomain};
use super::{Body, Resources, builder};
use crate::PhysicalPlan;
use crate::emit::BufferBindings;

/// A device body and its coordinate parameter order.
pub(in crate::emit::cuda) struct BodyDefinition {
    pub body: Body,
    pub parameters: Vec<String>,
}

pub(in crate::emit::cuda) struct ProgramBodies {
    pub bodies: Vec<BodyDefinition>,
    pub resources: Resources,
}

pub(in crate::emit::cuda) fn build_bodies(
    plan: &PhysicalPlan,
    bindings: &BufferBindings,
    domains: &mut [TaskDomain],
) -> Result<ProgramBodies, EmitError> {
    let mut bodies = Vec::new();
    let mut resources = Resources::default();

    for i in 0..domains.len() {
        // Reuse only the same computation and coordinate ABI. Access events,
        // concrete coordinates and readiness counts cannot affect this key.
        let previous = (0..i).find(|&j| {
            domains[j].statements == domains[i].statements
                && domains[j]
                    .task
                    .domain
                    .iter()
                    .map(|d| (&d.variable, &d.step))
                    .eq(domains[i]
                        .task
                        .domain
                        .iter()
                        .map(|d| (&d.variable, &d.step)))
        });

        let id = if let Some(j) = previous {
            domains[j].task.body
        } else {
            let body = builder::build(plan, bindings, &domains[i])?;

            resources.sequential(&body.resources());

            let id = bodies.len();

            bodies.push(BodyDefinition {
                body,
                parameters: domains[i]
                    .task
                    .domain
                    .iter()
                    .map(|d| d.variable.clone())
                    .collect(),
            });

            id
        };

        domains[i].task.body = id;
    }

    Ok(ProgramBodies { bodies, resources })
}
