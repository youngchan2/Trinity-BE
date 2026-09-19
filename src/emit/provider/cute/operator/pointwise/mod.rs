//! CuTe pointwise specifications and basic arithmetic implementations.

use crate::emit::provider::{
    Kernel, KernelBindings, KernelContext, KernelImplementation, KernelRequirements, ProviderError,
    SpecifiedKernel, ThreadPolicy,
};
use crate::emit::provider::{KernelInterface, KernelPort, RegisterLayout};
use crate::{Expression, LoopKind, Storage};

mod render;
mod special;
#[cfg(test)]
mod tests;

use super::{access::Access, unsupported};

pub(in crate::emit::provider::cute) struct PointwiseKernel;

// Partition each internal tile across the resolved CTA, with a bounded register fragment.
const VALUES_PER_THREAD: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) struct PointwiseSpecification {
    pub requirements: KernelRequirements,
    pub inputs: Vec<Input>,
    pub output: Access,
    pub expression: Expression,
}

/// A load and its projection from output coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::emit) struct Input {
    pub access: Access,
    pub broadcast: bool,
}

impl KernelImplementation for PointwiseKernel {
    fn specify(&self, context: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError> {
        let operation = context
            .prepared
            .plan
            .operation(context.operation)
            .ok_or(ProviderError::Failed("missing operation".into()))?;

        let expression = operation.expression();
        let Expression::Store { destination, value } = expression else {
            return Err(unsupported("pointwise requires a store"));
        };

        // In a single-operation sequential loop, store(dst, load(dst) + rhs) with
        // a loop-invariant destination denotes accumulation starting from zero.
        // Reject it here: this implementation reads existing memory and does not
        // initialize an accumulator before the loop.
        if let Some(loop_) = context.loops.last()
            && loop_.kind == LoopKind::Sequential
            && loop_.body.len() == 1
            && crate::plan::accumulation_rhs(expression, &loop_.domain.variable).is_some()
        {
            return Err(unsupported(
                "pointwise does not implement zero-initialized accumulation",
            ));
        }

        let output = Access::from_plan(context, destination)?;
        if !(1..=2).contains(&output.shape.len()) {
            return Err(unsupported("CuTe pointwise supports rank 1 and 2"));
        }

        let mut inputs = Vec::new();

        check_pointwise(context, value, &output, false, &mut inputs)?;

        if inputs.iter().any(|i| i.access.storage == Storage::Shared)
            || output.storage == Storage::Shared
        {
            return Err(unsupported("pointwise Shared storage is not supported"));
        }

        if inputs
            .iter()
            .any(|i| i.broadcast && i.access.storage == Storage::Register)
        {
            return Err(unsupported(
                "register broadcast needs a thread redistribution contract",
            ));
        }

        let expression = value.as_ref().clone();

        let mut alignments: Vec<_> = inputs
            .iter()
            .map(|input| (input.access.value, input.access.dtype.size_bytes()))
            .collect();

        if !alignments.iter().any(|(value, _)| *value == output.value) {
            alignments.push((output.value, output.dtype.size_bytes()));
        }

        Ok(SpecifiedKernel::CuTePointwise(PointwiseSpecification {
            requirements: KernelRequirements {
                shared_memory_bytes: 0,
                shared_memory_alignment: 1,
                thread_policy: if inputs
                    .iter()
                    .any(|input| input.access.storage == Storage::Register)
                {
                    ThreadPolicy::FollowInput
                } else {
                    ThreadPolicy::Flexible {
                        supported: &[32, 64, 128, 256, 512, 1024],
                    }
                },
                alignments,
            },
            inputs,
            output,
            expression,
        }))
    }

    fn render(
        &self,
        specification: &SpecifiedKernel,
        bindings: &KernelBindings,
    ) -> Result<Kernel, ProviderError> {
        let SpecifiedKernel::CuTePointwise(specification) = specification else {
            return Err(ProviderError::Failed(
                "expected CuTe pointwise specification".into(),
            ));
        };

        render::render(specification, bindings)
    }
}

fn check_pointwise(
    context: &KernelContext<'_, '_>,
    expression: &Expression,
    output: &Access,
    broadcast: bool,
    inputs: &mut Vec<Input>,
) -> Result<(), ProviderError> {
    match expression {
        Expression::Load(source) => {
            let access = Access::from_plan(context, source)?;

            let matching = if broadcast {
                access.shape.len() == 1
                    && access.shape[0] == output.shape[0]
                    && access.axes[0] == output.axes[0]
            } else {
                access.shape == output.shape && access.axes == output.axes
            };
            if !matching {
                return Err(unsupported(
                    "pointwise operands need matching shapes and accesses after projection",
                ));
            }
            let input = Input { access, broadcast };
            if !inputs.contains(&input) {
                inputs.push(input);
            }
        }
        Expression::Broadcast { value, axis: 1 } if !broadcast && output.shape.len() == 2 => {
            return check_pointwise(context, value, output, true, inputs);
        }
        Expression::Constant(_)
        | Expression::Sqr(_)
        | Expression::Sqrt(_)
        | Expression::Sigmoid(_)
        | Expression::Relu(_)
        | Expression::Add(_)
        | Expression::Sub(_)
        | Expression::Mul(_)
        | Expression::Div(_) => {}
        _ => return Err(unsupported("operator is not supported by CuTe pointwise")),
    }

    for child in expression.children() {
        check_pointwise(context, child, output, broadcast, inputs)?;
    }

    Ok(())
}

fn add(left: &str, right: &str) -> String {
    format!("{left} + {right}")
}

fn sub(left: &str, right: &str) -> String {
    format!("{left} - {right}")
}

fn mul(left: &str, right: &str) -> String {
    format!("{left} * {right}")
}

fn div(left: &str, right: &str) -> String {
    format!("{left} / {right}")
}

fn sqr(operand: &str) -> String {
    format!("{operand} * {operand}")
}

fn sqrt(operand: &str) -> String {
    format!("sqrtf({operand})")
}

impl PointwiseSpecification {
    pub(in crate::emit) fn interface(&self, block_threads: usize) -> KernelInterface {
        let follows = self
            .inputs
            .iter()
            .any(|i| i.access.storage == Storage::Register);

        let layout = if follows {
            RegisterLayout::FollowInput {
                representation: "cuda.scalar",
            }
        } else {
            RegisterLayout::Fixed {
                representation: "cuda.scalar",
                distribution: format!("cute.pointwise.{block_threads}x4"),
            }
        };

        KernelInterface {
            inputs: self
                .inputs
                .iter()
                .map(|input| {
                    let mut port = KernelPort::memory(&input.access);

                    if input.access.storage == Storage::Register {
                        port.register = Some(RegisterLayout::FollowInput {
                            representation: "cuda.scalar",
                        });
                    }

                    port
                })
                .collect(),
            outputs: vec![KernelPort {
                access: self.output.clone(),
                register: Some(layout),
                projection: (0..self.output.axes.len()).collect(),
            }],
            iteration: None,
        }
    }
}
