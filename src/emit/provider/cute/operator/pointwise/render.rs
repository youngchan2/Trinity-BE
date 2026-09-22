//! Prepare the typed context for the CuTe pointwise template.

use super::super::access::{Access, Axis};
pub(super) use super::super::render::cpp_type;
use super::super::render::{
    failed, memory_element, output_code, render_coordinate, render_index_expression, render_kernel,
    render_template, validate_prefix,
};
use super::{
    Input, Kernel, KernelBindings, PointwiseSpecification, ProviderError, VALUES_PER_THREAD, add,
    div, mul, special, sqr, sqrt, sub,
};
use crate::emit::provider::{KernelCode, RegisterBinding};
use crate::{Constant, Expression, Storage};
use serde::Serialize;

#[derive(Serialize)]
struct TensorContext {
    name: String,
    pointer: String,
    shape: String,
    strides: String,
}

#[derive(Serialize)]
struct AxisContext {
    origin: String,
    width: usize,
}

#[derive(Serialize)]
struct ExpressionContext {
    name: String,
    value: String,
}

#[derive(Serialize)]
struct PointwiseContext<'a> {
    prefix: &'a str,
    tensors: Vec<TensorContext>,
    inputs: Vec<String>,
    axes: Vec<AxisContext>,
    thread_shape: String,
    tile_shape: String,
    predicate: String,
    expressions: Vec<ExpressionContext>,
    result: String,
    output_type: &'static str,
}

fn build_tensor_context(
    access: &Access,
    name: String,
    broadcast_columns: Option<usize>,
    bindings: &KernelBindings,
) -> Result<TensorContext, ProviderError> {
    let pointer = bindings
        .values
        .get(&access.value)
        .filter(|expression| !expression.is_empty())
        .ok_or_else(|| failed(format!("missing buffer binding {}", access.value.index())))?;

    let mut shape = access
        .shape
        .iter()
        .map(|n| format!("int64_t({n})"))
        .collect::<Vec<_>>()
        .join(", ");

    let mut strides = match access.shape.as_slice() {
        [_] => "int64_t(1)".to_owned(),
        [_, columns] => format!("int64_t({columns}), int64_t(1)"),
        _ => unreachable!("candidate rank was checked"),
    };

    // Lift a row vector to the output rank with a zero stride on the broadcast axis.
    if let Some(columns) = broadcast_columns {
        shape = format!("int64_t({}), int64_t({columns})", access.shape[0]);
        strides = "int64_t(1), int64_t(0)".into();
    }

    Ok(TensorContext {
        name,
        pointer: pointer.clone(),
        shape,
        strides,
    })
}

/// Name each intermediate once, so nested expressions are not duplicated in the template.
fn render_scalar(
    expression: &Expression,
    inputs: &[Input],
    broadcast: bool,
    prefix: &str,
    expressions: &mut Vec<ExpressionContext>,
) -> String {
    let value = match expression {
        Expression::Constant(constant) => {
            let bits = match constant {
                Constant::Integer(value) => (*value as f32).to_bits(),
                Constant::Float32(bits) => *bits,
                Constant::Float64(bits) => (f64::from_bits(*bits) as f32).to_bits(),
            };
            return format!("__uint_as_float({bits}u)");
        }
        Expression::Load(access) => {
            let index = inputs
                .iter()
                .position(|input| input.broadcast == broadcast && input.access.matches(access))
                .expect("specified input");

            return format!("{prefix}_load{index}");
        }
        Expression::Broadcast { value, axis: 1 } => {
            return render_scalar(value, inputs, true, prefix, expressions);
        }
        Expression::Sqr(operand)
        | Expression::Sqrt(operand)
        | Expression::Sigmoid(operand)
        | Expression::Relu(operand) => {
            let operand = render_scalar(operand, inputs, broadcast, prefix, expressions);

            match expression {
                Expression::Sqr(_) => sqr(&operand),
                Expression::Sqrt(_) => sqrt(&operand),
                Expression::Sigmoid(_) => special::sigmoid(&operand),
                Expression::Relu(_) => special::relu(&operand),
                _ => unreachable!("matched unary operator"),
            }
        }
        Expression::Add(values)
        | Expression::Sub(values)
        | Expression::Mul(values)
        | Expression::Div(values) => {
            let lhs = render_scalar(&values[0], inputs, broadcast, prefix, expressions);
            let rhs = render_scalar(&values[1], inputs, broadcast, prefix, expressions);

            match expression {
                Expression::Add(_) => add(&lhs, &rhs),
                Expression::Sub(_) => sub(&lhs, &rhs),
                Expression::Mul(_) => mul(&lhs, &rhs),
                Expression::Div(_) => div(&lhs, &rhs),
                _ => unreachable!("matched binary operator"),
            }
        }
        _ => unreachable!("candidate support was checked"),
    };

    let name = format!("{prefix}_expr{}", expressions.len());
    expressions.push(ExpressionContext {
        name: name.clone(),
        value,
    });

    name
}

fn build_context<'a>(
    specification: &PointwiseSpecification,
    bindings: &'a KernelBindings,
) -> Result<PointwiseContext<'a>, ProviderError> {
    let prefix = validate_prefix(bindings)?;

    let mut tensors = Vec::new();
    let mut inputs = Vec::new();
    for (index, input) in specification.inputs.iter().enumerate() {
        let name = format!("{prefix}_input{index}");
        tensors.push(build_tensor_context(
            &input.access,
            name.clone(),
            input.broadcast.then(|| specification.output.shape[1]),
            bindings,
        )?);
        inputs.push(name);
    }

    let mut axes = Vec::new();
    let mut predicates = Vec::new();
    for (axis, (access, size)) in specification
        .output
        .axes
        .iter()
        .zip(&specification.output.shape)
        .enumerate()
    {
        let (origin, width, clipped) = match access {
            Axis::Full => ("0".into(), *size, false),
            Axis::Tile {
                variable,
                width,
                clipped,
            } => (render_coordinate(bindings, variable)?, *width, *clipped),
            Axis::Element { variable, step } => (
                format!(
                    "({} / {})",
                    render_coordinate(bindings, variable)?,
                    render_index_expression(step, bindings)?
                ),
                1,
                false,
            ),
        };

        let coordinate = format!("({prefix}_origin{axis} + cute::get<{axis}>(coordinate))");
        if clipped {
            predicates.push(format!(
                "{coordinate} >= 0 && {coordinate} < int64_t({size})"
            ));
        }

        axes.push(AxisContext { origin, width });
    }

    // Tile the contiguous axis; CuTe partitions its values among the CTA's threads.
    let shape = |width| {
        let mut dimensions = vec!["cute::Int<1>{}".to_owned(); axes.len()];
        *dimensions.last_mut().unwrap() = format!("cute::Int<{width}>{{}}");
        dimensions.join(", ")
    };

    let thread_shape = shape(bindings.block_threads);
    let tile_shape = shape(bindings.block_threads * VALUES_PER_THREAD);
    let mut expressions = Vec::new();
    let result = render_scalar(
        &specification.expression,
        &specification.inputs,
        false,
        prefix,
        &mut expressions,
    );

    Ok(PointwiseContext {
        prefix,
        tensors,
        inputs,
        axes,
        thread_shape,
        tile_shape,
        predicate: if predicates.is_empty() {
            "true".into()
        } else {
            predicates.join(" && ")
        },
        expressions,
        result,
        output_type: cpp_type(specification.output.dtype),
    })
}

pub(super) fn render(
    specification: &PointwiseSpecification,
    bindings: &KernelBindings,
) -> Result<Kernel, ProviderError> {
    let register_ports: Vec<_> = specification
        .inputs
        .iter()
        .enumerate()
        .filter(|(_, input)| input.access.storage == Storage::Register)
        .map(|(port, _)| port)
        .collect();
    if register_ports.len() != bindings.registers.len()
        || register_ports
            .iter()
            .any(|port| !bindings.registers.contains_key(port))
    {
        return Err(failed(
            "register bindings differ from the specified input ports",
        ));
    }
    if !register_ports.is_empty() {
        return render_element(specification, bindings);
    }
    let context = build_context(specification, bindings)?;
    let prefix = validate_prefix(bindings)?;
    let coordinates = (0..specification.output.axes.len())
        .map(|axis| {
            format!(
                "{prefix}_origin{axis} + cute::get<{axis}>({prefix}_thread_coordinates(element))"
            )
        })
        .collect();
    let mut kernel = render_kernel(&context, include_str!("template/prologue.cu.j2"), None, "")?;
    if let Kernel::Native { epilogue, .. } = &mut kernel {
        *epilogue = KernelCode::Scope {
            before: render_template(
                &context,
                "epilogue",
                include_str!("template/epilogue.cu.j2"),
            )?,
            body: Box::new(output_code(
                &specification.output,
                bindings,
                RegisterBinding {
                    value: format!("{prefix}_result"),
                    coordinates,
                },
            )?),
            after: include_str!("template/epilogue_end.cu.j2").into(),
        };
    }
    Ok(kernel)
}

/// Elementwise consumption inside the producer's guarded output scope.
fn render_element(
    spec: &PointwiseSpecification,
    bindings: &KernelBindings,
) -> Result<Kernel, ProviderError> {
    let prefix = validate_prefix(bindings)?;
    let coordinates = &bindings
        .registers
        .values()
        .next()
        .ok_or_else(|| failed("missing register input"))?
        .coordinates;
    let mut inputs = Vec::new();
    for (port, input) in spec.inputs.iter().enumerate() {
        let value = if input.access.storage == Storage::Register {
            let register = bindings
                .registers
                .get(&port)
                .ok_or_else(|| failed("missing register port"))?;
            register.value.clone()
        } else {
            let projection = if input.broadcast {
                &coordinates[..1]
            } else {
                coordinates.as_slice()
            };
            memory_element(&input.access, bindings, projection)?
        };
        inputs.push(value);
    }
    let mut expressions = Vec::new();
    let result = render_scalar(
        &spec.expression,
        &spec.inputs,
        false,
        prefix,
        &mut expressions,
    );
    #[derive(Serialize)]
    struct ElementContext<'a> {
        prefix: &'a str,
        inputs: Vec<String>,
        expressions: Vec<ExpressionContext>,
        result: String,
        output_type: &'static str,
    }
    let context = ElementContext {
        prefix,
        inputs,
        expressions,
        result,
        output_type: cpp_type(spec.output.dtype),
    };
    let mut kernel = render_kernel(&context, "", None, "")?;
    if let Kernel::Native { epilogue, .. } = &mut kernel {
        *epilogue = KernelCode::Scope {
            before: render_template(&context, "element", include_str!("template/element.cu.j2"))?,
            body: Box::new(output_code(
                &spec.output,
                bindings,
                RegisterBinding {
                    value: format!("{prefix}_result"),
                    coordinates: coordinates.clone(),
                },
            )?),
            after: "}\n".into(),
        };
    }
    Ok(kernel)
}
