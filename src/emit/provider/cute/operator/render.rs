//! C++ bindings and template rendering shared by CuTe implementations.

use super::access::{Access, Axis};
use crate::emit::provider::{Kernel, KernelBindings, ProviderError};
use crate::{DType, ValueInstanceId};
use serde::Serialize;

pub(super) fn failed(message: impl Into<String>) -> ProviderError {
    ProviderError::Failed(message.into())
}

pub(super) fn render_coordinate(
    bindings: &KernelBindings,
    name: &str,
) -> Result<String, ProviderError> {
    bindings
        .indices
        .get(name)
        .filter(|expression| !expression.is_empty())
        .map(|expression| format!("({expression})"))
        .ok_or_else(|| failed(format!("missing loop binding {name}")))
}

pub(super) use crate::emit::provider::render_index_expression;

pub(super) fn cpp_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Fp32 => "float",
        DType::Bf16 => "cutlass::bfloat16_t",
    }
}

pub(super) fn validate_prefix(bindings: &KernelBindings) -> Result<&str, ProviderError> {
    let prefix = &bindings.prefix;
    if !prefix
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphabetic)
        || !prefix
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(failed(
            "body prefix must start with a letter and contain only identifier characters",
        ));
    }

    Ok(prefix)
}

pub(super) fn resolve_pointer(
    bindings: &KernelBindings,
    value: ValueInstanceId,
) -> Result<&str, ProviderError> {
    bindings
        .values
        .get(&value)
        .filter(|s| !s.is_empty())
        .map(String::as_str)
        .ok_or_else(|| failed(format!("missing buffer binding {}", value.index())))
}

pub(super) fn render_origin(
    axis: &Axis,
    bindings: &KernelBindings,
) -> Result<String, ProviderError> {
    match axis {
        Axis::Full => Ok("0".into()),
        Axis::Tile { variable, .. } => render_coordinate(bindings, variable),
        Axis::Element { variable, step } => Ok(format!(
            "({} / {})",
            render_coordinate(bindings, variable)?,
            render_index_expression(step, bindings)?
        )),
    }
}

/// Predicate for a logical coordinate along an access axis.
pub(super) fn render_bounds_predicate(access: &Access, axis: usize, coordinate: &str) -> String {
    if matches!(access.axes[axis], Axis::Tile { clipped: true, .. }) {
        format!(
            "({coordinate}) >= 0 && ({coordinate}) < int64_t({})",
            access.shape[axis]
        )
    } else {
        "true".into()
    }
}

pub(super) fn render_kernel(
    context: impl Serialize,
    prologue: &str,
    mainloop: Option<&str>,
    epilogue: &str,
) -> Result<Kernel, ProviderError> {
    let render = |name, template| render_template(&context, name, template);

    Ok(Kernel::Native {
        includes: vec![
            "cute/tensor.hpp",
            "cutlass/bfloat16.h",
            "cuda_runtime.h",
            "cstdint",
            "cmath",
        ],
        prologue: render("prologue", prologue)?.into(),
        mainloop: mainloop
            .map(|template| render("mainloop", template))
            .transpose()?
            .map(Into::into),
        epilogue: render("epilogue", epilogue)?.into(),
    })
}

pub(super) fn render_template(
    context: impl Serialize,
    name: &str,
    template: &str,
) -> Result<String, ProviderError> {
    let mut environment = minijinja::Environment::new();
    environment.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    environment.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment
        .render_str(template, context)
        .map_err(|e| failed(format!("CuTe {name}: {e}")))
}

/// Store a materialized result before exporting it to any connected continuation.
pub(super) fn output_code(
    access: &Access,
    bindings: &KernelBindings,
    result: crate::emit::provider::RegisterBinding,
) -> Result<crate::emit::provider::KernelCode, ProviderError> {
    use crate::emit::provider::KernelCode;
    let mut code = Vec::new();
    if access.storage != crate::Storage::Register {
        let target = memory_element(access, bindings, &result.coordinates)?;
        code.push(KernelCode::Text(format!("{target} = {};\n", result.value)));
    }
    code.push(KernelCode::Output {
        port: 0,
        binding: result,
    });
    Ok(KernelCode::Sequence(code))
}

pub(super) fn memory_element(
    access: &Access,
    bindings: &KernelBindings,
    coordinates: &[String],
) -> Result<String, ProviderError> {
    let pointer = resolve_pointer(bindings, access.value)?;
    let offset = coordinates
        .iter()
        .zip(&access.shape)
        .fold("0".to_owned(), |offset, (coordinate, extent)| {
            format!("(({offset}) * int64_t({extent}) + ({coordinate}))")
        });
    Ok(format!("({pointer})[{offset}]"))
}
