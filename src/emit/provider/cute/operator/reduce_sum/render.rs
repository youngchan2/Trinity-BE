use super::super::render::{
    render_bounds_predicate, render_kernel, render_origin, resolve_pointer, validate_prefix,
};
use super::ReduceSumSpecification;
use crate::emit::provider::cute::operator::render::{output_code, render_template};
use crate::emit::provider::{Kernel, KernelBindings, ProviderError};
use crate::emit::provider::{KernelCode, RegisterBinding};
use serde::Serialize;

#[derive(Serialize)]
struct ReduceContext<'a> {
    prefix: &'a str,
    input: &'a str,
    stride: usize,
    rows: usize,
    columns: usize,
    row_groups: usize,
    row_origin: String,
    column_origin: String,
    valid_row: String,
    valid_column: String,
    square: bool,
}

pub(super) fn render(
    spec: &ReduceSumSpecification,
    bindings: &KernelBindings,
) -> Result<Kernel, ProviderError> {
    let prefix = validate_prefix(bindings)?;
    let context = ReduceContext {
        prefix,
        input: resolve_pointer(bindings, spec.input.value)?,
        stride: spec.input.shape[1],
        rows: spec.input.width(0),
        columns: spec.input.width(1),
        row_groups: spec.input.width(0).div_ceil(4),
        row_origin: render_origin(&spec.input.axes[0], bindings)?,
        column_origin: render_origin(&spec.input.axes[1], bindings)?,
        valid_row: render_bounds_predicate(&spec.input, 0, &format!("{prefix}_row_origin + row")),
        valid_column: render_bounds_predicate(&spec.input, 1, "column_origin + column"),
        square: spec.square,
    };
    let mut kernel = render_kernel(
        &context,
        include_str!("template/prologue.cu.j2"),
        Some(include_str!("template/mainloop.cu.j2")),
        "",
    )?;
    let code = KernelCode::Scope {
        before: render_template(
            &context,
            "epilogue",
            include_str!("template/epilogue.cu.j2"),
        )?,
        body: Box::new(output_code(
            &spec.output,
            bindings,
            RegisterBinding {
                value: format!("{prefix}_result"),
                coordinates: vec![format!("{prefix}_row_origin + row")],
            },
        )?),
        after: include_str!("template/epilogue_end.cu.j2").into(),
    };
    if let Kernel::Native { epilogue, .. } = &mut kernel {
        *epilogue = code;
    }
    Ok(kernel)
}
