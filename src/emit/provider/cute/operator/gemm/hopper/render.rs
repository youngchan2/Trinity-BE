//! Rendering of one logical K iteration and its surrounding accumulator lifetime.

use crate::emit::provider::cute::operator::render::{output_code, render_template};
use crate::emit::provider::{KernelCode, RegisterBinding};

use super::super::super::render::{
    cpp_type, failed, render_bounds_predicate, render_kernel, render_origin, resolve_pointer,
    validate_prefix,
};
use super::HopperGemmSpecification;
use crate::emit::provider::{Kernel, KernelBindings, ProviderError};
use serde::Serialize;

#[derive(Serialize)]
struct GemmContext<'a> {
    prefix: &'a str,
    shared: &'a str,
    lhs: &'a str,
    rhs: &'a str,
    output_type: &'static str,
    lhs_stride: usize,
    rhs_stride: usize,
    output_stride: usize,
    rows: usize,
    k_width: usize,
    m_origin: String,
    n_origin: String,
    k_origin: String,
    valid_row: String,
}

pub(super) fn render(
    spec: &HopperGemmSpecification,
    bindings: &KernelBindings,
) -> Result<Kernel, ProviderError> {
    let prefix = validate_prefix(bindings)?;
    let shared = bindings
        .shared_memory
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| failed("missing Hopper GEMM shared-memory binding"))?;

    let context = GemmContext {
        prefix,
        shared,
        lhs: resolve_pointer(bindings, spec.lhs.value)?,
        rhs: resolve_pointer(bindings, spec.rhs.value)?,
        output_type: cpp_type(spec.output.dtype),
        lhs_stride: spec.lhs.shape[1],
        rhs_stride: spec.rhs.shape[1],
        output_stride: spec.output.shape[1],
        rows: spec.lhs.width(0),
        k_width: spec.lhs.width(1),
        m_origin: render_origin(&spec.output.axes[0], bindings)?,
        n_origin: render_origin(&spec.output.axes[1], bindings)?,
        k_origin: render_origin(&spec.lhs.axes[1], bindings)?,
        valid_row: render_bounds_predicate(&spec.output, 0, &format!("{prefix}_m + row")),
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
                coordinates: vec![format!("{prefix}_m + row"), format!("{prefix}_n + column")],
            },
        )?),
        after: include_str!("template/epilogue_end.cu.j2").into(),
    };
    if let Kernel::Native { epilogue, .. } = &mut kernel {
        *epilogue = code;
    }
    Ok(kernel)
}
