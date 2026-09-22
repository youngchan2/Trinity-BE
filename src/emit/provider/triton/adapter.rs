//! Operation candidates use the same typed program provider as whole programs.
use crate::emit::request::KernelRequest;
use crate::triton::{Error, Options, TritonPlan, invalid};
use crate::{PhysicalPlanBuilder, Statement, Storage};

pub(super) fn lower(request: &KernelRequest) -> Result<TritonPlan, Error> {
    let mut builder = PhysicalPlanBuilder::new(request.target, 1);
    let len = request
        .tensors
        .keys()
        .map(|id| id.index() + 1)
        .max()
        .unwrap_or(0);
    let mut ids = vec![usize::MAX; len];
    let mut values = std::collections::BTreeMap::new();
    for (id, t) in &request.tensors {
        let value = builder.add_named_value(
            format!("v{}", id.index()),
            t.dtype,
            t.shape.clone(),
            Storage::External,
        );
        ids[id.index()] = value.index();
        values.insert(*id, value);
        if request.inputs.contains(id) {
            builder.bind_input(format!("v{}", id.index()), value);
        }
    }
    let mut expression = request.expression.clone();
    expression.remap_values(&ids);
    let output = values[&request.output];
    let op = builder.add_operation(
        request.inputs.iter().map(|id| values[id]),
        [output],
        expression,
    );
    let physical = builder
        .build(
            vec![Statement::Operation(op)],
            format!("v{}", request.output.index()),
            output,
        )
        .map_err(|e| invalid(e.to_string()))?;
    Ok(super::TritonKernelProvider
        .lower_program(&physical, Options::default())?
        .into_plan())
}
