//! Program bindings shared by kernel providers.

use super::EmitError;
use crate::compile::BufferBindingRequirement;
use crate::{PhysicalPlan, Storage, ValueInstanceId};

pub(super) struct PreparedPlan<'a> {
    pub plan: &'a PhysicalPlan,
    pub bindings: BufferBindings,
}

pub(super) struct BufferBindings {
    pub requirements: Vec<BufferBindingRequirement>,
    slots: Vec<Option<usize>>,
}

impl BufferBindings {
    pub fn slot(&self, value: ValueInstanceId) -> Option<usize> {
        self.slots.get(value.index()).copied().flatten()
    }
}

pub(super) fn prepare(plan: &PhysicalPlan) -> Result<PreparedPlan<'_>, EmitError> {
    // Build buffer requirements and assign launch binding slots to External/Global values.
    let bindings = build_bindings(plan)?;

    Ok(PreparedPlan { plan, bindings })
}

/// Build base bindings and mapping.
fn build_bindings(plan: &PhysicalPlan) -> Result<BufferBindings, EmitError> {
    let mut requirements = Vec::new();
    let mut slots = vec![None; plan.value_instances().len()];

    // Group input names
    let mut input_names = vec![Vec::new(); plan.value_instances().len()];

    for binding in plan.inputs() {
        input_names[binding.value().index()].push(binding.tensor().to_owned());
    }

    for ((id, value), input_names) in plan.value_instances().zip(input_names) {
        // Local values do not need an external buffer or launch binding slot.
        if matches!(value.storage(), Storage::Shared | Storage::Register) {
            continue;
        }

        let shape = value.shape();
        let mut strides = vec![1; shape.len()];
        let mut elements = 1usize;

        // Contiguous row-major strides in elements: each axis skips the
        // product of the trailing dimensions (e.g. [2, 3, 4] -> [12, 4, 1]).
        for (axis, &extent) in shape.iter().enumerate().rev() {
            strides[axis] = elements;
            elements = elements
                .checked_mul(extent)
                .filter(|&n| n > 0 && n <= i64::MAX as usize)
                .ok_or_else(|| EmitError::InvalidExecution {
                    reason: "buffer extent/stride overflows int64 or is empty".into(),
                })?;
        }

        let bytes = elements
            .checked_mul(value.dtype().size_bytes())
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or_else(|| EmitError::InvalidExecution {
                reason: "buffer bytes overflow int64".into(),
            })?;

        // Slots stay contiguous even when local values are skipped.
        let slot = requirements.len();
        slots[id.index()] = Some(slot);

        requirements.push(BufferBindingRequirement {
            value: slot,
            input_names,
            output_name: (plan.output().value() == id).then(|| plan.output().tensor().to_owned()),
            shape: shape.to_vec(),
            dtype: value.dtype(),
            strides,
            bytes,
            alignment: value.dtype().size_bytes(),
            external: value.storage() == Storage::External,
            symmetric: false,
        });
    }

    Ok(BufferBindings {
        requirements,
        slots,
    })
}
