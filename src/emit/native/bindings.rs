//! Dense buffer slots and allocation requirements for the native CUDA host ABI.
use crate::compile::BufferBindingRequirement;
use crate::emit::EmitError;
use crate::{PhysicalPlan, Storage, ValueInstanceId};

pub(in crate::emit) struct BufferBindings {
    pub requirements: Vec<BufferBindingRequirement>,
    slots: Vec<Option<usize>>,
}

impl BufferBindings {
    pub fn slot(&self, value: ValueInstanceId) -> Option<usize> {
        self.slots.get(value.index()).copied().flatten()
    }
}

/// Build base bindings and mapping.
pub(in crate::emit) fn build(plan: &PhysicalPlan) -> Result<BufferBindings, EmitError> {
    if plan.outputs().len() != 1 || !plan.mutable_inputs().is_empty() {
        return Err(EmitError::UnsupportedExecution { reason: "native CUDA ABI requires one output and immutable inputs; use the Triton program provider".into() });
    }
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
