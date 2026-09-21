//! Common device buffer bindings.

use serde::Serialize;

use super::EmitError;
use crate::{PhysicalPlan, Storage, ValueInstanceId};

/// Allocation and launch-binding requirements for a device buffer.
#[derive(Debug, Clone, Serialize)]
pub struct BufferBindingRequirement {
    /// Dense index in the launch binding array; local phase values have no slot.
    pub value: usize,
    pub input_names: Vec<String>,
    pub output_name: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: crate::DType,
    pub strides: Vec<usize>,
    pub bytes: usize,
    pub alignment: usize,
    /// Whether this value is a program input/output rather than a Global intermediate.
    pub external: bool,
    pub symmetric: bool,
}

/// Buffer requirements and dense launch slots indexed by value ID.
pub(crate) struct BufferBindings {
    pub(super) requirements: Vec<BufferBindingRequirement>,
    slots: Vec<Option<usize>>,
}

impl BufferBindings {
    pub(crate) fn slot(&self, value: ValueInstanceId) -> Result<usize, EmitError> {
        self.slots
            .get(value.index())
            .copied()
            .flatten()
            .ok_or_else(|| EmitError::Contract("local value has no launch binding".into()))
    }
}

/// Build base bindings and their value-to-slot mapping.
pub(super) fn build_bindings(plan: &PhysicalPlan) -> BufferBindings {
    let mut buffers = Vec::new();
    let mut slots = vec![None; plan.value_instances().len()];

    // Group input names by dense value ID once, preserving binding order.
    let mut input_names = vec![Vec::new(); plan.value_instances().len()];

    for binding in plan.inputs() {
        let idx = binding.value().index();
        let tensor = binding.tensor().to_owned();

        input_names[idx].push(tensor);
    }

    for ((id, value), input_names) in plan.value_instances().zip(input_names) {
        if matches!(value.storage(), Storage::Shared | Storage::Register) {
            continue;
        }

        let shape = value.shape().to_vec();
        let elements = shape.iter().product::<usize>();
        let output_name = (plan.output().value() == id).then(|| plan.output().tensor().to_owned());

        slots[id.index()] = Some(buffers.len());

        buffers.push(BufferBindingRequirement {
            dtype: value.dtype(),
            // Contiguous row-major strides in elements: each axis skips the
            // product of the trailing dimensions (e.g. [2, 3, 4] -> [12, 4, 1]).
            strides: (0..shape.len())
                .map(|i| shape[i + 1..].iter().product())
                .collect(),
            value: buffers.len(),
            bytes: elements * value.dtype().size_bytes(),
            shape,
            alignment: 1,
            external: value.storage() == Storage::External,
            symmetric: false,
            input_names,
            output_name,
        });
    }

    BufferBindings {
        requirements: buffers,
        slots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CudaTargetCapability, DType, PhysicalPlanBuilder, TargetCapability};

    #[test]
    fn slots_skip_local_values_and_input_aliases_share_one_buffer() {
        let mut builder =
            PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
        let input = builder.add_value(DType::Bf16, [16], Storage::External);
        let shared = builder.add_value(DType::Bf16, [16], Storage::Shared);
        let first = builder.add_value(DType::Bf16, [16], Storage::Global);
        let register = builder.add_value(DType::Bf16, [16], Storage::Register);
        let second = builder.add_value(DType::Bf16, [16], Storage::Global);
        builder.bind_input("input", input);
        builder.bind_input("alias", input);
        let plan = builder.finalize("output", input).unwrap();
        let bindings = build_bindings(&plan);

        assert_eq!(bindings.requirements.len(), 3);
        assert_eq!(bindings.slot(input).unwrap(), 0);
        assert_eq!(bindings.slot(first).unwrap(), 1);
        assert_eq!(bindings.slot(second).unwrap(), 2);
        assert!(bindings.slot(shared).is_err());
        assert!(bindings.slot(register).is_err());
        assert_eq!(bindings.requirements[0].input_names, ["alias", "input"]);
        assert_eq!(
            bindings.requirements[0].output_name.as_deref(),
            Some("output")
        );
        assert_eq!(bindings.requirements[1].value, 1);
        assert_eq!(bindings.requirements[2].value, 2);
    }
}
