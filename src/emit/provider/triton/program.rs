//! Whole scheduled-region fallback. Operations inside a region are never split
//! into independent launches merely to fit the operation-candidate interface.
use super::TritonKernelProvider;
use crate::analysis::{self, Bindings, ScheduledIr, TensorMetadata};
use crate::triton::{self, Error, Options, TritonPlan, invalid};
use crate::{PhysicalPlan, PhysicalPlanBuilder, ScheduledConfig, Storage};

#[derive(Clone)]
pub struct TritonProgram {
    physical: PhysicalPlan,
    plan: TritonPlan,
}
impl TritonProgram {
    pub fn physical_plan(&self) -> &PhysicalPlan {
        &self.physical
    }
    pub fn plan(&self) -> &TritonPlan {
        &self.plan
    }
    pub fn emit(&self) -> String {
        self.plan.emit()
    }
    pub(crate) fn into_plan(self) -> TritonPlan {
        self.plan
    }
}

impl TritonKernelProvider {
    /// Specify an entire selected program from the common plan. No re-parsing or
    /// scheduling occurs: the scope/access tables are projected from typed data.
    pub fn lower_program(
        &self,
        physical: &PhysicalPlan,
        mut options: Options,
    ) -> Result<TritonProgram, Error> {
        if physical.world_size() != 1 {
            return Err(invalid("Triton fallback requires single-GPU computation"));
        }
        let projected = analysis::from_physical(physical)?;
        for (_, value) in physical.value_instances() {
            if value.storage() == Storage::Shared {
                return Err(invalid(
                    "explicit Shared transport needs a provider layout contract",
                ));
            }
            let name = value.name().unwrap();
            if let Some(shape) = options.shapes.get(name)
                && shape != value.shape()
            {
                return Err(invalid(format!(
                    "shape override differs from PhysicalPlan value {name}"
                )));
            }
            options.shapes.insert(name.into(), value.shape().to_vec());
            let dtype = value.dtype().into();
            if options.dtypes.get(name).is_some_and(|old| *old != dtype) {
                return Err(invalid(format!(
                    "dtype override differs from PhysicalPlan value {name}"
                )));
            }
            options.dtypes.insert(name.into(), dtype);
        }
        let mut symbols = physical.bindings().clone();
        symbols.extend(options.symbols);
        options.symbols = symbols;
        let contracts = physical
            .value_instances()
            .map(|(id, value)| (analysis::TensorId(id.index()), value.storage()))
            .collect();
        let mut plan = triton::lowering::lower_projected(projected, options, contracts)?;
        plan.metadata.output_order = physical
            .outputs()
            .iter()
            .map(|b| analysis::TensorId(b.value().index()))
            .collect();
        Ok(TritonProgram {
            physical: physical.clone(),
            plan,
        })
    }

    /// Source frontend: shared collection -> common PhysicalPlan -> provider.
    /// Defaults for free tuning parameters are a provider decision; common
    /// construction receives those bindings without selecting a new schedule.
    pub fn lower_source(
        &self,
        source: ScheduledIr,
        mut options: Options,
    ) -> Result<TritonProgram, Error> {
        let mut bindings = Bindings {
            shapes: options.shapes.clone(),
            symbols: options.symbols.clone(),
        };
        let metadata = TensorMetadata::collect(&source, &mut bindings)?;
        let tuning =
            triton::lowering::metadata::resolve(&source, &mut bindings, &metadata, &mut options)?;
        for (name, candidates) in tuning.candidates {
            options.tuning.entry(name).or_insert(candidates);
        }
        let physical = PhysicalPlanBuilder::from_scheduled(
            &source,
            ScheduledConfig {
                target: crate::TargetCapability::Cuda(crate::CudaTargetCapability::Hopper),
                bindings,
                default_dtype: options.default_dtype.into(),
                dtypes: options
                    .dtypes
                    .iter()
                    .map(|(n, d)| (n.clone(), (*d).into()))
                    .collect(),
            },
        )
        .map_err(|e| invalid(e.to_string()))?;
        self.lower_program(&physical, options)
    }
}
