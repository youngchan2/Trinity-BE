//! Selects one provider and one kernel specification per operation.

use super::EmitError;
use super::execution::ExecutionModel;
use super::prepare::PreparedPlan;
use super::provider::{KernelContext, KernelProvider, ProviderError, SpecifiedKernel};
use crate::{Loop, OperationId, Statement};
use std::collections::BTreeMap;

pub(super) struct SelectedKernels<'provider> {
    pub kernels: BTreeMap<OperationId, SelectedKernel<'provider>>,
}

pub(super) struct SelectedKernel<'provider> {
    /// Registry position identifies the provider independently of its name or address.
    pub provider_index: usize,
    pub provider: &'provider dyn KernelProvider,
    pub specification: SpecifiedKernel,
}

pub(super) fn collect<'provider>(
    prepared: &PreparedPlan<'_>,
    execution: ExecutionModel,
    providers: &[&'provider dyn KernelProvider],
) -> Result<SelectedKernels<'provider>, EmitError> {
    let mut kernels = BTreeMap::new();
    visit(
        prepared,
        execution,
        providers,
        prepared.plan.statements(),
        &mut Vec::new(),
        &mut kernels,
    )?;
    Ok(SelectedKernels { kernels })
}

fn visit<'plan, 'provider>(
    prepared: &PreparedPlan<'plan>,
    execution: ExecutionModel,
    providers: &[&'provider dyn KernelProvider],
    statements: &'plan [Statement],
    loops: &mut Vec<&'plan Loop>,
    kernels: &mut BTreeMap<OperationId, SelectedKernel<'provider>>,
) -> Result<(), EmitError> {
    for statement in statements {
        match statement {
            Statement::Region(body) => visit(prepared, execution, providers, body, loops, kernels)?,
            Statement::Loop(loop_) => {
                loops.push(loop_);

                visit(prepared, execution, providers, &loop_.body, loops, kernels)?;

                loops.pop();
            }
            Statement::Operation(operation) => {
                let context = KernelContext {
                    prepared,
                    execution,
                    operation: *operation,
                    loops,
                };

                kernels.insert(*operation, select(&context, providers)?);
            }
        }
    }
    Ok(())
}

fn select<'provider>(
    context: &KernelContext<'_, '_>,
    providers: &[&'provider dyn KernelProvider],
) -> Result<SelectedKernel<'provider>, EmitError> {
    let mut reasons = Vec::new();
    for (provider_index, &provider) in providers.iter().enumerate() {
        match provider.specify(context) {
            Ok(specification) => {
                return Ok(SelectedKernel {
                    provider_index,
                    provider,
                    specification,
                });
            }
            Err(ProviderError::Unsupported(reason)) => {
                reasons.push(format!("{}: {reason}", provider.name()));
            }
            Err(ProviderError::Failed(message)) => {
                return Err(EmitError::Provider {
                    operation: context.operation.index(),
                    provider: provider.name().into(),
                    message,
                });
            }
        }
    }
    Err(EmitError::NoProvider {
        operation: context.operation.index(),
        reasons,
    })
}
