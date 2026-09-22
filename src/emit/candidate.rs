//! Candidate discovery, without selection, compilation, or runtime availability claims.

use super::EmitError;
use super::execution::plan_execution;
use super::prepare::{PreparedPlan, prepare};
use super::provider::quack::{QuackKernelProvider, QuackSpecification};
use super::provider::triton::{TritonKernelProvider, TritonSpecification};
use super::provider::{CuTeKernelProvider, KernelContext, KernelProvider, ProviderError};
use crate::emit::native::SpecifiedKernel;
use crate::{Loop, OperationId, PhysicalPlan, Statement};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    Native,
    Opaque,
}

#[derive(Debug, Clone)]
pub(super) enum CandidateSpecification {
    Native(SpecifiedKernel),
    Quack(QuackSpecification),
    Triton(Box<TritonSpecification>),
}

/// A candidate covers exactly one operation in the supplied PhysicalPlan, in its
/// original loop context. This is not a whole-program alternative to TritonPlan.
#[derive(Debug, Clone)]
pub struct KernelCandidate {
    operation: OperationId,
    provider: String,
    specification: CandidateSpecification,
}

impl KernelCandidate {
    pub fn operation(&self) -> OperationId {
        self.operation
    }
    pub fn provider(&self) -> &str {
        &self.provider
    }
    pub fn kind(&self) -> CandidateKind {
        match self.specification {
            CandidateSpecification::Native(_) => CandidateKind::Native,
            CandidateSpecification::Quack(_) | CandidateSpecification::Triton(_) => {
                CandidateKind::Opaque
            }
        }
    }
    pub fn quack(&self) -> Option<&QuackSpecification> {
        match &self.specification {
            CandidateSpecification::Quack(s) => Some(s),
            _ => None,
        }
    }
    pub fn triton(&self) -> Option<&TritonSpecification> {
        match &self.specification {
            CandidateSpecification::Triton(s) => Some(s),
            _ => None,
        }
    }
    /// Independent host-call source with a uniform run(values) ABI. Native
    /// fragments need CUDA composition and cannot be launched by this route.
    pub fn emit_python(&self) -> Option<String> {
        match &self.specification {
            CandidateSpecification::Quack(s) => Some(s.emit_python()),
            CandidateSpecification::Triton(s) => Some(s.emit_python()),
            CandidateSpecification::Native(_) => None,
        }
    }
    /// A native implementation may cover the operation's enclosing reduction
    /// loop. That scope must be included when comparing its execution cost.
    pub fn iteration(&self) -> Option<&str> {
        match &self.specification {
            CandidateSpecification::Native(s) => s.iteration(),
            CandidateSpecification::Quack(_) | CandidateSpecification::Triton(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRejection {
    pub provider: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default)]
pub struct OperationCandidates {
    pub candidates: Vec<KernelCandidate>,
    pub rejections: Vec<CandidateRejection>,
}

/// Enumerate CuTe, Triton and optional Quack implementations. No package import, JIT,
/// benchmark, or winner selection occurs here. Empty candidate lists retain
/// support diagnostics. Internal provider errors remain errors.
///
/// This API does not change the legacy native-only `emit` path. Independent
/// candidates can be assembled and selected through `emit_python`.
pub fn kernel_candidates(
    plan: &PhysicalPlan,
) -> Result<BTreeMap<OperationId, OperationCandidates>, EmitError> {
    discover(
        plan,
        &[
            &CuTeKernelProvider,
            &TritonKernelProvider,
            &QuackKernelProvider,
        ],
    )
}

fn discover(
    plan: &PhysicalPlan,
    providers: &[&dyn KernelProvider],
) -> Result<BTreeMap<OperationId, OperationCandidates>, EmitError> {
    let prepared = prepare(plan)?;
    let mut result = BTreeMap::new();
    visit(
        &prepared,
        providers,
        plan.statements(),
        &mut Vec::new(),
        &mut result,
    )?;
    Ok(result)
}

fn visit<'p>(
    prepared: &PreparedPlan<'p>,
    providers: &[&dyn KernelProvider],
    statements: &'p [Statement],
    loops: &mut Vec<&'p Loop>,
    result: &mut BTreeMap<OperationId, OperationCandidates>,
) -> Result<(), EmitError> {
    for statement in statements {
        match statement {
            Statement::Region(body) => visit(prepared, providers, body, loops, result)?,
            Statement::Loop(l) => {
                loops.push(l);
                visit(prepared, providers, &l.body, loops, result)?;
                loops.pop();
            }
            Statement::Operation(operation) => {
                let context = KernelContext {
                    prepared,
                    operation: *operation,
                    execution: plan_execution(prepared.plan),
                    loops,
                };
                let mut entry = OperationCandidates::default();
                for provider in providers {
                    match provider.candidates(&context) {
                        Ok(specs) if specs.is_empty() => {
                            entry.rejections.push(CandidateRejection {
                                provider: provider.name().into(),
                                reason: "no applicable implementations".into(),
                            })
                        }
                        Ok(specs) => {
                            entry
                                .candidates
                                .extend(specs.into_iter().map(|specification| KernelCandidate {
                                    operation: *operation,
                                    provider: provider.name().into(),
                                    specification,
                                }))
                        }
                        Err(ProviderError::Unsupported(reason)) => {
                            entry.rejections.push(CandidateRejection {
                                provider: provider.name().into(),
                                reason,
                            })
                        }
                        Err(ProviderError::Failed(message)) => {
                            return Err(EmitError::Provider {
                                operation: operation.index(),
                                provider: provider.name().into(),
                                message,
                            });
                        }
                    }
                }
                result.insert(*operation, entry);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_retains_alternatives_instead_of_selecting_the_first() {
        let plan = super::super::combine::tests::pipelines().remove(0);
        let all = discover(&plan, &[&CuTeKernelProvider, &CuTeKernelProvider]).unwrap();
        assert!(all.values().all(|entry| entry.candidates.len() == 2));
    }
}
