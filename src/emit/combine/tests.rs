use super::*;
use crate::emit::provider::{
    CuTeKernelProvider, Kernel, KernelBindings, KernelContext, KernelProvider, ProviderError,
    SpecifiedKernel,
};
use crate::emit::{collect::collect, execution::plan_execution, prepare::prepare};
use crate::{
    AccessIndex as I, DType, Expression as E, IndexExpr, Loop, LoweringConfig, PhysicalPlanBuilder,
    TensorAccess,
};

fn load(value: ValueInstanceId, rank: usize) -> E {
    E::Load(TensorAccess::new(value, vec![I::FullTile; rank]))
}
fn store(
    builder: &mut PhysicalPlanBuilder,
    output: ValueInstanceId,
    rank: usize,
    inputs: &[ValueInstanceId],
    value: E,
) -> Statement {
    Statement::Operation(builder.add_operation(
        inputs.iter().copied(),
        [output],
        E::Store {
            destination: TensorAccess::new(output, vec![I::FullTile; rank]),
            value: Box::new(value),
        },
    ))
}
fn loop_(
    kind: LoopKind,
    variable: &str,
    start: i64,
    stop: i64,
    step: i64,
    body: Vec<Statement>,
) -> Statement {
    Statement::Loop(Loop {
        kind,
        domain: LoopDomain {
            variable: variable.into(),
            start: IndexExpr::Constant(start),
            stop: IndexExpr::Constant(stop),
            step: IndexExpr::Constant(step),
        },
        body,
    })
}

/// Root implementations have different thread mappings; consumers use the same contract.
pub(in crate::emit) fn pipelines() -> Vec<PhysicalPlan> {
    (0..3)
        .map(|kind| pipeline(kind, Storage::Register))
        .collect()
}

fn pipeline(kind: usize, storage: Storage) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
    let (shape, input_shape) = match kind {
        0 => (vec![5, 131], vec![5, 131]),
        1 => (vec![16, 128], vec![16, 256]),
        _ => (vec![5], vec![5, 131]),
    };
    let dtype = if kind == 2 { DType::Fp32 } else { DType::Bf16 };
    let x = b.add_value(dtype, input_shape, Storage::External);
    b.bind_input("X", x);
    let r = b.add_value(dtype, shape.clone(), storage);
    let s = b.add_value(dtype, shape.clone(), storage);
    let y = b.add_value(dtype, shape.clone(), Storage::External);
    let rank = shape.len();
    let producer = if kind == 0 {
        store(&mut b, r, rank, &[x], E::Sqr(Box::new(load(x, rank))))
    } else {
        let tile = |width: usize| I::Tile {
            variable: "k".into(),
            width: width.into(),
        };
        let rhs = if kind == 1 {
            let weight = b.add_value(DType::Bf16, [256, 128], Storage::External);
            b.bind_input("W", weight);
            E::Matmul(Box::new([
                E::Load(TensorAccess::new(x, [I::FullTile, tile(128)])),
                E::Load(TensorAccess::new(weight, [tile(128), I::FullTile])),
            ]))
        } else {
            E::ReduceSum {
                value: Box::new(E::Load(TensorAccess::new(
                    x,
                    [
                        I::FullTile,
                        I::ClippedTile {
                            variable: "k".into(),
                            width: 65usize.into(),
                        },
                    ],
                ))),
                axis: 1,
            }
        };
        let inputs: Vec<_> = if kind == 1 {
            vec![
                x,
                match &rhs {
                    E::Matmul(args) => match &args[1] {
                        E::Load(a) => a.value,
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
            ]
        } else {
            vec![x]
        };
        let update = store(
            &mut b,
            r,
            rank,
            &inputs,
            E::Add(Box::new([load(r, rank), rhs])),
        );
        if kind == 1 {
            loop_(LoopKind::Sequential, "k", 8, 136, 64, vec![update])
        } else {
            loop_(LoopKind::Sequential, "k", 0, 195, 65, vec![update])
        }
    };
    let first = store(&mut b, s, rank, &[r], E::Sigmoid(Box::new(load(r, rank))));
    // Both the root result and the first consumer remain live here.
    let second = store(
        &mut b,
        y,
        rank,
        &[r, s],
        E::Mul(Box::new([load(r, rank), load(s, rank)])),
    );
    b.build(
        vec![loop_(
            LoopKind::Parallel,
            "p",
            0,
            1,
            1,
            vec![producer, first, second],
        )],
        "Y",
        y,
    )
    .unwrap()
}

fn bindings(plan: &PhysicalPlan) -> KernelBindings {
    KernelBindings {
        block_threads: 128,
        values: plan
            .value_instances()
            .filter(|(_, v)| v.storage() != Storage::Register)
            .map(|(id, _)| (id, format!("buffer{}", id.index())))
            .collect(),
        registers: BTreeMap::new(),
        indices: BTreeMap::new(),
        shared_memory: Some("scratch".into()),
        prefix: "combined".into(),
    }
}
fn bodies(statements: &[CombinedStatement]) -> Vec<&CombinedBody> {
    statements
        .iter()
        .flat_map(|s| match s {
            CombinedStatement::Body(b) => vec![b],
            CombinedStatement::Loop { body, .. } => bodies(body),
        })
        .collect()
}

#[test]
fn forwards_registers_in_each_producers_output_scope_with_rounding_and_fanout() {
    for plan in pipelines() {
        let prepared = prepare(&plan).unwrap();
        let provider = CuTeKernelProvider;
        let selected = collect(&prepared, plan_execution(&plan), &[&provider]).unwrap();
        let combined = combine(&prepared, plan_execution(&plan), selected).unwrap();
        let body = bodies(&combined.statements)[0];
        assert!(
            body.roots
                .iter()
                .flat_map(|root| {
                    std::iter::once(root.operation).chain(root.followers.iter().copied())
                })
                .eq(plan.statements().iter().flat_map(Statement::operations))
        );
        assert_eq!(body.roots.len(), 1);
        assert_eq!(body.connections.len(), 3);
        let rendered = combined.render_body(body, &bindings(&plan)).unwrap();
        assert!(rendered.includes.contains(&"cute/tensor.hpp"));
        for (id, value) in plan
            .value_instances()
            .filter(|(_, v)| v.storage() == Storage::Register)
        {
            assert!(!rendered.code.contains(&format!("buffer{}", id.index())));
            if value.dtype() == DType::Bf16 {
                assert!(rendered.code.contains("static_cast<cutlass::bfloat16_t>"));
            }
        }
        let end = rendered.code.find("_op2_load0").unwrap();
        assert!(rendered.code[..end].contains("_op0_result"));
        assert!(rendered.code[..end].contains("_op1_result"));
        if body.roots[0].iteration.is_some() {
            let initial = rendered.code.find("accumulator").unwrap();
            let loop_start = rendered.code.find("for (int64_t combined_op0_").unwrap();
            assert!(initial < loop_start);
            assert!(rendered.code.contains("_op0_result ="));
        }
    }
}

#[test]
fn global_values_keep_stores_and_uniform_barriers() {
    let plan = pipeline(0, Storage::Global);
    let prepared = prepare(&plan).unwrap();
    let provider = CuTeKernelProvider;
    let selected = collect(&prepared, plan_execution(&plan), &[&provider]).unwrap();
    let combined = combine(&prepared, plan_execution(&plan), selected).unwrap();
    let body = bodies(&combined.statements)[0];
    assert_eq!(body.roots.len(), 3);
    assert!(body.connections.is_empty());
    let code = combined.render_body(body, &bindings(&plan)).unwrap().code;
    assert_eq!(code.matches("__syncthreads();").count(), 2);
    for (id, _) in plan.value_instances() {
        assert!(code.contains(&format!("buffer{}", id.index())));
    }
}

struct RecordingProvider {
    unsupported_operation: Option<usize>,
    specified: std::cell::RefCell<Vec<usize>>,
    rendered: std::cell::Cell<usize>,
}

impl RecordingProvider {
    fn new(unsupported_operation: Option<usize>) -> Self {
        Self {
            unsupported_operation,
            specified: Default::default(),
            rendered: Default::default(),
        }
    }
}

impl KernelProvider for RecordingProvider {
    fn name(&self) -> &str {
        "recording"
    }

    fn specify(&self, context: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError> {
        let operation = context.operation.index();
        self.specified.borrow_mut().push(operation);
        if self.unsupported_operation == Some(operation) {
            return Err(ProviderError::Unsupported("unsupported operation".into()));
        }
        CuTeKernelProvider.specify(context)
    }

    fn render(&self, s: &SpecifiedKernel, b: &KernelBindings) -> Result<Kernel, ProviderError> {
        self.rendered.set(self.rendered.get() + 1);
        CuTeKernelProvider.render(s, b)
    }
}

#[test]
fn selects_first_supported_provider_per_operation_in_both_execution_models() {
    let plan = pipeline(0, Storage::Register);
    let prepared = prepare(&plan).unwrap();
    for execution in [ExecutionModel::CudaStreamed, ExecutionModel::CudaPersistent] {
        let first = RecordingProvider::new(None);
        let second = RecordingProvider::new(None);
        let selected = collect(&prepared, execution, &[&first, &second]).unwrap();
        let combined = combine(&prepared, execution, selected).unwrap();
        assert_eq!(combined.execution, execution);
        let body = bodies(&combined.statements)[0];
        assert_eq!(body.connections.len(), 3);
        assert!(
            body.roots
                .iter()
                .flat_map(|root| {
                    std::iter::once(root.operation).chain(root.followers.iter().copied())
                })
                .eq(plan.statements().iter().flat_map(Statement::operations))
        );
        combined.render_body(body, &bindings(&plan)).unwrap();
        assert_eq!(*first.specified.borrow(), [0, 1, 2]);
        assert!(second.specified.borrow().is_empty());
        assert_eq!(first.rendered.get(), 3);
        assert_eq!(second.rendered.get(), 0);
    }
}

#[test]
fn different_providers_share_a_plan_but_never_a_body_even_with_matching_names() {
    let plan = pipeline(0, Storage::Global);
    let prepared = prepare(&plan).unwrap();
    for execution in [ExecutionModel::CudaStreamed, ExecutionModel::CudaPersistent] {
        for unsupported_operation in [1, 2] {
            let first = RecordingProvider::new(Some(unsupported_operation));
            let second = RecordingProvider::new(Some(0));
            let mut selected = collect(&prepared, execution, &[&first, &second]).unwrap();
            for kernel in selected.kernels.values_mut() {
                kernel.specification.requirements_mut().thread_policy = ThreadPolicy::Flexible {
                    supported: if kernel.provider_index == 0 {
                        &[64]
                    } else {
                        &[256]
                    },
                };
            }
            let combined = combine(&prepared, execution, selected).unwrap();
            let bodies = bodies(&combined.statements);
            assert_eq!(bodies.len(), if unsupported_operation == 1 { 3 } else { 2 });
            assert!(
                bodies
                    .iter()
                    .flat_map(|body| body.roots.iter().map(|root| root.operation))
                    .eq(plan.statements().iter().flat_map(Statement::operations))
            );
            for body in bodies {
                let provider = combined.kernels[&body.roots[0].operation].provider_index;
                let threads = if provider == 0 { 64 } else { 256 };
                assert_eq!(body.requirements.block_threads, threads);
                assert!(
                    body.roots
                        .iter()
                        .all(|root| combined.kernels[&root.operation].provider_index == provider)
                );
                let code = combined.render_body(body, &bindings(&plan)).unwrap().code;
                assert!(code.contains(&format!("cute::Int<{threads}>")));
            }
            assert_eq!(*first.specified.borrow(), [0, 1, 2]);
            assert_eq!(*second.specified.borrow(), [unsupported_operation]);
            assert_eq!(first.rendered.get(), 2);
            assert_eq!(second.rendered.get(), 1);
        }
    }
}

#[test]
fn register_edges_cannot_cross_providers_or_trigger_reselection() {
    let plan = pipeline(0, Storage::Register);
    let prepared = prepare(&plan).unwrap();
    for execution in [ExecutionModel::CudaStreamed, ExecutionModel::CudaPersistent] {
        let first = RecordingProvider::new(Some(1));
        let second = RecordingProvider::new(None);
        let selected = collect(&prepared, execution, &[&first, &second]).unwrap();
        let Err(EmitError::Combination { reason }) = combine(&prepared, execution, selected) else {
            panic!("Register transport cannot cross provider boundaries");
        };
        assert!(reason.contains("Register input has no producer"));
        assert_eq!(*first.specified.borrow(), [0, 1, 2]);
        assert_eq!(*second.specified.borrow(), [1]);
        assert_eq!(first.rendered.get(), 0);
        assert_eq!(second.rendered.get(), 0);
    }
}

#[test]
fn cta_size_is_resolved_for_the_whole_body_without_mutating_specifications() {
    let cases: &[([&'static [usize]; 3], usize)] = &[
        ([&[64, 128], &[128, 256], &[128, 512]], 128),
        // The common size only becomes unique after the last operation.
        ([&[64, 128, 256], &[128, 256], &[256, 512]], 256),
        // Support-list order does not determine the body's choice.
        ([&[512, 256], &[512, 256, 64], &[512, 256]], 256),
        ([&[64, 256, 512], &[256, 512], &[256, 512]], 256),
    ];
    for execution in [ExecutionModel::CudaStreamed, ExecutionModel::CudaPersistent] {
        let provider = CuTeKernelProvider;
        for &(policies, threads) in cases {
            let plan = pipeline(0, Storage::Global);
            let prepared = prepare(&plan).unwrap();
            let mut selected = collect(&prepared, execution, &[&provider]).unwrap();
            for (kernel, supported) in selected.kernels.values_mut().zip(policies) {
                kernel.specification.requirements_mut().thread_policy =
                    ThreadPolicy::Flexible { supported };
            }
            let specifications: Vec<_> = selected
                .kernels
                .values()
                .map(|kernel| kernel.specification.clone())
                .collect();
            let combined = combine(&prepared, execution, selected).unwrap();
            assert_eq!(bodies(&combined.statements).len(), 1);
            let body = bodies(&combined.statements)[0];
            assert_eq!(body.requirements.block_threads, threads);
            for (kernel, original) in combined.kernels.values().zip(&specifications) {
                assert_eq!(&kernel.specification, original);
                assert!(matches!(
                    kernel.specification.interface(threads).outputs[0].register.as_ref(),
                    Some(RegisterLayout::Fixed { distribution, .. })
                        if distribution == &format!("cute.pointwise.{threads}x4")
                ));
            }
            let code = combined.render_body(body, &bindings(&plan)).unwrap().code;
            assert!(code.contains(&format!("cute::Int<{threads}>")));
            assert!(!code.contains("threadIdx.x <"));
        }

        // Register followers have no independent CTA size, whether their root
        // is a flexible pointwise partition or a fixed GEMM fragment.
        for (producer, threads) in [(0, 256), (1, 128)] {
            let plan = pipeline(producer, Storage::Register);
            let prepared = prepare(&plan).unwrap();
            let mut selected = collect(&prepared, execution, &[&provider]).unwrap();
            if producer == 0 {
                selected
                    .kernels
                    .values_mut()
                    .next()
                    .unwrap()
                    .specification
                    .requirements_mut()
                    .thread_policy = ThreadPolicy::Flexible { supported: &[256] };
            }
            for kernel in selected.kernels.values().skip(1) {
                assert_eq!(
                    kernel.specification.requirements().thread_policy,
                    ThreadPolicy::FollowInput
                );
            }
            let combined = combine(&prepared, execution, selected).unwrap();
            let body = bodies(&combined.statements)[0];
            assert_eq!(body.requirements.block_threads, threads);
            assert_eq!(body.roots.len(), 1);
            assert_eq!(body.roots[0].followers.len(), 2);
            let code = combined.render_body(body, &bindings(&plan)).unwrap().code;
            assert!(!code.contains("threadIdx.x <"));
            if producer == 0 {
                assert!(code.contains("cute::Int<256>"));
            }
        }
    }
}

fn subgroup_pipeline() -> PhysicalPlan {
    let mut builder = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
    let input = builder.add_value(DType::Fp32, [5, 131], Storage::External);
    let sum = builder.add_value(DType::Fp32, [5], Storage::Register);
    let squared = builder.add_value(DType::Fp32, [5], Storage::Global);
    let output = builder.add_value(DType::Fp32, [5], Storage::External);
    builder.bind_input("X", input);
    let reduce = store(
        &mut builder,
        sum,
        1,
        &[input],
        E::Add(Box::new([
            load(sum, 1),
            E::ReduceSum {
                value: Box::new(E::Load(TensorAccess::new(
                    input,
                    [
                        I::FullTile,
                        I::ClippedTile {
                            variable: "k".into(),
                            width: 65usize.into(),
                        },
                    ],
                ))),
                axis: 1,
            },
        ])),
    );
    let square = store(
        &mut builder,
        squared,
        1,
        &[sum],
        E::Sqr(Box::new(load(sum, 1))),
    );
    let last = store(
        &mut builder,
        output,
        1,
        &[squared],
        E::Sqrt(Box::new(load(squared, 1))),
    );
    builder
        .build(
            vec![loop_(
                LoopKind::Parallel,
                "p",
                0,
                1,
                1,
                vec![
                    loop_(LoopKind::Sequential, "k", 0, 195, 65, vec![reduce]),
                    square,
                    last,
                ],
            )],
            "Y",
            output,
        )
        .unwrap()
}

#[test]
fn subgroup_guards_cover_accumulation_and_followers_but_not_cta_barriers() {
    let plan = subgroup_pipeline();
    let provider = CuTeKernelProvider;
    let prepared = prepare(&plan).unwrap();
    for execution in [ExecutionModel::CudaStreamed, ExecutionModel::CudaPersistent] {
        let mut selected = collect(&prepared, execution, &[&provider]).unwrap();
        let requirements = selected
            .kernels
            .values_mut()
            .last()
            .unwrap()
            .specification
            .requirements_mut();
        requirements.thread_policy = ThreadPolicy::Flexible { supported: &[256] };
        let combined = combine(&prepared, execution, selected).unwrap();
        assert_eq!(bodies(&combined.statements).len(), 1);
        let body = bodies(&combined.statements)[0];
        assert_eq!(body.requirements.block_threads, 256);
        assert_eq!(body.roots.len(), 2);
        assert_eq!(body.roots[0].followers.len(), 1);
        let code = combined.render_body(body, &bindings(&plan)).unwrap().code;
        assert!(code.starts_with("if (threadIdx.x < 128) {"));
        assert_eq!(code.matches("__syncthreads();").count(), 1);
        let barrier = code.find("__syncthreads();").unwrap();
        assert!(code[..barrier].ends_with("}\n"));
        assert!(code[..barrier].contains("combined_op1_result"));
        assert!(code[barrier..].contains("combined_op2_result"));
        assert!(code.contains("cute::Int<256>"));
    }
}

#[test]
fn subgroup_participation_requires_whole_warps() {
    let plan = subgroup_pipeline();
    let provider = CuTeKernelProvider;
    let prepared = prepare(&plan).unwrap();
    let mut selected = collect(&prepared, plan_execution(&plan), &[&provider]).unwrap();
    selected
        .kernels
        .values_mut()
        .next()
        .unwrap()
        .specification
        .requirements_mut()
        .thread_policy = ThreadPolicy::Subgroup { threads: 33 };
    assert!(
        matches!(combine(&prepared, plan_execution(&plan), selected),
        Err(EmitError::Combination { reason }) if reason.contains("complete warps"))
    );
}

#[test]
#[ignore = "requires nvcc and vendored CuTe headers; no GPU required"]
fn mixed_thread_bodies_compile_with_nvcc() {
    use std::fmt::Write;
    let mut source = String::from(
        "#include <cute/tensor.hpp>\n#include <cutlass/bfloat16.h>\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <cmath>\n",
    );
    for (index, plan) in [subgroup_pipeline(), pipeline(0, Storage::Global)]
        .iter()
        .enumerate()
    {
        let prepared = prepare(plan).unwrap();
        let provider = CuTeKernelProvider;
        let mut selected = collect(&prepared, plan_execution(plan), &[&provider]).unwrap();
        let requirements = selected
            .kernels
            .values_mut()
            .last()
            .unwrap()
            .specification
            .requirements_mut();
        requirements.thread_policy = ThreadPolicy::Flexible { supported: &[256] };
        let combined = combine(&prepared, plan_execution(plan), selected).unwrap();
        let body = bodies(&combined.statements)[0];
        assert_eq!(bodies(&combined.statements).len(), 1);
        let parameters = plan
            .value_instances()
            .filter(|(_, v)| v.storage() != Storage::Register)
            .map(|(id, v)| {
                format!(
                    "{}* buffer{}",
                    match v.dtype() {
                        DType::Fp16 => "cutlass::half_t",
                        DType::Fp32 => "float",
                        DType::Bf16 => "cutlass::bfloat16_t",
                    },
                    id.index()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(source, "__global__ void test{index}({parameters}) {{").unwrap();
        source.push_str(&combined.render_body(body, &bindings(plan)).unwrap().code);
        source.push_str("}\n");
    }
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("mixed_threads.cu");
    std::fs::write(&input, &source).unwrap();
    let output =
        std::process::Command::new(std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
            .args(["-std=c++17", "-arch=sm_90a", "-c"])
            .arg(&input)
            .arg("-I")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/third_party/cutlass/include"
            ))
            .arg("-o")
            .arg(directory.path().join("mixed_threads.o"))
            .output()
            .expect("run nvcc");
    assert!(
        output.status.success(),
        "{}\n{source}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn resources_split_memory_bodies_but_never_spill_registers() {
    let provider = CuTeKernelProvider;
    for storage in [Storage::Global, Storage::Register] {
        for flexible in [false, true] {
            let plan = pipeline(0, storage);
            let prepared = prepare(&plan).unwrap();
            let mut selected = collect(&prepared, plan_execution(&plan), &[&provider]).unwrap();
            for (index, kernel) in selected.kernels.values_mut().enumerate() {
                kernel.specification.requirements_mut().thread_policy = if flexible {
                    ThreadPolicy::Flexible {
                        supported: if index == 1 { &[256] } else { &[128] },
                    }
                } else {
                    ThreadPolicy::FullCta {
                        threads: if index == 1 { 256 } else { 128 },
                    }
                };
            }
            let result = combine(&prepared, plan_execution(&plan), selected);
            if storage == Storage::Register {
                assert!(matches!(result, Err(EmitError::Combination { .. })));
            } else {
                let combined = result.unwrap();
                assert_eq!(
                    bodies(&combined.statements)
                        .iter()
                        .map(|body| body.requirements.block_threads)
                        .collect::<Vec<_>>(),
                    [128, 256, 128]
                );
            }
        }
    }
    let plan = pipeline(1, Storage::Register);
    let prepared = prepare(&plan).unwrap();
    let mut selected = collect(&prepared, plan_execution(&plan), &[&provider]).unwrap();
    let SpecifiedKernel::CuTeHopperGemm(s) =
        &mut selected.kernels.values_mut().next().unwrap().specification
    else {
        panic!()
    };
    s.requirements.shared_memory_bytes = usize::MAX;
    assert!(matches!(
        combine(&prepared, plan_execution(&plan), selected),
        Err(EmitError::Combination { .. })
    ));
}

#[test]
fn rejects_mismatched_access_and_collective_consumption_of_register_elements() {
    let provider = CuTeKernelProvider;
    for collective in [false, true] {
        let plan = pipeline(0, Storage::Register);
        let prepared = prepare(&plan).unwrap();
        let mut selected = collect(&prepared, plan_execution(&plan), &[&provider]).unwrap();
        let SpecifiedKernel::CuTePointwise(s) =
            &mut selected.kernels.values_mut().nth(1).unwrap().specification
        else {
            panic!()
        };
        if collective {
            s.requirements.thread_policy = ThreadPolicy::FullCta { threads: 128 };
        } else {
            s.inputs[0].access.axes[0] = crate::emit::provider::access::Axis::Tile {
                variable: "lv0".into(),
                width: 1,
                clipped: false,
            };
        }
        assert!(matches!(
            combine(&prepared, plan_execution(&plan), selected),
            Err(EmitError::Combination { .. })
        ));
    }
}

struct FailingProvider(bool);
impl KernelProvider for FailingProvider {
    fn name(&self) -> &str {
        "failure"
    }
    fn specify(&self, _: &KernelContext<'_, '_>) -> Result<SpecifiedKernel, ProviderError> {
        if self.0 {
            Err(ProviderError::Failed("broken implementation".into()))
        } else {
            Err(ProviderError::Unsupported("unsupported dtype".into()))
        }
    }
    fn render(&self, _: &SpecifiedKernel, _: &KernelBindings) -> Result<Kernel, ProviderError> {
        unreachable!()
    }
}
#[test]
fn collection_preserves_unsupported_diagnostics_and_propagates_internal_failures() {
    let plan = pipeline(0, Storage::Global);
    let prepared = prepare(&plan).unwrap();
    let execution = plan_execution(&plan);
    let unsupported = FailingProvider(false);
    let failed = FailingProvider(true);
    let provider = CuTeKernelProvider;
    assert!(collect(&prepared, execution, &[&unsupported, &provider]).is_ok());
    assert!(
        matches!(collect(&prepared,execution,&[&unsupported]),Err(EmitError::NoProvider{reasons,..}) if reasons[0].contains("unsupported dtype"))
    );
    assert!(
        matches!(collect(&prepared,execution,&[&failed,&provider]),Err(EmitError::Provider{message,..}) if message=="broken implementation")
    );
}

#[test]
fn register_values_do_not_escape_a_loop_or_an_unrelated_output_scope() {
    for loop_boundary in [false, true] {
        let mut builder = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
        let input = builder.add_value(DType::Fp32, [16], Storage::External);
        let temporary = builder.add_value(DType::Fp32, [16], Storage::Register);
        let other = builder.add_value(DType::Fp32, [16], Storage::Global);
        let output = builder.add_value(DType::Fp32, [16], Storage::External);
        builder.bind_input("X", input);
        let producer = store(
            &mut builder,
            temporary,
            1,
            &[input],
            E::Sqr(Box::new(load(input, 1))),
        );
        let consumer = store(&mut builder, output, 1, &[temporary], load(temporary, 1));
        let body = if loop_boundary {
            vec![
                loop_(LoopKind::Sequential, "j", 0, 2, 1, vec![producer]),
                consumer,
            ]
        } else {
            let unrelated = store(&mut builder, other, 1, &[input], load(input, 1));
            vec![producer, unrelated, consumer]
        };
        let plan = builder
            .build(
                vec![loop_(LoopKind::Parallel, "i", 0, 1, 1, body)],
                "Y",
                output,
            )
            .unwrap();
        let prepared = prepare(&plan).unwrap();
        let provider = CuTeKernelProvider;
        let selected = collect(&prepared, plan_execution(&plan), &[&provider]).unwrap();
        assert!(matches!(
            combine(&prepared, plan_execution(&plan), selected),
            Err(EmitError::Combination { .. })
        ));
    }
}

#[test]
fn rewriting_a_register_value_connects_its_latest_definition() {
    let mut builder = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
    let input = builder.add_value(DType::Fp32, [16], Storage::External);
    let temporary = builder.add_value(DType::Bf16, [16], Storage::Register);
    let output = builder.add_value(DType::Fp32, [16], Storage::External);
    builder.bind_input("X", input);
    let first = store(
        &mut builder,
        temporary,
        1,
        &[input],
        E::Sqr(Box::new(load(input, 1))),
    );
    let update = store(
        &mut builder,
        temporary,
        1,
        &[temporary],
        E::Sigmoid(Box::new(load(temporary, 1))),
    );
    let last = store(&mut builder, output, 1, &[temporary], load(temporary, 1));
    let plan = builder
        .build(
            vec![loop_(
                LoopKind::Parallel,
                "i",
                0,
                1,
                1,
                vec![first, update, last],
            )],
            "Y",
            output,
        )
        .unwrap();
    let prepared = prepare(&plan).unwrap();
    let provider = CuTeKernelProvider;
    let selected = collect(&prepared, plan_execution(&plan), &[&provider]).unwrap();
    let combined = combine(&prepared, plan_execution(&plan), selected).unwrap();
    let body = bodies(&combined.statements)[0];
    let code = combined.render_body(body, &bindings(&plan)).unwrap().code;
    assert!(code.contains("combined_op2_load0 = static_cast<float>(combined_op1_result)"));
    assert_eq!(code.matches("static_cast<cutlass::bfloat16_t>").count(), 2);
}
