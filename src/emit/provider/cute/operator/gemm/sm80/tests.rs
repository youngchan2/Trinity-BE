use super::*;
use crate::emit::execution::ExecutionModel;
use crate::emit::prepare::prepare;
use crate::emit::provider::{CuTeKernelProvider, KernelProvider};
use crate::{
    AccessIndex as I, AsStr, Expression as E, Loop, LoopDomain, LoopKind, PhysicalPlan,
    PhysicalPlanBuilder, Statement, TensorAccess,
};
use std::fmt::Write;
use std::process::Command;

#[derive(Clone)]
struct Case {
    target: CudaTargetCapability,
    m: usize,
    n: usize,
    k: usize,
    rows: usize,
    columns: usize,
    width: usize,
    origin: [i64; 2],
    start: i64,
    step: i64,
    iterations: i64,
    clipped: [bool; 3],
    input_dtype: DType,
    output_dtype: DType,
    input_storage: Storage,
    output_storage: Storage,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            target: CudaTargetCapability::Sm89,
            m: 17,
            n: 128,
            k: 256,
            rows: 17,
            columns: 128,
            width: 64,
            origin: [0, 0],
            start: 0,
            step: 64,
            iterations: 2,
            clipped: [false; 3],
            input_dtype: DType::Bf16,
            output_dtype: DType::Fp32,
            input_storage: Storage::External,
            output_storage: Storage::External,
        }
    }
}

fn tile(name: &str, width: usize, clipped: bool) -> I {
    if clipped {
        I::ClippedTile {
            variable: name.into(),
            width,
        }
    } else {
        I::Tile {
            variable: name.into(),
            width,
        }
    }
}

fn loop_(
    kind: LoopKind,
    name: &str,
    start: i64,
    step: i64,
    count: i64,
    body: Vec<Statement>,
) -> Statement {
    Statement::Loop(Loop {
        kind,
        domain: LoopDomain {
            variable: name.into(),
            start: IndexExpr::Constant(start),
            stop: IndexExpr::Constant(start + step * count),
            step: IndexExpr::Constant(step),
        },
        body,
    })
}

fn plan(c: &Case) -> PhysicalPlan {
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(c.target), 1);
    let external_a = b.add_named_value("A", c.input_dtype, [c.m, c.k], Storage::External);
    let rhs = b.add_named_value("B", c.input_dtype, [c.k, c.n], Storage::External);
    let external_c = b.add_named_value("C", c.output_dtype, [c.m, c.n], Storage::External);
    b.bind_input("A", external_a);
    b.bind_input("B", rhs);
    let mut body = Vec::new();
    let lhs = if c.input_storage == Storage::External {
        external_a
    } else {
        let a = b.add_named_value("local_a", c.input_dtype, [c.m, c.k], c.input_storage);
        let op = b.add_operation(
            [external_a],
            [a],
            E::Store {
                destination: TensorAccess::new(a, [I::FullTile, I::FullTile]),
                value: Box::new(E::Load(TensorAccess::new(
                    external_a,
                    [I::FullTile, I::FullTile],
                ))),
            },
        );
        body.push(Statement::Operation(op));
        a
    };
    let output = if c.output_storage == Storage::External {
        external_c
    } else {
        b.add_named_value("local_c", c.output_dtype, [c.m, c.n], c.output_storage)
    };
    let mi = tile("m", c.rows, c.clipped[0]);
    let ni = tile("n", c.columns, c.clipped[1]);
    let ki = tile("k", c.width, c.clipped[2]);
    let destination = TensorAccess::new(output, [mi.clone(), ni.clone()]);
    let op = b.add_operation(
        [lhs, rhs],
        [output],
        E::Store {
            destination: destination.clone(),
            value: Box::new(E::Add(Box::new([
                E::Load(destination.clone()),
                E::Matmul(Box::new([
                    E::Load(TensorAccess::new(lhs, [mi.clone(), ki.clone()])),
                    E::Load(TensorAccess::new(rhs, [ki, ni.clone()])),
                ])),
            ]))),
        },
    );
    body.push(loop_(
        LoopKind::Sequential,
        "k",
        c.start,
        c.step,
        c.iterations,
        vec![Statement::Operation(op)],
    ));
    if output != external_c {
        let op = b.add_operation(
            [output],
            [external_c],
            E::Store {
                destination: TensorAccess::new(external_c, [mi, ni]),
                value: Box::new(E::Load(destination)),
            },
        );
        body.push(Statement::Operation(op));
    }
    b.build(
        vec![loop_(
            LoopKind::Parallel,
            "m",
            c.origin[0],
            128,
            1,
            vec![loop_(LoopKind::Parallel, "n", c.origin[1], 128, 1, body)],
        )],
        "C",
        external_c,
    )
    .unwrap()
}

fn loops(plan: &PhysicalPlan) -> Vec<&Loop> {
    let mut result = Vec::new();
    let mut statements = plan.statements();
    while let Some(l) = statements.iter().find_map(|s| {
        if let Statement::Loop(l) = s {
            Some(l)
        } else {
            None
        }
    }) {
        result.push(l);
        statements = &l.body;
    }
    result
}

fn candidates(plan: &PhysicalPlan, execution: ExecutionModel) -> Vec<SpecifiedKernel> {
    let prepared = prepare(plan).unwrap();
    let loops = loops(plan);
    let Statement::Operation(operation) = loops.last().unwrap().body[0] else {
        panic!("GEMM")
    };
    match CuTeKernelProvider.specify(&KernelContext {
        prepared: &prepared,
        execution,
        operation,
        loops: &loops,
    }) {
        Ok(specification) => vec![specification],
        Err(ProviderError::Unsupported(_)) => vec![],
        Err(error) => panic!("{error}"),
    }
}

fn specification(plan: &PhysicalPlan) -> Sm80GemmSpecification {
    let specs = candidates(plan, ExecutionModel::CudaStreamed);
    let [SpecifiedKernel::CuTeSm80Gemm(spec)] = specs.as_slice() else {
        panic!("expected SM80: {specs:?}")
    };
    spec.clone()
}

fn bindings(plan: &PhysicalPlan, spec: &Sm80GemmSpecification) -> KernelBindings {
    KernelBindings {
        block_threads: 128,
        values: BTreeMap::from([
            (spec.lhs.value, "a".into()),
            (spec.rhs.value, "b".into()),
            (spec.output.value, "c".into()),
        ]),
        registers: BTreeMap::new(),
        indices: loops(plan)
            .iter()
            .zip(["tile_m", "tile_n", "logical_k"])
            .map(|(l, s)| (l.domain.variable.clone(), s.into()))
            .collect(),
        shared_memory: Some("scratch".into()),
        prefix: "sm80".into(),
    }
}

#[test]
fn targets_dispatch_without_changing_hopper() {
    for (target, arch) in [
        (CudaTargetCapability::Sm89, "sm_89"),
        (CudaTargetCapability::Sm120, "sm_120"),
    ] {
        assert_eq!(target.as_str(), arch);
        assert_eq!(target.max_shared_memory_per_cta(), 99 * 1024);
        assert!(crate::gemm_implementations(TargetCapability::Cuda(target)).is_empty());
        let p = plan(&Case {
            target,
            ..Default::default()
        });
        let streamed = candidates(&p, ExecutionModel::CudaStreamed);
        assert!(matches!(
            streamed.as_slice(),
            [SpecifiedKernel::CuTeSm80Gemm(_)]
        ));
        assert_eq!(streamed, candidates(&p, ExecutionModel::CudaPersistent));
    }
    let p = plan(&Case {
        target: CudaTargetCapability::Hopper,
        ..Default::default()
    });
    assert!(matches!(
        candidates(&p, ExecutionModel::CudaStreamed).as_slice(),
        [SpecifiedKernel::CuTeHopperGemm(_)]
    ));
}

#[test]
fn supported_tiles_preserve_accumulation_and_resource_contracts() {
    for target in [CudaTargetCapability::Sm89, CudaTargetCapability::Sm120] {
        for width in [32, 64, 96, 128] {
            for output_dtype in [DType::Bf16, DType::Fp32] {
                let p = plan(&Case {
                    target,
                    width,
                    start: 8,
                    step: 64,
                    output_dtype,
                    ..Default::default()
                });
                let spec = specification(&p);
                assert_eq!(spec.lhs.width(1), width);
                assert_eq!(spec.requirements.shared_memory_bytes, 32768);
                assert_eq!(spec.requirements.shared_memory_alignment, 128);
                assert_eq!(
                    spec.requirements.thread_policy,
                    crate::emit::provider::ThreadPolicy::FullCta { threads: 128 }
                );
                assert_eq!(spec.requirements.alignments[0], (spec.lhs.value, 16));
                assert_eq!(
                    spec.interface().iteration.as_deref(),
                    Some(spec.iteration.as_str())
                );
                let Kernel::Native {
                    prologue,
                    mainloop,
                    epilogue,
                    ..
                } = CuTeKernelProvider
                    .render(
                        &SpecifiedKernel::CuTeSm80Gemm(spec.clone()),
                        &bindings(&p, &spec),
                    )
                    .unwrap()
                else {
                    panic!("native")
                };
                assert!(!prologue.source().contains("logical_k"));
                let mainloop = mainloop.unwrap().source();
                assert!(!mainloop.contains("cute::clear"));
                assert!(mainloop.contains(&format!("offset + 32 < int64_t({width})")));
                assert!(!mainloop.contains("warpgroup"));
                assert!(!mainloop.contains("fence.proxy"));
                assert!(epilogue.source().contains("sm80_accumulator(element)"));
            }
        }
    }
    for c in [
        Case {
            rows: 1,
            m: 1,
            ..Default::default()
        },
        Case {
            rows: 128,
            m: 128,
            ..Default::default()
        },
        Case {
            rows: 128,
            clipped: [true, false, false],
            origin: [8, 8],
            n: 144,
            ..Default::default()
        },
        Case {
            input_storage: Storage::Global,
            output_storage: Storage::Global,
            ..Default::default()
        },
    ] {
        specification(&plan(&c));
    }
}

#[test]
fn unsupported_accesses_do_not_become_candidates() {
    for c in [
        Case {
            rows: 129,
            m: 129,
            ..Default::default()
        },
        Case {
            columns: 64,
            ..Default::default()
        },
        Case {
            width: 48,
            ..Default::default()
        },
        Case {
            start: 1,
            ..Default::default()
        },
        Case {
            step: 33,
            ..Default::default()
        },
        Case {
            origin: [0, 1],
            n: 144,
            ..Default::default()
        },
        Case {
            k: 255,
            ..Default::default()
        },
        Case {
            n: 129,
            ..Default::default()
        },
        Case {
            clipped: [false, true, false],
            ..Default::default()
        },
        Case {
            clipped: [false, false, true],
            ..Default::default()
        },
        Case {
            input_dtype: DType::Fp32,
            ..Default::default()
        },
        Case {
            input_storage: Storage::Shared,
            ..Default::default()
        },
        Case {
            input_storage: Storage::Register,
            ..Default::default()
        },
        Case {
            output_storage: Storage::Shared,
            ..Default::default()
        },
    ] {
        assert!(candidates(&plan(&c), ExecutionModel::CudaStreamed).is_empty());
    }
}

#[test]
fn register_output_converts_before_continuation_and_bindings_are_required() {
    let p = plan(&Case {
        output_dtype: DType::Bf16,
        output_storage: Storage::Register,
        ..Default::default()
    });
    let spec = specification(&p);
    let mut binding = bindings(&p, &spec);
    binding.values.remove(&spec.output.value);
    let candidate = SpecifiedKernel::CuTeSm80Gemm(spec.clone());
    let Kernel::Native { epilogue, .. } = CuTeKernelProvider.render(&candidate, &binding).unwrap()
    else {
        panic!("native")
    };
    let mut calls = 0;
    let source = epilogue
        .connect(&mut |port, element| {
            calls += 1;
            assert_eq!(port, 0);
            assert_eq!(element.value, "sm80_result");
            assert_eq!(element.coordinates, ["sm80_m + row", "sm80_n + column"]);
            Ok("consume(sm80_result);\n".into())
        })
        .unwrap();
    assert_eq!(calls, 1);
    assert!(
        source.find("static_cast<cutlass::bfloat16_t>").unwrap() < source.find("consume(").unwrap()
    );
    assert_ne!(
        spec.interface().outputs[0].register,
        Some(RegisterLayout::Fixed {
            representation: "cuda.scalar",
            distribution: "cute.wgmma.128x128".into()
        })
    );
    let missing = [
        {
            let mut b = binding.clone();
            b.shared_memory = None;
            b
        },
        {
            let mut b = binding.clone();
            b.values.remove(&spec.lhs.value);
            b
        },
        {
            let mut b = binding.clone();
            b.indices.remove(&spec.iteration);
            b
        },
        {
            let mut b = binding;
            b.prefix = "bad-prefix".into();
            b
        },
    ];
    for b in missing {
        assert!(matches!(
            CuTeKernelProvider.render(&candidate, &b),
            Err(ProviderError::Failed(_))
        ));
    }
}

fn numerical_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for width in [32, 64, 96, 128] {
        for output_dtype in [DType::Bf16, DType::Fp32] {
            cases.push(Case {
                width,
                k: width * 3,
                step: width as i64,
                iterations: 3,
                output_dtype,
                ..Default::default()
            });
        }
    }
    cases.extend([
        Case {
            m: 1,
            rows: 1,
            width: 32,
            k: 32,
            iterations: 1,
            ..Default::default()
        },
        Case {
            m: 128,
            rows: 128,
            width: 128,
            k: 128,
            iterations: 1,
            ..Default::default()
        },
        Case {
            m: 133,
            rows: 128,
            origin: [128, 8],
            n: 144,
            clipped: [true, false, false],
            start: 8,
            step: 40,
            width: 96,
            k: 144,
            ..Default::default()
        },
        Case {
            start: 8,
            step: 96,
            width: 32,
            k: 136,
            ..Default::default()
        },
    ]);
    cases
}

fn cuda_source() -> String {
    let mut source = String::from(
        "#include <cute/tensor.hpp>\n#include <cutlass/bfloat16.h>\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <cstdio>\n#include <vector>\n#include <cmath>\n",
    );
    for (i, c) in numerical_cases().iter().enumerate() {
        let p = plan(c);
        let spec = specification(&p);
        let Kernel::Native {
            prologue,
            mainloop,
            epilogue,
            ..
        } = CuTeKernelProvider
            .render(
                &SpecifiedKernel::CuTeSm80Gemm(spec.clone()),
                &bindings(&p, &spec),
            )
            .unwrap()
        else {
            panic!("native")
        };
        let output = if c.output_dtype == DType::Bf16 {
            "cutlass::bfloat16_t"
        } else {
            "float"
        };
        writeln!(source, "__global__ void gemm{i}(const cutlass::bfloat16_t* a, const cutlass::bfloat16_t* b, {output}* c) {{\nextern __shared__ __align__(128) unsigned char scratch[];\nconst int64_t tile_m = {}, tile_n = {};\n{}\nfor (int64_t logical_k = {}; logical_k < {}; logical_k += {}) {{\n{}\n}}\n{}\n}}", c.origin[0], c.origin[1], prologue.source(), c.start, c.start + c.step * c.iterations, c.step, mainloop.unwrap().source(), epilogue.source()).unwrap();
    }
    source.push_str(include_str!("../tests/reference.cu"));
    source.push_str("int main() {\n");
    for (i, c) in numerical_cases().iter().enumerate() {
        writeln!(
            source,
            "if (check(gemm{i}, {i}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})) return 1;",
            c.m,
            c.n,
            c.k,
            c.rows,
            c.columns,
            c.origin[0],
            c.origin[1],
            c.start,
            c.step,
            c.iterations,
            c.width
        )
        .unwrap();
    }
    source.push_str("return 0;\n}\n");
    source
}

fn compile_cuda(directory: &std::path::Path, arch: &str) -> std::path::PathBuf {
    let input = directory.join("sm80.cu");
    let output = directory.join(arch);
    std::fs::write(&input, cuda_source()).unwrap();
    let result = Command::new(std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .args(["-std=c++17", "-O3", &format!("-arch={arch}")])
        .arg(&input)
        .arg("-I")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/third_party/cutlass/include"
        ))
        .arg("-o")
        .arg(&output)
        .output()
        .expect("run nvcc");
    assert!(
        result.status.success(),
        "{arch}: {}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    output
}

#[test]
#[ignore = "requires nvcc and vendored CuTe headers; no GPU required"]
fn sm80_kernels_compile_with_nvcc() {
    let directory = tempfile::tempdir().unwrap();
    for arch in ["sm_80", "sm_89", "sm_120"] {
        compile_cuda(directory.path(), arch);
    }
}

#[test]
#[ignore = "requires CUDA GPU; SM80_TEST_ARCH selects sm_89 or sm_120; optional SM80_SANITIZER=memcheck/racecheck/synccheck"]
fn sm80_kernels_match_cpu_reference() {
    let directory = tempfile::tempdir().unwrap();
    let arch = std::env::var("SM80_TEST_ARCH").unwrap_or_else(|_| "sm_89".into());
    assert!(["sm_89", "sm_120"].contains(&arch.as_str()));
    let executable = compile_cuda(directory.path(), &arch);
    let mut command = if let Ok(tool) = std::env::var("SM80_SANITIZER") {
        let mut c = Command::new("compute-sanitizer");
        c.args(["--error-exitcode", "1", "--tool", &tool])
            .arg(executable);
        c
    } else {
        Command::new(executable)
    };
    let result = command.output().expect("run CUDA reference test");
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
