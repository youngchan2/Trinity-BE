use super::*;
use crate::emit::execution::ExecutionModel;
use crate::emit::prepare::prepare;
use crate::emit::provider::{CuTeKernelProvider, KernelProvider};
use crate::{
    AccessIndex as I, Expression as E, Loop, LoopDomain, LoopKind, PhysicalPlan,
    PhysicalPlanBuilder, Statement, TensorAccess,
};
use std::fmt::Write;
use std::process::Command;

struct Case {
    m: usize,
    rows: usize,
    origin: [i64; 2],
    width: usize,
    start: i64,
    step: i64,
    iterations: i64,
    output_dtype: DType,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            m: 17,
            rows: 17,
            origin: [0, 0],
            width: 128,
            start: 0,
            step: 128,
            iterations: 3,
            output_dtype: DType::Fp32,
        }
    }
}

impl Case {
    fn k(&self) -> usize {
        // End the allocation at the last logical access to catch an extra prefetch.
        (self.start + (self.iterations - 1) * self.step) as usize + self.width
    }

    fn n(&self) -> usize {
        self.origin[1] as usize + 128
    }

    fn plan(&self) -> PhysicalPlan {
        let mut builder =
            PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);

        let a = builder.add_named_value("A", DType::Bf16, [self.m, self.k()], Storage::External);
        let b = builder.add_named_value("B", DType::Bf16, [self.k(), self.n()], Storage::External);
        let c = builder.add_named_value(
            "C",
            self.output_dtype,
            [self.m, self.n()],
            Storage::External,
        );

        builder.bind_input("A", a);
        builder.bind_input("B", b);
        let mi = I::ClippedTile {
            variable: "m".into(),
            width: self.rows.into(),
        };

        let ni = I::Tile {
            variable: "n".into(),
            width: 128usize.into(),
        };

        let ki = I::Tile {
            variable: "k".into(),
            width: self.width.into(),
        };

        let output = TensorAccess::new(c, [mi.clone(), ni.clone()]);
        let op = builder.add_operation(
            [a, b],
            [c],
            E::Store {
                destination: output.clone(),
                value: Box::new(E::Add(Box::new([
                    E::Load(output),
                    E::Matmul(Box::new([
                        E::Load(TensorAccess::new(a, [mi, ki.clone()])),
                        E::Load(TensorAccess::new(b, [ki, ni])),
                    ])),
                ]))),
            },
        );

        let mut statement = Statement::Operation(op);
        for (kind, name, start, step, count) in [
            (
                LoopKind::Sequential,
                "k",
                self.start,
                self.step,
                self.iterations,
            ),
            (LoopKind::Parallel, "n", self.origin[1], 128, 1),
            (LoopKind::Parallel, "m", self.origin[0], 128, 1),
        ] {
            statement = Statement::Loop(Loop {
                kind,
                domain: LoopDomain {
                    variable: name.into(),
                    start: IndexExpr::Constant(start),
                    stop: IndexExpr::Constant(start + step * count),
                    step: IndexExpr::Constant(step),
                },
                body: vec![statement],
            });
        }
        builder.build(vec![statement], "C", c).unwrap()
    }
}

fn specify(plan: &PhysicalPlan, execution: ExecutionModel) -> HopperGemmSpecification {
    let prepared = prepare(plan).unwrap();
    let mut loops = Vec::new();
    let mut statement = &plan.statements()[0];
    while let Statement::Loop(l) = statement {
        loops.push(l);
        statement = &l.body[0];
    }
    let Statement::Operation(operation) = statement else {
        panic!("GEMM")
    };
    let SpecifiedKernel::CuTeHopperGemm(spec) = CuTeKernelProvider
        .specify(&KernelContext {
            prepared: &prepared,
            execution,
            operation: *operation,
            loops: &loops,
        })
        .unwrap()
    else {
        panic!("Hopper GEMM")
    };
    spec
}

fn render(spec: &HopperGemmSpecification) -> Kernel {
    CuTeKernelProvider
        .render(
            &SpecifiedKernel::CuTeHopperGemm(spec.clone()),
            &KernelBindings {
                block_threads: 128,
                values: BTreeMap::from([
                    (spec.lhs.value, "a".into()),
                    (spec.rhs.value, "b".into()),
                    (spec.output.value, "c".into()),
                ]),
                registers: BTreeMap::new(),
                indices: spec
                    .output
                    .axes
                    .iter()
                    .chain([&spec.lhs.axes[1]])
                    .zip(["tile_m", "tile_n", "logical_k"])
                    .map(|(axis, binding)| {
                        let Axis::Tile { variable, .. } = axis else {
                            panic!("tiled test access")
                        };
                        (variable.clone(), binding.into())
                    })
                    .collect(),
                shared_memory: Some("scratch".into()),
                prefix: "hopper".into(),
            },
        )
        .unwrap()
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for width in [64, 128, 192, 256] {
        for output_dtype in [DType::Bf16, DType::Fp32] {
            cases.push(Case {
                width,
                step: width as i64,
                output_dtype,
                ..Default::default()
            });
        }
    }
    cases.extend([
        Case {
            m: 1,
            rows: 1,
            width: 64,
            iterations: 1,
            ..Default::default()
        },
        Case {
            m: 128,
            rows: 128,
            width: 256,
            iterations: 1,
            ..Default::default()
        },
        // Overlapping logical accesses and a clipped M tile with nonzero origins.
        Case {
            m: 133,
            rows: 128,
            origin: [128, 8],
            width: 192,
            start: 8,
            step: 64,
            ..Default::default()
        },
        // A gap between logical accesses must not be consumed by the pipeline.
        Case {
            width: 64,
            start: 8,
            step: 128,
            ..Default::default()
        },
    ]);
    cases
}

#[test]
fn double_buffering_preserves_logical_accesses_and_execution_contracts() {
    for case in cases() {
        let plan = case.plan();
        let spec = specify(&plan, ExecutionModel::CudaStreamed);
        assert_eq!(spec, specify(&plan, ExecutionModel::CudaPersistent));
        assert_eq!(spec.lhs.width(1), case.width);
        assert_eq!(spec.requirements.shared_memory_bytes, 65536);
        assert_eq!(spec.requirements.shared_memory_alignment, 128);
        assert_eq!(
            spec.requirements.thread_policy,
            crate::emit::provider::ThreadPolicy::FullCta { threads: 128 }
        );
        let Kernel::Native {
            prologue,
            mainloop,
            epilogue,
            ..
        } = render(&spec)
        else {
            panic!("native")
        };
        assert!(!prologue.source().contains("logical_k"));
        let mainloop = mainloop.unwrap().source();
        assert!(!mainloop.contains("cute::clear"));
        assert!(mainloop.contains(&format!("offset + 64 < int64_t({})", case.width)));
        assert!(epilogue.source().contains("hopper_accumulator(element)"));
    }
}

fn cuda_source() -> String {
    let mut source = String::from(
        "#include <cute/tensor.hpp>\n#include <cutlass/bfloat16.h>\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <cstdio>\n#include <vector>\n",
    );
    for (i, c) in cases().iter().enumerate() {
        let spec = specify(&c.plan(), ExecutionModel::CudaStreamed);
        let Kernel::Native {
            prologue,
            mainloop,
            epilogue,
            ..
        } = render(&spec)
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
    for (i, c) in cases().iter().enumerate() {
        let spec = specify(&c.plan(), ExecutionModel::CudaStreamed);
        writeln!(
            source,
            "if (check(gemm{i}, {i}, {}, {}, {}, {}, 128, {}, {}, {}, {}, {}, {}, {})) return 1;",
            c.m,
            c.n(),
            c.k(),
            c.rows,
            c.origin[0],
            c.origin[1],
            c.start,
            c.step,
            c.iterations,
            c.width,
            spec.requirements.shared_memory_bytes,
        )
        .unwrap();
    }
    source.push_str("return 0;\n}\n");
    source
}

fn compile_cuda(directory: &std::path::Path) -> std::path::PathBuf {
    let input = directory.join("hopper.cu");
    let output = directory.join("hopper");
    std::fs::write(&input, cuda_source()).unwrap();
    let result = Command::new(std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .args(["-std=c++17", "-O3", "-arch=sm_90a"])
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
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    output
}

#[test]
#[ignore = "requires nvcc and vendored CuTe headers; no GPU required"]
fn hopper_kernels_compile_with_nvcc() {
    let directory = tempfile::tempdir().unwrap();
    compile_cuda(directory.path());
}

#[test]
#[ignore = "requires Hopper GPU; optional HOPPER_SANITIZER=memcheck/racecheck/synccheck"]
fn hopper_kernels_match_cpu_reference() {
    let directory = tempfile::tempdir().unwrap();
    let executable = compile_cuda(directory.path());
    let mut command = if let Ok(tool) = std::env::var("HOPPER_SANITIZER") {
        let mut command = Command::new("compute-sanitizer");
        command
            .args(["--error-exitcode", "1", "--tool", &tool])
            .arg(executable);
        command
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
