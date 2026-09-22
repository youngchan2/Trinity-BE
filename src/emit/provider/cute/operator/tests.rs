use super::*;
use crate::emit::native::NativeKernelProvider;
use crate::emit::native::{Kernel, KernelBindings, SpecifiedKernel};
use crate::emit::provider::CuTeKernelProvider;
use crate::emit::{execution::plan_execution, native::collect::collect, prepare::prepare};
use crate::{
    AccessIndex as I, Constant, DType, Expression as E, IndexExpr, LoopDomain, LoweringConfig,
    PhysicalPlan, PhysicalPlanBuilder, Statement, Storage, TensorAccess,
};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::process::Command;

fn builder() -> PhysicalPlanBuilder {
    PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1)
}

fn tile(name: &str, width: usize) -> I {
    I::Tile {
        variable: name.into(),
        width: width.into(),
    }
}

fn serial(operation: crate::OperationId, start: i64, stop: i64, step: i64) -> Statement {
    Statement::Loop(Loop {
        kind: LoopKind::Sequential,
        domain: LoopDomain {
            variable: "k".into(),
            start: IndexExpr::Constant(start),
            stop: IndexExpr::Constant(stop),
            step: IndexExpr::Constant(step),
        },
        body: vec![Statement::Operation(operation)],
    })
}

fn gemm(m: usize, n: usize, width: usize, start: i64, step: i64, dtype: DType) -> PhysicalPlan {
    let mut builder = builder();
    let a = builder.add_named_value("A", DType::Bf16, [m, 256], Storage::External);
    let b = builder.add_named_value("B", DType::Bf16, [256, n], Storage::External);
    let c = builder.add_named_value("C", dtype, [m, n], Storage::External);
    builder.bind_input("A", a);
    builder.bind_input("B", b);
    let destination = TensorAccess::new(c, [I::FullTile, I::FullTile]);
    let rhs = E::Matmul(Box::new([
        E::Load(TensorAccess::new(a, [I::FullTile, tile("k", width)])),
        E::Load(TensorAccess::new(b, [tile("k", width), I::FullTile])),
    ]));
    let op = builder.add_operation(
        [a, b],
        [c],
        E::Store {
            destination: destination.clone(),
            value: Box::new(E::Add(Box::new([E::Load(destination), rhs]))),
        },
    );
    builder
        .build(vec![serial(op, start, start + 128, step)], "C", c)
        .unwrap()
}

fn reduction(
    rows: usize,
    columns: usize,
    width: usize,
    dtype: DType,
    square: bool,
) -> PhysicalPlan {
    let mut builder = builder();
    let input = builder.add_named_value("X", dtype, [rows, columns], Storage::External);
    let output = builder.add_named_value("Y", DType::Fp32, [rows], Storage::External);
    builder.bind_input("X", input);
    let destination = TensorAccess::new(output, [I::FullTile]);
    let load = E::Load(TensorAccess::new(
        input,
        [
            I::FullTile,
            I::ClippedTile {
                variable: "k".into(),
                width: width.into(),
            },
        ],
    ));
    let rhs = E::ReduceSum {
        value: Box::new(if square { E::Sqr(Box::new(load)) } else { load }),
        axis: 1,
    };
    let op = builder.add_operation(
        [input],
        [output],
        E::Store {
            destination: destination.clone(),
            value: Box::new(E::Add(Box::new([E::Load(destination), rhs]))),
        },
    );
    let stop = columns.div_ceil(width) * width;
    builder
        .build(vec![serial(op, 0, stop as i64, width as i64)], "Y", output)
        .unwrap()
}

fn normalization(dtype: DType) -> PhysicalPlan {
    let mut builder = builder();
    let x = builder.add_named_value("X", dtype, [3, 131], Storage::External);
    let sum = builder.add_named_value("sum", DType::Fp32, [3], Storage::External);
    let y = builder.add_named_value("Y", dtype, [3, 131], Storage::External);
    builder.bind_input("X", x);
    builder.bind_input("sum", sum);
    let rhs = E::Div(Box::new([
        E::Load(TensorAccess::new(x, [I::FullTile, I::FullTile])),
        E::Broadcast {
            axis: 1,
            value: Box::new(E::Sqrt(Box::new(E::Div(Box::new([
                E::Load(TensorAccess::new(sum, [I::FullTile])),
                E::Constant(Constant::Integer(131)),
            ]))))),
        },
    ]));
    let op = builder.add_operation(
        [x, sum],
        [y],
        E::Store {
            destination: TensorAccess::new(y, [I::FullTile, I::FullTile]),
            value: Box::new(rhs),
        },
    );
    builder
        .build(vec![Statement::Operation(op)], "Y", y)
        .unwrap()
}

fn candidates(plan: &PhysicalPlan) -> Vec<SpecifiedKernel> {
    let prepared = prepare(plan).unwrap();
    let provider = CuTeKernelProvider;
    match collect(&prepared, plan_execution(plan), &[&provider]) {
        Ok(selected) => selected
            .kernels
            .into_values()
            .map(|kernel| kernel.specification)
            .collect(),
        Err(crate::emit::EmitError::NoProvider { .. }) => vec![],
        Err(error) => panic!("{error}"),
    }
}

fn bindings(plan: &PhysicalPlan) -> KernelBindings {
    let mut indices = BTreeMap::new();
    fn loops(statements: &[Statement], indices: &mut BTreeMap<String, String>) {
        for statement in statements {
            if let Statement::Loop(l) = statement {
                indices.insert(l.domain.variable.clone(), l.domain.variable.clone());
                loops(&l.body, indices);
            }
        }
    }
    loops(plan.statements(), &mut indices);
    KernelBindings {
        block_threads: 128,
        registers: BTreeMap::new(),
        values: plan
            .value_instances()
            .map(|(id, _)| (id, format!("buffer{}", id.index())))
            .collect(),
        indices,
        prefix: "body".into(),
        shared_memory: Some("scratch".into()),
    }
}

#[test]
fn gemm_support_and_requirements_preserve_logical_k_accesses() {
    for m in [16, 128] {
        for dtype in [DType::Bf16, DType::Fp32] {
            let plan = gemm(m, 128, 128, 8, 64, dtype);
            let specs = candidates(&plan);
            let [SpecifiedKernel::CuTeHopperGemm(spec)] = specs.as_slice() else {
                panic!("GEMM candidate")
            };
            assert_eq!(spec.lhs.width(1), 128);
            assert_eq!(spec.output.dtype, dtype);
            assert_eq!(spec.requirements.shared_memory_bytes, 65536);
            assert_eq!(spec.requirements.shared_memory_alignment, 128);
            assert_eq!(
                spec.requirements.thread_policy,
                crate::emit::native::ThreadPolicy::FullCta { threads: 128 }
            );
            let Kernel {
                prologue,
                mainloop,
                epilogue,
                ..
            } = CuTeKernelProvider
                .render(&specs[0], &bindings(&plan))
                .unwrap();
            assert!(prologue.source().contains("cute::clear(body_accumulator)"));
            assert!(
                !prologue.source().contains("lv0"),
                "K is not in scope before the loop"
            );
            let mainloop = mainloop.unwrap().source();
            assert!(mainloop.contains("offset < int64_t(128)"));
            assert!(mainloop.contains("warpgroup_wait<0>()"));
            assert!(!mainloop.contains("cute::clear"));
            assert!(!mainloop.contains("buffer2"));
            assert!(epilogue.source().contains("body_accumulator(element)"));
            let mut missing = bindings(&plan);
            missing.shared_memory = None;
            assert!(matches!(
                CuTeKernelProvider.render(&specs[0], &missing),
                Err(ProviderError::Failed(_))
            ));
        }
    }
    for plan in [
        gemm(129, 128, 64, 0, 64, DType::Bf16),
        gemm(16, 64, 64, 0, 64, DType::Bf16),
        gemm(16, 128, 32, 0, 64, DType::Bf16),
        gemm(16, 128, 64, 1, 64, DType::Bf16),
    ] {
        assert!(candidates(&plan).is_empty());
    }
}

#[test]
fn reduction_retains_fp32_state_across_tiles() {
    for dtype in [DType::Bf16, DType::Fp32] {
        let plan = reduction(5, 131, 65, dtype, true);
        let specs = candidates(&plan);
        let [SpecifiedKernel::CuTeReduceSum(spec)] = specs.as_slice() else {
            panic!("reduction")
        };
        assert!(spec.square);
        assert_eq!(spec.output.dtype, DType::Fp32);
        assert_eq!(spec.requirements.shared_memory_bytes, 0);
        let Kernel {
            prologue,
            mainloop,
            epilogue,
            ..
        } = CuTeKernelProvider
            .render(&specs[0], &bindings(&plan))
            .unwrap();
        assert!(prologue.source().contains("float body_accumulator[2] = {}"));
        assert!(!prologue.source().contains("lv0"));
        let mainloop = mainloop.unwrap().source();
        assert!(mainloop.contains("__shfl_down_sync"));
        assert!(mainloop.contains("< int64_t(131)"));
        assert!(!mainloop.contains("buffer1"));
        assert!(epilogue.source().contains("buffer1"));
    }
}

#[test]
fn normalization_projects_row_vectors_and_all_ffn_operations_have_candidates() {
    let plan = normalization(DType::Bf16);
    let specs = candidates(&plan);
    let [SpecifiedKernel::CuTePointwise(spec)] = specs.as_slice() else {
        panic!("normalization")
    };
    assert_eq!(spec.inputs.len(), 2);
    assert!(!spec.inputs[0].broadcast);
    assert!(spec.inputs[1].broadcast);
    let Kernel {
        prologue, mainloop, ..
    } = CuTeKernelProvider
        .render(&specs[0], &bindings(&plan))
        .unwrap();
    assert!(mainloop.is_none());
    assert!(
        prologue
            .source()
            .contains("cute::make_stride(int64_t(1), int64_t(0))")
    );
}

/// Only the launch wrapper is test-specific; phase placement comes from combine.
fn body(plan: &PhysicalPlan, source: &mut String) {
    use crate::emit::native::combine::{CombinedPlan, CombinedStatement, combine};
    fn walk(
        plan: &CombinedPlan<'_>,
        statements: &[CombinedStatement],
        bindings: &KernelBindings,
        source: &mut String,
    ) {
        for statement in statements {
            match statement {
                CombinedStatement::Body(body) => {
                    source.push_str(&plan.render_body(body, bindings).unwrap().code)
                }
                CombinedStatement::Loop { domain, body, .. } => {
                    writeln!(
                        source,
                        "for (int64_t {} = {}; {} < {}; {} += {}) {{",
                        domain.variable,
                        render::render_index_expression(&domain.start, bindings).unwrap(),
                        domain.variable,
                        render::render_index_expression(&domain.stop, bindings).unwrap(),
                        domain.variable,
                        render::render_index_expression(&domain.step, bindings).unwrap()
                    )
                    .unwrap();
                    let mut nested = bindings.clone();
                    nested
                        .indices
                        .insert(domain.variable.clone(), domain.variable.clone());
                    walk(plan, body, &nested, source);
                    source.push_str("}\n");
                }
            }
        }
    }
    let prepared = prepare(plan).unwrap();
    let provider = CuTeKernelProvider;
    let collected = collect(&prepared, plan_execution(plan), &[&provider]).unwrap();
    let combined = combine(&prepared, plan_execution(plan), collected).unwrap();
    walk(&combined, &combined.statements, &bindings(plan), source);
}

#[test]
#[ignore = "requires nvcc and the vendored CuTe headers"]
fn ffn_kernels_compile_with_nvcc() {
    let mut source = String::from(
        "#include <cute/tensor.hpp>\n#include <cutlass/bfloat16.h>\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <cmath>\n",
    );
    let mut plans = vec![
        gemm(16, 128, 64, 0, 64, DType::Bf16),
        gemm(128, 128, 128, 8, 64, DType::Fp32),
        reduction(5, 131, 65, DType::Bf16, true),
        reduction(5, 131, 65, DType::Fp32, false),
        normalization(DType::Fp32),
        normalization(DType::Bf16),
    ];
    plans.extend(crate::emit::native::combine::tests::pipelines());
    for (i, plan) in plans.iter().enumerate() {
        let parameters = plan
            .value_instances()
            .map(|(id, v)| format!("{}* buffer{}", render::cpp_type(v.dtype()), id.index()))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(source, "__global__ void test{i}({parameters}) {{\nextern __shared__ __align__(128) unsigned char scratch[];").unwrap();
        body(plan, &mut source);
        source.push_str("}\n");
    }
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("ffn.cu");
    let object = directory.path().join("ffn.o");
    std::fs::write(&input, &source).unwrap();
    let result = Command::new(std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .args(["-std=c++17", "-arch=sm_90a", "-c"])
        .arg(&input)
        .arg("-I")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/third_party/cutlass/include"
        ))
        .arg("-o")
        .arg(&object)
        .output()
        .expect("run nvcc");
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
