use super::super::access;
use super::*;
use crate::emit::native::NativeKernelProvider;
use crate::emit::provider::CuTeKernelProvider;
use crate::emit::{collect::collect, execution::plan_execution, prepare::prepare};
use crate::{
    DType, IndexExpr, Loop, LoopDomain, LoweringConfig, PhysicalPlan, PhysicalPlanBuilder,
    Statement, Storage,
};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::process::Command;

fn view(name: &str, shape: &[usize]) -> String {
    let axes = shape
        .iter()
        .enumerate()
        .map(|(axis, size)| format!("(axis a{axis} {size})"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("(view (tensor {name}) (layout {axes}))")
}

fn load(name: &str, shape: &[usize], index: &str) -> String {
    format!("(load {} {index})", view(name, shape))
}

fn store(name: &str, shape: &[usize], rhs: &str, index: &str) -> String {
    format!("(store {} {rhs} {index})", view(name, shape))
}

fn loop_(kind: LoopKind, step: i64) -> Loop {
    Loop {
        kind,
        domain: LoopDomain {
            variable: "i".into(),
            start: IndexExpr::Constant(0),
            stop: IndexExpr::Constant(step * 2),
            step: IndexExpr::Constant(step),
        },
        body: Vec::new(),
    }
}

fn program(
    shape: &[usize],
    input_dtype: DType,
    output_dtype: DType,
    rhs: &str,
    index: &str,
    loops: Vec<Loop>,
) -> PhysicalPlan {
    let mut builder = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
    let x = builder.add_named_value("X", input_dtype, shape.iter().copied(), Storage::External);
    let y = builder.add_named_value("Y", output_dtype, shape.iter().copied(), Storage::External);
    builder.bind_input("X", x);
    let inflows = if rhs.contains("$y") {
        vec![x, y]
    } else {
        vec![x]
    };
    let rhs = rhs
        .replace("$x", &load("X", shape, index))
        .replace("$y", &load("Y", shape, index));
    let op = builder.add_operation(
        inflows,
        [y],
        builder
            .parse_expression(&store("Y", shape, &rhs, index))
            .unwrap(),
    );
    let mut statement = Statement::Operation(op);
    for mut loop_ in loops.into_iter().rev() {
        loop_.body = vec![statement];
        statement = Statement::Loop(loop_);
    }
    builder.build(vec![statement], "Y", y).unwrap()
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

fn bindings(plan: &PhysicalPlan, prefix: &str) -> KernelBindings {
    KernelBindings {
        block_threads: 128,
        registers: BTreeMap::new(),
        values: plan
            .value_instances()
            .map(|(id, _)| (id, format!("buffer{}", id.index())))
            .collect(),
        indices: BTreeMap::from([
            ("lv0".into(), "tile_origin".into()),
            ("lv1".into(), "inner_origin".into()),
        ]),
        prefix: prefix.into(),
        shared_memory: None,
    }
}

fn render_body(specification: &SpecifiedKernel, bindings: &KernelBindings) -> String {
    let Kernel {
        prologue,
        mainloop,
        epilogue,
        ..
    } = CuTeKernelProvider.render(specification, bindings).unwrap();
    assert!(mainloop.is_none(), "pointwise has no main loop");
    format!("{{\n{}\n{}\n}}\n", prologue.source(), epilogue.source())
}

#[test]
fn nested_pointwise_deduplicates_loads_and_exposes_requirements() {
    let plan = program(
        &[3, 131],
        DType::Bf16,
        DType::Fp32,
        "(relu (- (/ (+ (sqr $x) (sqrt $x)) 2) (* $x (sigmoid $x))))",
        "(keyed_index)",
        vec![],
    );
    let candidates = candidates(&plan);
    assert_eq!(candidates.len(), 1);
    let SpecifiedKernel::CuTePointwise(specification) = &candidates[0] else {
        panic!("pointwise candidate")
    };
    assert_eq!(specification.inputs.len(), 1);
    assert_eq!(specification.inputs[0].access.dtype, DType::Bf16);
    assert_eq!(specification.output.dtype, DType::Fp32);
    let requirements = candidates[0].requirements();
    assert_eq!(
        requirements.thread_policy,
        ThreadPolicy::Flexible {
            supported: &[32, 64, 128, 256, 512, 1024],
        }
    );
    assert_eq!(requirements.shared_memory_bytes, 0);
    assert_eq!(
        requirements.alignments,
        vec![
            (specification.inputs[0].access.value, 2),
            (specification.output.value, 4)
        ]
    );
    let body = render_body(&candidates[0], &bindings(&plan, "pointwise"));
    assert_eq!(
        body.matches("cute::copy_if(pointwise_valid, pointwise_input0_thread,")
            .count(),
        1
    );
    assert!(body.contains("cute::make_tensor"));
    assert!(body.contains("cute::make_stride(int64_t(131), int64_t(1))"));
    assert!(body.contains("sqrtf("));
    assert!(body.contains("expf("));
    assert!(!body.contains("blockIdx"));
}

#[test]
fn execution_models_share_the_same_specification_and_body() {
    let plan = program(
        &[17],
        DType::Fp32,
        DType::Bf16,
        "(+ $x (float_bits 2147483648))",
        "(keyed_index)",
        vec![],
    );
    let prepared = prepare(&plan).unwrap();
    let operation = plan.operations().next().unwrap().0;
    let specifications: Vec<_> = [
        crate::emit::execution::ExecutionModel::CudaStreamed,
        crate::emit::execution::ExecutionModel::CudaPersistent,
    ]
    .into_iter()
    .map(|execution| {
        CuTeKernelProvider
            .specify(&KernelContext {
                prepared: &prepared,
                execution,
                operation,
                loops: &[],
            })
            .unwrap()
    })
    .collect();
    assert_eq!(specifications[0], specifications[1]);
    let body = render_body(&specifications[0], &bindings(&plan, "pw"));
    assert!(body.contains("__uint_as_float(2147483648u)"));
    assert!(body.contains("static_cast<cutlass::bfloat16_t>"));
}

#[test]
fn access_width_is_independent_of_loop_step_and_clipping_guards_loads() {
    for (kind, width) in [("tile", 5), ("clipped_tile", 17)] {
        let plan = program(
            &[2, 19],
            DType::Fp32,
            DType::Fp32,
            "$x",
            &format!("(keyed_index (slot a1 ({kind} i {width})))"),
            vec![loop_(LoopKind::Parallel, 7)],
        );
        let candidates = candidates(&plan);
        let SpecifiedKernel::CuTePointwise(specification) = &candidates[0] else {
            panic!("pointwise candidate")
        };
        assert_eq!(specification.output.axes[0], access::Axis::Full);
        assert!(
            matches!(&specification.output.axes[1], access::Axis::Tile { width: w, .. } if *w == width)
        );
        let body = render_body(&candidates[0], &bindings(&plan, "pw"));
        assert!(body.contains(&format!(
            "pw_shape = cute::make_shape(int64_t(2), int64_t({width}))"
        )));
        if kind == "clipped_tile" {
            assert!(
                body.find("(pw_origin1 + cute::get<1>(coordinate)) < int64_t(19)")
                    .unwrap()
                    < body
                        .find("cute::copy_if(pw_valid, pw_input0_thread,")
                        .unwrap()
            );
        }
    }
}

#[test]
fn elem_uses_the_enclosing_step_and_render_requires_bindings() {
    let plan = program(
        &[4],
        DType::Fp32,
        DType::Fp32,
        "(sqrt $x)",
        "(keyed_index (slot a0 (elem i)))",
        vec![loop_(LoopKind::Parallel, 2)],
    );
    let candidate = candidates(&plan).remove(0);
    let mut bindings = bindings(&plan, "pw");
    let body = render_body(&candidate, &bindings);
    assert!(body.contains("(tile_origin) / int64_t(2)"));
    bindings.indices.clear();
    assert!(
        matches!(CuTeKernelProvider.render(&candidate, &bindings), Err(ProviderError::Failed(message)) if message.contains("loop binding"))
    );
    bindings.indices.insert("lv0".into(), "tile_origin".into());
    bindings.values.clear();
    assert!(
        matches!(CuTeKernelProvider.render(&candidate, &bindings), Err(ProviderError::Failed(message)) if message.contains("buffer binding"))
    );
}

#[test]
fn unsupported_expressions_and_accumulations_produce_no_candidates() {
    for rhs in ["(rsum $x 0)", "(bcast $x 0)", "(@ $x $x)"] {
        let plan = program(
            &[16],
            DType::Fp32,
            DType::Fp32,
            rhs,
            "(keyed_index)",
            vec![],
        );
        assert!(candidates(&plan).is_empty(), "{rhs}");
    }
    let plan = program(
        &[16],
        DType::Fp32,
        DType::Fp32,
        "(+ $y $x)",
        "(keyed_index)",
        vec![loop_(LoopKind::Sequential, 1)],
    );
    assert!(candidates(&plan).is_empty());
    let plan = program(
        &[2, 2, 2],
        DType::Fp32,
        DType::Fp32,
        "$x",
        "(keyed_index)",
        vec![],
    );
    assert!(candidates(&plan).is_empty());
}

#[test]
fn register_ports_are_supported_while_shared_transport_is_deferred() {
    for storage in [Storage::Global, Storage::Shared, Storage::Register] {
        let mut builder = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
        let x = builder.add_named_value("X", DType::Fp32, [16], Storage::External);
        let z = builder.add_named_value("Z", DType::Fp32, [16], storage);
        let y = builder.add_named_value("Y", DType::Fp32, [16], Storage::External);
        builder.bind_input("X", x);
        let index = "(keyed_index)";
        let first = builder.add_operation(
            [x],
            [z],
            builder
                .parse_expression(&store("Z", &[16], &load("X", &[16], index), index))
                .unwrap(),
        );
        let second = builder.add_operation(
            [z],
            [y],
            builder
                .parse_expression(&store("Y", &[16], &load("Z", &[16], index), index))
                .unwrap(),
        );
        let mut loop_ = loop_(LoopKind::Parallel, 1);
        loop_.body = vec![Statement::Operation(first), Statement::Operation(second)];
        let plan = builder.build(vec![Statement::Loop(loop_)], "Y", y).unwrap();
        assert_eq!(
            candidates(&plan).len(),
            if storage == Storage::Shared { 0 } else { 2 }
        );
    }
}

#[test]
fn mismatched_accesses_and_communication_are_unsupported() {
    let mut builder = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 1);
    let x = builder.add_named_value("X", DType::Fp32, [16], Storage::External);
    let y = builder.add_named_value("Y", DType::Fp32, [16], Storage::External);
    builder.bind_input("X", x);
    let rhs = load("X", &[16], "(keyed_index)");
    let op = builder.add_operation(
        [x],
        [y],
        builder
            .parse_expression(&store(
                "Y",
                &[16],
                &rhs,
                "(keyed_index (slot a0 (tile i 4)))",
            ))
            .unwrap(),
    );
    let mut loop_ = loop_(LoopKind::Parallel, 4);
    loop_.body = vec![Statement::Operation(op)];
    let plan = builder.build(vec![Statement::Loop(loop_)], "Y", y).unwrap();
    assert!(candidates(&plan).is_empty());

    let mut builder = PhysicalPlanBuilder::new(LoweringConfig::default().target(), 2);
    let x = builder.add_named_value("X", DType::Fp32, [16], Storage::External);
    let y = builder.add_named_value("Y", DType::Fp32, [32], Storage::External);
    builder.bind_input("X", x);
    let expression = builder
        .parse_expression(&format!(
            "(all_gather {} (keyed_index) {} (keyed_index) 0)",
            view("X", &[16]),
            view("Y", &[32])
        ))
        .unwrap();
    let op = builder.add_operation([x], [y], expression);
    let plan = builder
        .build(vec![Statement::Operation(op)], "Y", y)
        .unwrap();
    assert!(candidates(&plan).is_empty());
}

#[test]
fn body_prefixes_keep_generated_locals_separate() {
    let plan = program(
        &[7],
        DType::Fp32,
        DType::Fp32,
        "(sqr $x)",
        "(keyed_index)",
        vec![],
    );
    let specification = candidates(&plan).remove(0);
    let first = render_body(&specification, &bindings(&plan, "first"));
    let second = render_body(&specification, &bindings(&plan, "second"));
    assert!(!first.contains("second_"));
    assert!(!second.contains("first_"));
    assert!(matches!(
        CuTeKernelProvider.render(&specification, &bindings(&plan, "bad-prefix")),
        Err(ProviderError::Failed(_))
    ));
}

/// Compile device bodies and execute CuTe partitions on the host without a GPU.
#[test]
#[ignore = "requires nvcc and the vendored CuTe headers"]
fn generated_bodies_compile_with_nvcc() {
    let mut source = String::from(
        "#include <cute/tensor.hpp>\n#include <cutlass/bfloat16.h>\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <cmath>\n#include <cstdio>\n",
    );
    let mut plans = Vec::new();
    for input in [DType::Fp32, DType::Bf16] {
        for output in [DType::Fp32, DType::Bf16] {
            plans.push(program(
                &[3, 131],
                input,
                output,
                "(relu (- (/ (+ (sqr $x) (sqrt $x)) 2) (* $x (sigmoid $x))))",
                "(keyed_index)",
                vec![],
            ));
        }
    }
    for access in ["(tile i 5)", "(clipped_tile i 17)", "(elem i)"] {
        plans.push(program(
            &[2, 19],
            DType::Bf16,
            DType::Bf16,
            "$x",
            &format!("(keyed_index (slot a1 {access}))"),
            vec![loop_(LoopKind::Parallel, 7)],
        ));
    }
    plans.push(program(
        &[17],
        DType::Fp32,
        DType::Fp32,
        "(+ $x (float_bits 2147483648))",
        "(keyed_index)",
        vec![],
    ));
    for (index, plan) in plans.iter().enumerate() {
        let candidate = candidates(plan).remove(0);
        let parameters = plan
            .value_instances()
            .map(|(id, value)| format!("{}* buffer{}", render::cpp_type(value.dtype()), id.index()))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            source,
            "__global__ void test{index}({parameters}, int64_t tile_origin) {{"
        )
        .unwrap();
        // Two adjacent bodies exercise variable isolation in the same CUDA scope.
        source.push_str(&render_body(&candidate, &bindings(plan, "first")));
        source.push_str(&render_body(&candidate, &bindings(plan, "second")));
        source.push_str("}\n");
    }

    // CuTe's tensor operations also run on the host. Execute the generated body
    // for every thread to check strided rows, internal tile tails, and clipping.
    let plan = program(
        &[3, 1031],
        DType::Fp32,
        DType::Fp32,
        "(sqr $x)",
        "(keyed_index (slot a1 (clipped_tile i 517)))",
        vec![loop_(LoopKind::Parallel, 7)],
    );
    let candidate = candidates(&plan).remove(0);
    let ThreadPolicy::Flexible { supported } = candidate.requirements().thread_policy else {
        panic!("pointwise supports flexible CTA sizes");
    };
    for &threads in supported {
        let mut binding = bindings(&plan, "check");
        binding.block_threads = threads;
        writeln!(source, "void check_partition_{threads}(float* buffer0, float* buffer1, int64_t tile_origin, unsigned thread) {{\nconst dim3 threadIdx(thread);").unwrap();
        source.push_str(&render_body(&candidate, &binding));
        source.push_str("}\n");
    }
    let functions = supported
        .iter()
        .map(|n| format!("check_partition_{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    let counts = supported
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(source, "using Partition = void(*)(float*, float*, int64_t, unsigned);\nconst Partition partitions[] = {{{functions}}};\nconst unsigned thread_counts[] = {{{counts}}};").unwrap();
    source.push_str(
        r#"
int main() {
  constexpr int rows = 3, columns = 1031, count = rows * columns;
  float input[count + 2], output[count + 2];
  const int origins[] = {-5, 0, 517, 1029, 1035};
  for (unsigned config = 0; config < sizeof(thread_counts)/sizeof(thread_counts[0]); ++config) {
  for (int origin : origins) {
    for (int i = 0; i < count + 2; ++i) {
      input[i] = float(i % 17 + 1);
      output[i] = -1.0f;
    }
"#,
    );
    source.push_str("    for (unsigned thread = 0; thread < thread_counts[config]; ++thread)\npartitions[config](input + 1, output + 1, origin, thread);\n");
    source.push_str(
        r#"
    for (int i = 0; i < count; ++i) {
      int column = i % columns;
      float expected = column >= origin && column < origin + 517
          ? input[i + 1] * input[i + 1] : -1.0f;
      if (output[i + 1] != expected) {
        std::fprintf(stderr, "origin=%d element=%d expected=%g actual=%g\n",
                     origin, i, expected, output[i + 1]);
        return 1;
      }
    }
    if (output[0] != -1.0f || output[count + 1] != -1.0f) return 2;
  }
}
}
"#,
    );
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("pointwise.cu");
    let executable = directory.path().join("pointwise");
    std::fs::write(&input, &source).unwrap();
    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let result = Command::new(nvcc)
        .args(["-std=c++17", "-arch=sm_90a"])
        .arg(&input)
        .arg("-I")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/third_party/cutlass/include"
        ))
        .arg("-o")
        .arg(&executable)
        .output()
        .expect("run nvcc");
    assert!(
        result.status.success(),
        "{}\n{}\n{source}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let result = Command::new(&executable)
        .output()
        .expect("run host partitions");
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
