use trinity_lowering::{
    analysis::analyze_text,
    triton::{Options, TensorDType as D, TritonPlan, lower},
};

fn view(kind: &str, name: &str, vector: bool) -> String {
    format!(
        "(view ({kind} {name}) (layout (axis m 16) {}))",
        if vector { "" } else { "(axis n 16)" }
    )
}
fn load(kind: &str, name: &str, vector: bool) -> String {
    format!("(load {} (keyed_index))", view(kind, name, vector))
}
fn store(kind: &str, name: &str, vector: bool, expr: &str) -> String {
    format!(
        "(ploop 0 16 16 m (store {} {expr} (keyed_index)))",
        view(kind, name, vector)
    )
}
fn softmax(cast: bool) -> String {
    let exp = format!("(exp {})", load("input", "X", false));
    let exp = if cast {
        format!("(cast bf16 {exp})")
    } else {
        exp
    };
    format!(
        "(seq {} (seq {} (seq {} {})))",
        store("tensor", "E", false, &exp),
        store("tensor", "Copy", false, &load("tensor", "E", false)),
        store(
            "tensor",
            "S",
            true,
            &format!("(rsum {} 1)", load("tensor", "Copy", false))
        ),
        store(
            "output",
            "Y",
            false,
            &format!(
                "(/ {} (bcast {} 1))",
                load("tensor", "Copy", false),
                load("tensor", "S", true)
            )
        )
    )
}
fn lower_text(text: &str, options: Options) -> TritonPlan {
    lower(analyze_text(text).unwrap(), options).unwrap()
}
fn dtype(plan: &TritonPlan, name: &str) -> D {
    plan.tensor_dtype(plan.analysis().tensor_id(name).unwrap())
}

fn tiled_exp_sum() -> &'static str {
    "(ploop 0 16 16 m (seq
      (sloop 0 1024 64 n (store
        (view (tensor E) (layout (axis m 16) (axis n 1024)))
        (exp (load (view (input X) (layout (axis m 16) (axis n 1024)))
                   (keyed_index (slot m fulltile) (slot n (tile n 64)))))
        (keyed_index (slot m fulltile) (slot n (tile n 64)))))
      (store (view (output Y) (layout (axis m 16)))
        (rsum (load (view (tensor E) (layout (axis m 16) (axis n 1024)))
                    (keyed_index)) 1) (keyed_index))))"
}

#[test]
fn a_full_tile_read_waits_for_preceding_tiled_global_stores() {
    let plan = lower_text(
        tiled_exp_sum(),
        Options {
            default_dtype: D::Bf16,
            ..Default::default()
        },
    );
    assert_eq!(plan.kernels().len(), 1);
    let source = plan.emit();
    let store = source.find("tl.store(E_ptr").unwrap();
    let barrier = source.find("tl.debug_barrier()").unwrap();
    let load = source.find("tl.load(E_ptr").unwrap();
    assert!(store < barrier && barrier < load);
    // Publication belongs after the producer loop, not on every store iteration.
    assert!(source.contains("\n    tl.debug_barrier()\n"));
}

#[test]
fn fp32_opmath_does_not_promote_cross_kernel_storage() {
    for model in [D::Fp16, D::Bf16] {
        let plan = lower_text(
            &softmax(false),
            Options {
                default_dtype: model,
                ..Default::default()
            },
        );
        assert_eq!(plan.kernels().len(), 4);
        for name in ["E", "Copy", "S"] {
            assert_eq!(dtype(&plan, name), model);
        }
        for name in ["X", "Y"] {
            assert_eq!(dtype(&plan, name), model);
        }
        let code = plan.emit();
        for name in ["E", "Copy", "S"] {
            let allocation = code
                .lines()
                .find(|l| l.contains(&format!("{name} = torch.empty")))
                .unwrap();
            let ty = if model == D::Fp16 {
                "float16"
            } else {
                "bfloat16"
            };
            assert!(
                allocation.contains(&format!("dtype=torch.{ty}")),
                "{allocation}"
            );
            let store = code
                .lines()
                .find(|l| l.contains(&format!("tl.store({name}_ptr")))
                .unwrap();
            assert!(store.contains(&format!(".to(tl.{ty})")), "{store}");
        }
        assert!(
            code.lines()
                .filter(|l| l.contains(" = tl.load("))
                .all(|l| l.ends_with(".to(tl.float32)"))
        );
        assert!(
            code.lines()
                .filter(|l| l.contains("tl.sum("))
                .all(|l| l.contains("dtype=tl.float32"))
        );
    }
}

#[test]
fn explicit_storage_and_casts_remain_precision_boundaries() {
    let plan = lower_text(
        &softmax(false),
        Options {
            default_dtype: D::Bf16,
            dtypes: [("E".into(), D::Fp32)].into(),
            ..Default::default()
        },
    );
    assert_eq!(dtype(&plan, "E"), D::Fp32);
    assert_eq!(dtype(&plan, "S"), D::Fp32);
    let cast = lower_text(
        &softmax(true),
        Options {
            default_dtype: D::Bf16,
            ..Default::default()
        },
    );
    // A following reduction computes in FP32 without widening the cast value.
    assert_eq!(dtype(&cast, "E"), D::Bf16);
    let source = cast.emit();
    let exp = source
        .lines()
        .find(|line| line.contains("tl.exp("))
        .unwrap();
    assert!(exp.contains("to(tl.bfloat16)"), "{exp}");
    assert_eq!(dtype(&cast, "Copy"), D::Bf16);
    let default_plan = lower_text(&softmax(false), Options::default());
    assert_eq!(dtype(&default_plan, "E"), D::Fp16);
}

#[test]
fn exp_does_not_request_wider_logits_through_copies() {
    let dot = format!(
        "(@ {} {})",
        load("input", "A", false),
        load("input", "B", false)
    );
    let text = format!(
        "(seq {} (seq {} {}))",
        store("tensor", "Logits", false, &dot),
        store("tensor", "Alias", false, &load("tensor", "Logits", false)),
        store(
            "output",
            "Y",
            false,
            &format!("(exp {})", load("tensor", "Alias", false))
        )
    );
    let plan = lower_text(
        &text,
        Options {
            default_dtype: D::Bf16,
            ..Default::default()
        },
    );
    for name in ["Logits", "Alias"] {
        assert_eq!(dtype(&plan, name), D::Bf16);
    }
    for name in ["A", "B", "Y"] {
        assert_eq!(dtype(&plan, name), D::Bf16);
    }
    let code = plan.emit();
    let dot = code.lines().find(|l| l.contains("tl.dot(")).unwrap();
    assert_eq!(
        dot.split(", out_dtype")
            .next()
            .unwrap()
            .matches("to(tl.bfloat16)")
            .count(),
        2,
        "{dot}"
    );
    assert!(!dot.contains("input_precision='ieee'"));
}

#[test]
fn bf16_dot_never_silently_narrows_to_fp16() {
    let dot = format!(
        "(@ {} {})",
        load("input", "A", false),
        load("input", "B", false)
    );
    for options in [
        Options {
            default_dtype: D::Bf16,
            ..Default::default()
        },
        Options {
            dtypes: ["A", "B", "Y"].map(|n| (n.into(), D::Bf16)).into(),
            ..Default::default()
        },
    ] {
        let plan = lower_text(&store("output", "Y", false, &dot), options);
        let source = plan.emit();
        let line = source.lines().find(|l| l.contains("tl.dot(")).unwrap();
        assert!(line.matches("to(tl.bfloat16)").count() >= 2, "{line}");
        assert!(!source.contains("tl.float16"));
    }
}

#[test]
fn internal_recurrence_computes_in_fp32_without_widening_storage() {
    let matrix = |role, name| view(role, name, false);
    let acc = load("tensor", "Acc", false);
    let text = format!(
        "(seq (ploop 0 16 16 m (sloop 0 2 1 k
        (store {} (+ {acc} {}) (keyed_index)))) {})",
        matrix("tensor", "Acc"),
        load("input", "X", false),
        store("output", "Y", false, &acc)
    );
    let plan = lower_text(
        &text,
        Options {
            default_dtype: D::Bf16,
            ..Default::default()
        },
    );
    assert_eq!(dtype(&plan, "Acc"), D::Bf16);
    assert_eq!(dtype(&plan, "Y"), D::Bf16);
    assert!(plan.emit().contains("tl.zeros((16, 16), dtype=tl.float32)"));
}

#[test]
fn fp32_accumulator_used_by_another_gemm_keeps_its_logical_operand_type() {
    let a = load("input", "A", false);
    let b = load("input", "B", false);
    let acc = load("tensor", "Acc", false);
    let text = format!(
        "(ploop 0 16 16 m (seq (sloop 0 2 1 k
          (store {} (+ {acc} (@ {a} {b})) (keyed_index)))
          (store {} (@ {acc} {b}) (keyed_index))))",
        view("tensor", "Acc", false),
        view("output", "Y", false)
    );
    // Explicit BF16 sources under an FP16 default also exercise recurrence
    // inference: an unknown self-read must not spuriously promote to FP32.
    let plan = lower_text(
        &text,
        Options {
            dtypes: [("A".into(), D::Bf16), ("B".into(), D::Bf16)].into(),
            ..Default::default()
        },
    );
    assert_eq!(dtype(&plan, "Acc"), D::Bf16);
    let source = plan.emit();
    assert!(source.contains("tl.zeros((16, 16), dtype=tl.float32)"));
    assert!(!source.contains("input_precision='ieee'"));
    assert!(source.lines().filter(|l| l.contains("tl.dot(")).count() >= 2);
    for dot in source.lines().filter(|l| l.contains("tl.dot(")) {
        assert_eq!(dot.matches("to(tl.bfloat16)").count(), 2, "{dot}");
    }
}

fn exp_dot(cast: Option<&str>) -> String {
    let e = load("tensor", "E", false);
    let operand = cast
        .map(|dtype| format!("(cast {dtype} {e})"))
        .unwrap_or(e.clone());
    format!(
        "(seq {} (seq {} (seq {} {})))",
        store(
            "tensor",
            "E",
            false,
            &format!("(exp {})", load("input", "X", false))
        ),
        store(
            "tensor",
            "N",
            false,
            &format!("(@ {operand} {})", load("input", "B", false))
        ),
        store("tensor", "S", true, &format!("(rsum {e} 1)")),
        store(
            "output",
            "Y",
            false,
            &format!(
                "(/ {} (bcast {} 1))",
                load("tensor", "N", false),
                load("tensor", "S", true)
            )
        )
    )
}

#[test]
fn exp_dot_operands_follow_logical_types_and_explicit_casts() {
    for model in [D::Fp16, D::Bf16] {
        let model_ty = if model == D::Fp16 {
            "float16"
        } else {
            "bfloat16"
        };
        for (cast, expected) in [
            (None, model_ty),
            (Some("fp16"), "float16"),
            (Some("bf16"), "float32"),
            (Some("fp32"), "float32"),
        ] {
            let plan = lower_text(
                &exp_dot(cast),
                Options {
                    default_dtype: model,
                    ..Default::default()
                },
            );
            let source = plan.emit();
            let dot = source.lines().find(|l| l.contains("tl.dot(")).unwrap();
            // Mixed BF16/FP16 operands promote; identical explicit types do not.
            let expected = if cast == Some("fp16") && model == D::Bf16 {
                "float32"
            } else if cast == Some("bf16") && model == D::Bf16 {
                "bfloat16"
            } else {
                expected
            };
            assert!(dot.contains(&format!("to(tl.{expected})")), "{dot}");
            assert_eq!(
                dot.contains("input_precision='ieee'"),
                expected == "float32",
                "{dot}"
            );
        }
        let inline = store(
            "output",
            "Y",
            false,
            &format!(
                "(@ (exp {}) {})",
                load("input", "X", false),
                load("input", "B", false)
            ),
        );
        let source = lower_text(
            &inline,
            Options {
                default_dtype: model,
                ..Default::default()
            },
        )
        .emit();
        assert!(!source.contains("input_precision='ieee'"));
        let dot = source.lines().find(|l| l.contains("tl.dot(")).unwrap();
        assert_eq!(
            dot.split(", out_dtype")
                .next()
                .unwrap()
                .matches(&format!("to(tl.{model_ty})"))
                .count(),
            2,
            "{dot}"
        );
        assert!(dot.contains("out_dtype=tl.float32"), "{dot}");
    }
}

#[test]
fn explicit_fp32_sources_propagate_forward_to_storage_and_dot() {
    for name in ["X", "E"] {
        let plan = lower_text(
            &exp_dot(None),
            Options {
                default_dtype: D::Bf16,
                dtypes: [(name.into(), D::Fp32)].into(),
                ..Default::default()
            },
        );
        for tensor in ["E", "N", "S"] {
            assert_eq!(dtype(&plan, tensor), D::Fp32);
        }
        assert_eq!(dtype(&plan, "B"), D::Bf16);
        assert_eq!(dtype(&plan, "Y"), D::Bf16);
        assert!(plan.emit().contains("input_precision='ieee'"));
    }
}

#[test]
fn scalar_only_storage_defaults_propagate_to_consumers() {
    let text = format!(
        "(seq {} (seq {} {}))",
        store("tensor", "One", false, "1"),
        store(
            "tensor",
            "Mixed",
            false,
            &format!(
                "(+ {} {})",
                load("tensor", "One", false),
                load("input", "B", false)
            )
        ),
        store(
            "output",
            "Y",
            false,
            &format!(
                "(@ {} {})",
                load("tensor", "Mixed", false),
                load("input", "B", false)
            )
        )
    );
    let plan = lower_text(
        &text,
        Options {
            dtypes: [("B".into(), D::Bf16)].into(),
            ..Default::default()
        },
    );
    assert_eq!(dtype(&plan, "One"), D::Fp16);
    assert_eq!(dtype(&plan, "Mixed"), D::Fp32);
    assert!(plan.emit().contains("input_precision='ieee'"));
}

fn register_pointwise() -> String {
    format!(
        "(ploop 0 16 16 m (seq (store {} (+ {} {}) (keyed_index))
          (store {} (/ {} 2) (keyed_index))))",
        view("tensor", "Tmp", false),
        load("input", "X", false),
        load("input", "X", false),
        view("output", "Y", false),
        load("tensor", "Tmp", false),
    )
}

#[test]
fn register_assignment_keeps_fp32_until_a_real_precision_boundary() {
    let plan = lower_text(&register_pointwise(), Options::default());
    assert_eq!(dtype(&plan, "Tmp"), D::Fp16);
    let source = plan.emit();
    assert!(!source.contains("Tmp_ptr"), "{source}");
    let assignment = source
        .lines()
        .find(|l| l.trim_start().starts_with("Tmp = ") && l.contains(" + "))
        .unwrap();
    assert!(assignment.ends_with(".to(tl.float32)"), "{assignment}");
    assert!(!assignment.contains("tl.float16"), "{assignment}");
}

#[test]
fn fp32_reductions_and_nested_dots_keep_logical_half_operands() {
    let x = load("input", "X", false);
    let b = load("input", "B", false);
    for operand in [
        format!("(bcast (rsum {x} 1) 1)"),
        format!("(/ {x} 2)"),
        format!("(@ {x} {b})"),
    ] {
        // Broadcasting a reduction along the contracted dimension keeps K=16.
        let operand = if operand.starts_with("(bcast") {
            format!("(* {operand} {x})")
        } else {
            operand
        };
        let source = lower_text(
            &store("output", "Y", false, &format!("(@ {operand} {b})")),
            Options {
                default_dtype: D::Bf16,
                ..Default::default()
            },
        )
        .emit();
        assert!(!source.contains("input_precision='ieee'"), "{source}");
        for dot in source.lines().filter(|l| l.contains("tl.dot(")) {
            assert!(dot.contains("to(tl.bfloat16)"), "{dot}");
            assert!(dot.contains("out_dtype=tl.float32"), "{dot}");
        }
    }
}

#[test]
#[ignore = "requires CUDA; set TRINITY_TEST_PYTHON and CUDA_VISIBLE_DEVICES"]
fn numerical_precision_matches_torch_on_gpu() {
    use std::{fs, path::PathBuf, process::Command};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("target/tests/triton_precision");
    fs::create_dir_all(&out).unwrap();
    for (name, dtype) in [("softmax_fp16", D::Fp16), ("softmax_bf16", D::Bf16)] {
        let plan = lower_text(
            &softmax(false),
            Options {
                default_dtype: dtype,
                ..Default::default()
            },
        );
        fs::write(out.join(format!("{name}.py")), plan.emit()).unwrap();
    }
    for (name, dtype) in [("exp_dot_fp16", D::Fp16), ("exp_dot_bf16", D::Bf16)] {
        let plan = lower_text(
            &exp_dot(None),
            Options {
                default_dtype: dtype,
                ..Default::default()
            },
        );
        fs::write(out.join(format!("{name}.py")), plan.emit()).unwrap();
    }
    let dot = format!(
        "(@ {} {})",
        load("input", "A", false),
        load("input", "B", false)
    );
    let plan = lower_text(
        &store("output", "Y", false, &dot),
        Options {
            default_dtype: D::Bf16,
            ..Default::default()
        },
    );
    fs::write(out.join("gemm_bf16.py"), plan.emit()).unwrap();
    let plan = lower_text(
        tiled_exp_sum(),
        Options {
            default_dtype: D::Bf16,
            ..Default::default()
        },
    );
    fs::write(out.join("tiled_exp_sum.py"), plan.emit()).unwrap();
    let x = load("input", "X", false);
    let code = lower_text(
        &store(
            "output",
            "Y",
            false,
            &format!("(/ (+ (cast fp16 {x}) (cast fp16 {x})) 2)"),
        ),
        Options {
            ..Default::default()
        },
    )
    .emit();
    fs::write(out.join("cast_opmath.py"), code).unwrap();
    fs::write(
        out.join("register_opmath.py"),
        lower_text(&register_pointwise(), Options::default()).emit(),
    )
    .unwrap();
    let output = Command::new(std::env::var("TRINITY_TEST_PYTHON").unwrap_or("python3".into()))
        .arg(root.join("tests/triton_precision_gpu.py"))
        .arg(out)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!("{}", String::from_utf8_lossy(&output.stdout));
}
