use super::*;

fn builder() -> PhysicalPlanBuilder {
    PhysicalPlanBuilder::new(IrConfig::default().target, 1)
}

#[test]
fn scalar_decoding_preserves_integer_values_and_float_bits() {
    let builder = builder();
    for (text, expected) in [
        ("-9223372036854775808", Constant::Integer(i64::MIN)),
        ("9223372036854775807", Constant::Integer(i64::MAX)),
        ("1.25", Constant::Float32(1.25f32.to_bits())),
        ("-0.0", Constant::Float32((-0.0f32).to_bits())),
        ("(float_bits 2147483648)", Constant::Float32(0x80000000)),
        ("(float_bits 2143289345)", Constant::Float32(0x7fc00001)),
    ] {
        assert_eq!(
            builder.parse_expression(text).unwrap(),
            Expression::Constant(expected)
        );
    }
    assert_eq!(
        builder
            .parse_expression("(unsqueeze (bcast (rsum (sqr 2) 0) 1) 2)")
            .unwrap(),
        Expression::Unsqueeze {
            axis: 2,
            value: Box::new(Expression::Broadcast {
                axis: 1,
                value: Box::new(Expression::ReduceSum {
                    axis: 0,
                    value: Box::new(Expression::Sqr(Box::new(Expression::Constant(
                        Constant::Integer(2)
                    )))),
                }),
            }),
        }
    );
}

#[test]
fn access_decoding_resolves_ids_and_orders_slots_by_tensor_axis() {
    let mut builder = builder();
    let x = builder.add_named_value("X", DType::Fp32, [8, 16], Storage::External);
    let first = builder
        .parse_expression(
            "(load (view (tensor X) (layout (axis row 8) (axis col 16)))
          (keyed_index (slot col (clipped_tile j 7)) (slot row (elem i))))",
        )
        .unwrap();
    let renamed = builder
        .parse_expression(
            "(load (view (tensor X) (layout (axis a 8) (axis b 16)))
          (keyed_index (slot a (elem i)) (slot b (clipped_tile j 7))))",
        )
        .unwrap();
    assert_eq!(first, renamed);
    assert_eq!(
        first,
        Expression::Load(TensorAccess::new(
            x,
            [
                AccessIndex::Elem("i".into()),
                AccessIndex::ClippedTile {
                    variable: "j".into(),
                    width: 7
                },
            ]
        ))
    );
    let view = "(view (tensor X) (layout (axis row 8) (axis col 16)))";
    assert_eq!(
        builder
            .parse_expression(&format!("(load {view} (keyed_index))"))
            .unwrap(),
        builder
            .parse_expression(&format!(
                "(load {view} (keyed_index (slot col fulltile) (slot row fulltile)))"
            ))
            .unwrap(),
    );
}

#[test]
fn malformed_notation_is_rejected_at_the_input_boundary_with_offsets() {
    let mut builder = builder();
    builder.add_named_value("X", DType::Fp32, [16], Storage::External);
    let view = "(view (tensor X) (layout (axis a 16)))";
    for (text, diagnostic) in [
        ("(unknown 1)".to_owned(), "unsupported operation"),
        ("(+ 1)".into(), "expects 2 arguments"),
        ("(sqrt invalid)".into(), "invalid scalar constant"),
        ("(float_bits 4294967296)".into(), "invalid FP32 bits"),
        ("(rsum 1 0.5)".into(), "unresolved integer"),
        (
            format!("(load {view} (keyed_index (slot a (tile i 1.5))))"),
            "unresolved integer",
        ),
        (
            format!("(load {view} (keyed_index (slot a fulltile) (slot a fulltile)))"),
            "duplicate index slot",
        ),
        (
            format!("(load {view} (keyed_index (slot unknown fulltile)))"),
            "index slot absent",
        ),
        (
            format!("(load {} (keyed_index))", view.replace("a 16", "a 8")),
            "view extent differs",
        ),
        (
            format!(
                "(load {} (keyed_index))",
                view.replace("tensor X", "tensor missing")
            ),
            "unknown tensor",
        ),
    ] {
        let error = builder.parse_expression(&format!("  {text}")).unwrap_err();
        assert!(error.offset >= 2, "{text}: {error}");
        assert!(error.message.contains(diagnostic), "{text}: {error}");
    }
    builder.add_named_value("X", DType::Fp32, [16], Storage::External);
    let error = builder
        .parse_expression(&format!("(load {view} (keyed_index))"))
        .unwrap_err();
    assert!(error.message.contains("ambiguous tensor"));
}

#[test]
fn builder_only_forms_do_not_extend_the_text_ir_language() {
    let mut builder = builder();
    builder.add_named_value("X", DType::Fp32, [16], Storage::External);
    builder.add_named_value("Y", DType::Fp32, [32], Storage::External);
    let config = IrConfig {
        dtypes: [("X".into(), DType::Fp32), ("Y".into(), DType::Fp32)].into(),
        ..Default::default()
    };
    let source = "(view (input X) (layout (axis a 16)))";
    let destination = "(view (output Y) (layout (axis a 32)))";
    for rhs in [
        "(relu 1)".to_owned(),
        "(float_bits 0)".into(),
        format!("(load {source} (keyed_index (slot a (clipped_tile i 7))))"),
    ] {
        let text = format!("(store {destination} {rhs} (keyed_index))");
        builder.parse_expression(&text).unwrap();
        let error = lower_ir(&text, &config).err().unwrap();
        assert!(error.message.contains("unsupported"), "{error}");
    }
    let gather = format!("(all_gather {source} (keyed_index) {destination} (keyed_index) 0)");
    assert!(matches!(
        builder.parse_expression(&gather).unwrap(),
        Expression::AllGather { axis: 0, .. }
    ));
    assert!(
        lower_ir(&gather, &config)
            .err()
            .unwrap()
            .message
            .contains("unsupported program node all_gather")
    );
}
