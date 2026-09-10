use trinity_lowering::*;

/// One task accessing a large tensor; no device allocation or huge task graph.
pub fn large_offset(world_size: usize, gemm: bool) -> PhysicalPlan {
    let text = if gemm {
        let output = "(view (output Y) (layout (axis m 128) (axis n 128)))";
        let index = "(keyed_index (slot m fulltile) (slot n fulltile))";
        format!(
            "(ploop 65536 65664 128 row (sloop 0 64 64 k
              (store {output}
                (+ (load {output} {index})
                   (@ (load (view (input X) (layout (axis m 65664) (axis k 65536)))
                            (keyed_index (slot m (tile row 128)) (slot k (tile k 64))))
                      (load (view (input W) (layout (axis k 65536) (axis n 128)))
                            (keyed_index (slot k (tile k 64)) (slot n fulltile)))))
                {index})))"
        )
    } else {
        "(ploop 4294967296 4294967424 128 col
           (store (view (output Y) (layout (axis a 128)))
             (load (view (input X) (layout (axis a 4294967424)))
                   (keyed_index (slot a (tile col 128))))
             (keyed_index (slot a fulltile))))"
            .into()
    };
    let config = LoopIrConfig {
        world_size,
        dtypes: ["X", "W", "Y"]
            .into_iter()
            .map(|name| (name.into(), DType::Bf16))
            .collect(),
        ..Default::default()
    };
    lower_loop_ir(&text, &config).unwrap().remove(0)
}

pub fn ffn(split: i64) -> PhysicalPlan {
    let meta: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/loop_ir/IR.v3.meta.json")).unwrap();
    let mut c = LoopIrConfig {
        world_size: 2,
        ..Default::default()
    };
    for name in meta["tensor_shapes"].as_object().unwrap().keys() {
        c.dtypes.insert(
            name.clone(),
            if name == "attn_O3" {
                DType::Fp32
            } else {
                DType::Bf16
            },
        );
    }
    c.dtypes
        .insert("__split_fdba4cffaeb51ebe".into(), DType::Fp32);
    for (name, value) in meta["example_bindings"].as_object().unwrap() {
        c.symbols.insert(name.clone(), value.as_i64().unwrap());
    }
    c.symbols.insert("__nsplit_fdba4cffaeb51ebe".into(), split);
    let text = include_str!("../fixtures/loop_ir/IR.v3.split_k.txt")
        .replace("16384", "512")
        .replace("4096", "256");
    lower_loop_ir(&text, &c).unwrap().remove(0)
}
