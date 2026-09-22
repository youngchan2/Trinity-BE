use std::{collections::BTreeSet, fs, path::PathBuf, process::Command};
use trinity_lowering::analysis::analyze_text;
use trinity_lowering::triton::{AutotuneOptions, Options, TritonPlan, compile, lower};

fn gemm() -> &'static str {
    "(ploop 0 64 tile_m m (ploop 0 96 tile_n n (sloop 0 128 tile_k k
      (store (view (output Y) (layout (axis m 64) (axis n 96)))
        (+ (load (view (output Y) (layout (axis m 64) (axis n 96)))
                 (keyed_index (slot m (tile m tile_m)) (slot n (tile n tile_n))))
           (@ (load (view (input A) (layout (axis m 64) (axis k 128)))
                    (keyed_index (slot m (tile m tile_m)) (slot k (tile k tile_k))))
              (load (view (input B) (layout (axis k 128) (axis n 96)))
                    (keyed_index (slot k (tile k tile_k)) (slot n (tile n tile_n))))))
        (keyed_index (slot m (tile m tile_m)) (slot n (tile n tile_n)))))))"
}
fn options() -> Options {
    Options {
        symbols: [
            ("tile_m".into(), 16),
            ("tile_n".into(), 32),
            ("tile_k".into(), 32),
        ]
        .into(),
        ..Default::default()
    }
}
fn plan(options: Options) -> TritonPlan {
    lower(analyze_text(gemm()).unwrap(), options).unwrap()
}
fn mutate() -> &'static str {
    "(ploop 0 256 tile_m m (store (view (input X) (layout (axis m 256)))
       (+ (load (view (input X) (layout (axis m 256))) (keyed_index (slot m (tile m tile_m)))) 1)
       (keyed_index (slot m (tile m tile_m)))))"
}

fn split_sum() -> &'static str {
    "(seq (mloop 0 128 tile_k k s num_splits
          (store (view (tensor P) (layout (axis s num_splits) (axis m 16)))
            (+ (load (view (tensor P) (layout (axis s num_splits) (axis m 16)))
                     (keyed_index (slot s (elem s)) (slot m fulltile)))
               (unsqueeze (rsum (load (view (input X) (layout (axis k 128) (axis m 16)))
                                      (keyed_index (slot k (tile k tile_k)) (slot m fulltile))) 0) 0))
            (keyed_index (slot s (elem s)) (slot m fulltile))))
      (ploop 0 16 16 m (store (view (output Y) (layout (axis m 16)))
        (rsum (load (view (tensor P) (layout (axis s num_splits) (axis m 16))) (keyed_index)) 0)
        (keyed_index))))"
}

#[test]
fn split_owner_retains_tuning_and_passes_selected_value_to_consumer() {
    let p = lower(analyze_text(split_sum()).unwrap(), options()).unwrap();
    assert_eq!(p.kernels().len(), 2);
    assert!(
        p.tuning_configs(0)
            .iter()
            .map(|c| c.parameters["num_splits"])
            .collect::<BTreeSet<_>>()
            .len()
            > 1
    );
    assert!(
        p.tuning_configs(1)
            .iter()
            .all(|c| !c.parameters.contains_key("num_splits"))
    );
    assert!(
        p.emit()
            .contains("kernel_0.best_config.kwargs['META_num_splits']")
    );
}

#[test]
fn default_search_varies_tiles_warps_and_stages_with_a_budget() {
    let p = plan(options());
    let configs = p.tuning_configs(0);
    assert_eq!(configs.len(), 64);
    for symbol in ["tile_m", "tile_n", "tile_k"] {
        assert!(
            configs
                .iter()
                .map(|c| c.parameters[symbol])
                .collect::<BTreeSet<_>>()
                .len()
                >= 3
        );
    }
    assert_eq!(
        configs.iter().map(|c| c.num_warps).collect::<BTreeSet<_>>(),
        [2, 4, 8].into()
    );
    assert_eq!(
        configs
            .iter()
            .map(|c| c.num_stages)
            .collect::<BTreeSet<_>>(),
        [1, 2, 3].into()
    );
}

#[test]
fn explicit_candidates_and_disabled_search_are_respected() {
    let mut o = options();
    o.tuning.insert("tile_k".into(), vec![16, 64]);
    let p = plan(o);
    assert!(
        p.tuning_configs(0)
            .iter()
            .all(|c| [16, 64].contains(&c.parameters["tile_k"]))
    );
    let mut o = options();
    o.autotune.max_configs = 1;
    let p = plan(o);
    assert_eq!(p.tuning_configs(0).len(), 1);
    assert_eq!(p.tuning_configs(0)[0].parameters["tile_k"], 32);
    let mut o = options();
    o.autotune.num_warps = vec![3];
    assert!(lower(analyze_text(gemm()).unwrap(), o).is_err());
}

#[test]
fn fixed_tile_and_storage_contracts_restrict_tunable_steps() {
    let ir = gemm().replace("(tile n tile_n)", "(tile n 32)");
    let p = lower(analyze_text(&ir).unwrap(), options()).unwrap();
    assert!(
        p.tuning_configs(0)
            .iter()
            .all(|c| c.parameters["tile_n"] == 32)
    );
    // A symbolic tile must still fit the fixed-size destination's shape.
    let ir = "(ploop 0 64 tile_m m (store (view (output Y) (layout (axis m 4) (axis n 16)))
        (unsqueeze (load (view (input X) (layout (axis m 64))) (keyed_index (slot m (tile m tile_m)))) 0)
        (keyed_index (slot m (elem m)) (slot n fulltile))))";
    let p = lower(analyze_text(ir).unwrap(), options()).unwrap();
    assert!(
        p.tuning_configs(0)
            .iter()
            .all(|c| c.parameters["tile_m"] == 16)
    );
}

#[test]
fn autotuning_restores_mutated_inputs() {
    let code = compile(mutate(), options()).unwrap();
    assert!(code.contains("restore_value=['X_ptr']"));
    // Register-zeroed GEMM output is write-only globally; cloning it would
    // needlessly dominate the tuning time of small kernels.
    assert!(plan(options()).emit().contains("restore_value=[]"));
    let mut o = options();
    o.autotune = AutotuneOptions {
        max_configs: 12,
        ..Default::default()
    };
    assert_eq!(plan(o).tuning_configs(0).len(), 12);
}

#[test]
#[ignore = "requires CUDA; set TRINITY_TEST_PYTHON and CUDA_VISIBLE_DEVICES"]
fn tuned_kernels_match_torch_and_mutate_inputs_once() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("target/tests/triton_tuning");
    fs::create_dir_all(&out).unwrap();
    fs::write(out.join("gemm.py"), plan(options()).emit()).unwrap();
    fs::write(out.join("mutate.py"), compile(mutate(), options()).unwrap()).unwrap();
    fs::write(
        out.join("split_sum.py"),
        compile(split_sum(), options()).unwrap(),
    )
    .unwrap();
    let result = Command::new(std::env::var("TRINITY_TEST_PYTHON").unwrap_or("python3".into()))
        .arg(root.join("tests/triton_tuning_gpu.py"))
        .arg(out)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    println!("{}", String::from_utf8_lossy(&result.stdout));
}
