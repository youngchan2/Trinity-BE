use trinity_lowering::{
    analyzer::{ScopeKind, analyze_text},
    triton::{InitialValue, Options, Storage, compile, lower},
};

fn options(shapes: &[(&str, &[usize])]) -> Options {
    Options {
        shapes: shapes
            .iter()
            .map(|(name, shape)| (name.to_string(), shape.to_vec()))
            .collect(),
        symbols: [
            ("tile_n".into(), 128),
            ("tile_k".into(), 64),
            ("tile_p".into(), 64),
        ]
        .into(),
    }
}

fn corpus_options(ffn: bool) -> Options {
    let mut result = options(&[]);
    let pairs: &[(&str, &[usize])] = if ffn {
        &[
            ("O2", &[16, 4096]),
            ("WO", &[4096, 4096]),
            ("X", &[16, 4096]),
            ("attn_O1", &[16, 4096]),
            ("attn_O2", &[16, 4096]),
            ("attn_O3", &[16]),
            ("attn_O_norm", &[16, 4096]),
            ("WFF1a", &[4096, 16384]),
            ("WFF1b", &[4096, 16384]),
            ("FF1a", &[16, 16384]),
            ("FF1b", &[16, 16384]),
            ("FF1b_silu", &[16, 16384]),
            ("FF1", &[16, 16384]),
            ("WFF2", &[16384, 4096]),
            ("FF2", &[16, 4096]),
        ]
    } else {
        &[
            ("X", &[16, 4096]),
            ("WQ", &[4096, 4096]),
            ("WK", &[4096, 4096]),
            ("WV", &[4096, 4096]),
            ("Q1", &[16, 4096]),
            ("K1", &[16, 4096]),
            ("V1", &[16, 4096]),
            ("Q", &[32, 16, 128]),
            ("K", &[32, 16, 128]),
            ("V", &[32, 16, 128]),
            ("K_cache", &[32, 1024, 128]),
            ("V_cache", &[32, 1024, 128]),
            ("C", &[32, 16, 1024]),
            ("C_exp", &[32, 16, 1024]),
            ("C_div", &[32, 16, 1024]),
            ("C_sum", &[32, 16]),
            ("O", &[32, 16, 128]),
            ("O2", &[16, 4096]),
        ]
    };
    for (name, shape) in pairs {
        result.shapes.insert(name.to_string(), shape.to_vec());
    }
    result
}

#[test]
fn pinned_ffn_and_vanilla_generate_complete_python() {
    for (corpus, ffn) in [
        (include_str!("fixtures/analyzer/ffn_cases.txt"), true),
        (include_str!("fixtures/analyzer/vanilla_cases.txt"), false),
    ] {
        for line in corpus.lines() {
            let (id, text) = line.split_once(':').unwrap();
            let plan = lower(analyze_text(text).unwrap(), corpus_options(ffn))
                .unwrap_or_else(|e| panic!("{ffn}/{id}: {e}"));
            let source = plan.emit();
            assert_eq!(source, plan.emit(), "emission must not mutate analysis");
            assert!(source.contains("def forward("));
            assert!(source.contains("TENSOR_PARAMS = ["));
            assert!(source.contains("BLOCK_PARAMS = ["));
            assert!(!source.contains("**tensors"));
            assert!(!source.contains("torch.empty"));
            assert!(!source.contains("_scratch"));
            assert!(!source.contains("tl.debug_barrier"));
            assert!(source.contains("tl.dot("));
            assert!(!source.contains("TODO"));
            if ["177", "307", "1019", "1230"].contains(&id) {
                let tid = plan.analysis().tensor_id("attn_O_norm").unwrap();
                let tp = plan
                    .kernels()
                    .iter()
                    .find_map(|k| k.tensors.get(&tid))
                    .unwrap();
                assert_eq!(tp.storage, Storage::Register);
                assert!(tp.initialization.is_none());
                assert!(tp.export_scope.is_none());
            }
        }
    }
}

#[test]
fn tiled_matmul_initializes_before_reduction_and_exports_after_it() {
    let ir = "(ploop 0 10 4 n (sloop 0 9 4 k (store (output C) (+ (load (output C) (index fulltile (tile n))) (@ (load (input A) (index fulltile (tile k))) (load (input B) (index (tile k) (tile n))))) (index fulltile (tile n)))))";
    let p = lower(
        analyze_text(ir).unwrap(),
        options(&[("A", &[3, 9]), ("B", &[9, 10]), ("C", &[3, 10])]),
    )
    .unwrap();
    let tid = p.analysis().tensor_id("C").unwrap();
    let t = &p.kernels()[0].tensors[&tid];
    let init = t.initialization.as_ref().unwrap();
    assert_eq!(init.value, InitialValue::Zero);
    assert_eq!(p.analysis().scope(init.scope).kind, ScopeKind::ParallelLoop);
    assert_eq!(t.export_scope, Some(init.scope));
    let src = p.emit();
    assert!(src.find("tl.zeros").unwrap() < src.find("for k").unwrap());
    assert!(src.contains("tl.float16"));
    assert!(src.contains("< 9"));
    assert!(src.contains("< 10"));
}

#[test]
fn sibling_loop_tiles_use_the_existing_global_tensor_interface() {
    let line = include_str!("fixtures/analyzer/ffn_cases.txt")
        .lines()
        .next()
        .unwrap();
    let p = lower(
        analyze_text(line.split_once(':').unwrap().1).unwrap(),
        corpus_options(true),
    )
    .unwrap();
    let tid = p.analysis().tensor_id("attn_O_norm").unwrap();
    assert_eq!(p.kernels()[2].tensors[&tid].storage, Storage::Materialized);
    let src = p.emit();
    assert!(src.contains("attn_O_norm_ptr"));
    assert!(src.contains("attn_O_norm_stride0: tl.constexpr"));
    assert!(src.contains("tl.store(attn_O_norm_ptr"));
    assert!(src.contains("tl.load(attn_O_norm_ptr"));
    assert!(src.contains("range(0, 14336, BLOCK_P)"));
}

#[test]
fn prior_kernel_value_is_loaded_before_in_place_update() {
    let ir = "(seq (ploop 0 8 4 i (store (tensor T) (load (input A) (index (tile i))) (index (tile i)))) (ploop 0 8 4 i (seq (store (tensor T) (+ (load (tensor T) (index (tile i))) 1) (index (tile i))) (store (output O) (load (tensor T) (index (tile i))) (index (tile i))))))";
    let p = lower(
        analyze_text(ir).unwrap(),
        options(&[("A", &[8]), ("T", &[8]), ("O", &[8])]),
    )
    .unwrap();
    let tid = p.analysis().tensor_id("T").unwrap();
    assert!(p.kernels()[0].tensors[&tid].publish);
    assert_eq!(
        p.kernels()[1].tensors[&tid]
            .initialization
            .as_ref()
            .unwrap()
            .value,
        InitialValue::Global
    );
}

#[test]
fn rejects_undefined_non_additive_reads_and_overlapping_global_writes() {
    let ir = "(ploop 0 8 4 i (store (output O) (/ (load (output O) (index (tile i))) 2) (index (tile i))))";
    assert!(
        compile(ir, options(&[("O", &[8])]))
            .unwrap_err()
            .to_string()
            .contains("first read")
    );
    let ir = "(ploop 0 8 4 i (store (output O) i (index fulltile)))";
    assert!(
        compile(ir, options(&[("O", &[8])]))
            .unwrap_err()
            .to_string()
            .contains("disjoint")
    );
}

#[test]
fn keyed_views_use_layout_order_and_access_specific_shape() {
    let ir = "(ploop 0 6 4 k (store (view (output O) (layout (axis m 2) (axis n 6))) (load (view (input A) (layout (axis m 2) (axis n 6))) (keyed_index (slot n (tile k)))) (keyed_index (slot n (tile k)) (slot m fulltile))))";
    let p = lower(
        analyze_text(ir).unwrap(),
        options(&[("A", &[12]), ("O", &[2, 6])]),
    )
    .unwrap();
    assert_eq!(p.accesses()[0].shape, vec![2, 4]);
    assert_eq!(p.accesses()[0].axes[0].stride, 6);
    assert!(p.emit().contains("< 6"));
    let bad = ir.replace("(slot m fulltile)", "(slot unknown fulltile)");
    assert!(analyze_text(&bad).is_err());
    assert!(compile(ir, options(&[("A", &[13]), ("O", &[2, 6])])).is_err());
}

#[test]
fn incomplete_cross_kernel_regions_are_rejected() {
    let ir = "(seq (sloop 0 4 4 k (store (tensor T) 1 (index (tile k)))) (store (output O) (load (tensor T) (index fulltile)) (index fulltile)))";
    assert!(
        compile(ir, options(&[("T", &[8]), ("O", &[8])]))
            .unwrap_err()
            .to_string()
            .contains("not covered")
    );
}

#[test]
fn no_program_election_is_added_to_the_original_loop_schedule() {
    let ir = "(ploop 0 8 4 n (store (output O) (rsum (load (input A) (index fulltile)) 0) (index fulltile)))";
    let p = lower(
        analyze_text(ir).unwrap(),
        options(&[("A", &[7]), ("O", &[1])]),
    )
    .unwrap();
    assert!(!p.emit().contains("if tl.program_id"));
    let ir = ir.replace(
        "(rsum (load (input A) (index fulltile)) 0)",
        "(+ (rsum (load (input A) (index fulltile)) 0) n)",
    );
    assert!(compile(&ir, options(&[("A", &[7]), ("O", &[1])])).is_err());
}

#[test]
fn output_matches_the_original_backend_reference() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "trinity-source-reference-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    let mut files = Vec::new();
    for (corpus, number, ffn) in [
        (include_str!("fixtures/analyzer/ffn_cases.txt"), 177, true),
        (
            include_str!("fixtures/analyzer/vanilla_cases.txt"),
            591,
            false,
        ),
    ] {
        let line = corpus
            .lines()
            .find(|line| line.starts_with(&format!("{number}:")))
            .unwrap();
        let source = compile(line.split_once(':').unwrap().1, corpus_options(ffn)).unwrap();
        let file = directory.join(format!("{number}.py"));
        std::fs::write(&file, source).unwrap();
        files.push(file);
    }
    let shape_text = include_str!("fixtures/triton_reference/vanilla_falcon.shapes");
    let mut options = corpus_options(false);
    options.shapes = shape_text
        .lines()
        .map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next().unwrap().to_owned();
            (name, words.map(|n| n.parse().unwrap()).collect())
        })
        .collect();
    let source = compile(
        include_str!("fixtures/triton_reference/vanilla_falcon_591.ir"),
        options,
    )
    .unwrap();
    assert!(
        !source.contains("tl.where"),
        "zero-filled GEMM tails need no extra selects"
    );
    let file = directory.join("vanilla_falcon_591.py");
    std::fs::write(&file, source).unwrap();
    files.push(file);
    let result = std::process::Command::new("python3")
        .arg(root.join("tests/compare_triton_reference.py"))
        .arg("--sources")
        .args(&files)
        .output()
        .unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
