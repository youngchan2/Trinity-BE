use std::collections::BTreeSet;

use trinity_lowering::analyzer::{
    AccessKind, IndexDim, IndexExpr, IrNode, ProgramAnalysis, ScopeKind, TensorKind, analyze,
    analyze_text,
};

const FFN: &str = include_str!("fixtures/analyzer/ffn_cases.txt");
const VANILLA: &str = include_str!("fixtures/analyzer/vanilla_cases.txt");

fn fixture(corpus: &str, id: usize) -> ProgramAnalysis {
    let expression = corpus
        .lines()
        .find_map(|line| {
            let (key, expression) = line.split_once(':')?;
            (key.parse::<usize>().ok()? == id).then_some(expression.trim())
        })
        .unwrap();
    analyze_text(expression).unwrap()
}

#[test]
fn ffn_failures_have_one_real_scope_for_normalization() {
    for (id, kernel_count) in [(177, 3), (307, 4), (1019, 2), (1230, 3)] {
        let result = fixture(FFN, id);
        assert_eq!(result.kernels().len(), kernel_count);
        let tensor = result.tensor_id("attn_O_norm").unwrap();
        let reads = result.access_scopes(tensor, AccessKind::Read);
        let writes = result.access_scopes(tensor, AccessKind::Write);
        assert_eq!(
            reads, writes,
            "IR {id}: all consumers are in the defining loop"
        );
        assert_eq!(
            reads.len(),
            1,
            "IR {id}: parent p must not duplicate k's usage"
        );
        let k = *reads.first().unwrap();
        let k_info = result.scope(k);
        assert_eq!(k_info.loop_info.as_ref().unwrap().variable, "k");
        let p = k_info.parent.unwrap();
        assert_eq!(result.scope(p).loop_info.as_ref().unwrap().variable, "p");
        assert!(!result.scope(p).read_writes.reads.contains(&tensor));
        assert!(!result.scope(p).read_writes.writes.contains(&tensor));
        assert!(result.is_within(k, p));
        let uses: Vec<_> = k_info
            .accesses
            .iter()
            .map(|a| result.access(*a))
            .filter(|a| a.tensor == tensor)
            .collect();
        assert_eq!(uses.len(), 3);
        assert_eq!(uses[0].kind, AccessKind::Write);
        assert!(uses[1..].iter().all(|a| a.kind == AccessKind::Read));
    }
}

#[test]
fn sibling_k_loops_keep_distinct_bindings_even_with_identical_text() {
    let result = fixture(FFN, 4);
    let tensor = result.tensor_id("attn_O_norm").unwrap();
    let reads = result.access_scopes(tensor, AccessKind::Read);
    let writes = result.access_scopes(tensor, AccessKind::Write);
    assert_eq!(reads.len(), 1);
    assert_eq!(writes.len(), 1);
    let read_scope = *reads.first().unwrap();
    let write_scope = *writes.first().unwrap();
    assert_ne!(read_scope, write_scope);
    assert_eq!(
        result.scope(read_scope).parent,
        result.scope(write_scope).parent
    );
    for id in [read_scope, write_scope] {
        assert_eq!(result.scope(id).loop_info.as_ref().unwrap().variable, "k");
        let access = result
            .scope(id)
            .accesses
            .iter()
            .map(|id| result.access(*id))
            .find(|a| a.tensor == tensor)
            .unwrap();
        assert!(
            matches!(&access.index[1], IndexDim::Tile { start: IndexExpr::LoopVar(bound), .. } if *bound == id)
        );
    }
}

#[test]
fn grouped_projection_keeps_each_self_load_and_weight_paired() {
    let result = fixture(VANILLA, 591);
    for (projection, weight) in [("Q1", "WQ"), ("K1", "WK"), ("V1", "WV")] {
        let tensor = result.tensor_id(projection).unwrap();
        let write = result
            .accesses()
            .iter()
            .find(|a| a.tensor == tensor && a.kind == AccessKind::Write)
            .unwrap();
        let statement = result.statement(write.statement);
        let accesses: Vec<_> = statement
            .accesses
            .iter()
            .map(|id| result.access(*id))
            .collect();
        assert_eq!(accesses.last().unwrap().kind, AccessKind::Write);
        let read_names: BTreeSet<_> = accesses[..accesses.len() - 1]
            .iter()
            .map(|a| result.tensor(a.tensor).name.as_str())
            .collect();
        assert_eq!(read_names, BTreeSet::from([projection, "X", weight]));
    }
    let mutated: BTreeSet<_> = result
        .mutated_inputs()
        .into_iter()
        .map(|id| result.tensor(id).name.as_str())
        .collect();
    assert_eq!(mutated, BTreeSet::from(["K_cache", "V_cache"]));
    assert!(
        result
            .declared_tensors(TensorKind::Output)
            .contains(&result.tensor_id("O2").unwrap())
    );
    let o = result.tensor_id("O").unwrap();
    let writes: Vec<_> = result
        .accesses()
        .iter()
        .filter(|a| a.tensor == o && a.kind == AccessKind::Write)
        .collect();
    assert_eq!(
        writes.len(),
        2,
        "retain both accumulation and epilogue definitions"
    );
    assert_ne!(writes[0].statement, writes[1].statement);
    assert_ne!(writes[0].scope, writes[1].scope);
}

#[test]
fn standalone_sloop_remains_a_separate_kernel() {
    let result = fixture(FFN, 244);
    assert_eq!(result.kernels().len(), 4);
    let root = result.kernels()[2].root_scope;
    assert_eq!(result.scope(root).children.len(), 1);
    let trinity_lowering::analyzer::ScopeItem::Scope(loop_id) = result.scope(root).children[0]
    else {
        panic!()
    };
    assert_eq!(result.scope(loop_id).kind, ScopeKind::SequentialLoop);
}

#[test]
fn explicit_access_width_does_not_inherit_loop_step() {
    let result = analyze_text("(ploop 0 32 8 i (store (output O) (load (input A) (index (tile (* i 2) 4))) (index (const_tile 3 4))))").unwrap();
    let read = &result.accesses()[0];
    let IndexDim::Tile { start, width } = &read.index[0] else {
        panic!()
    };
    assert_eq!(width, &IndexExpr::Integer(4));
    assert_eq!(start.loop_dependencies(), BTreeSet::from([read.scope]));
    assert!(matches!(
        result.accesses()[1].index[0],
        IndexDim::ConstTile {
            start: IndexExpr::Integer(3),
            width: IndexExpr::Integer(4)
        }
    ));
}

#[test]
fn loop_shadowing_resolves_to_the_nearest_binding_then_restores_parent() {
    let result = analyze_text("(ploop 0 32 8 i (seq (sloop 0 16 4 i (store (tensor T) 1 (index (tile i)))) (store (output O) 2 (index (tile i)))))").unwrap();
    let inner = &result.accesses()[0];
    let outer = &result.accesses()[1];
    assert_ne!(inner.scope, outer.scope);
    assert_eq!(
        inner.index[0],
        IndexDim::Tile {
            start: IndexExpr::LoopVar(inner.scope),
            width: IndexExpr::Integer(4)
        }
    );
    assert_eq!(
        outer.index[0],
        IndexDim::Tile {
            start: IndexExpr::LoopVar(outer.scope),
            width: IndexExpr::Integer(8)
        }
    );
}

#[test]
fn preserves_read_before_write_and_kernel_local_tensor_summaries() {
    let result = analyze_text("(seq (ploop 0 16 16 i (store (tensor T) 1 (index (tile i)))) (ploop 0 16 16 i (store (tensor T) (+ (load (tensor T) (index (tile i))) 1) (index (tile i)))))").unwrap();
    assert_eq!(result.kernels().len(), 2);
    let tensor = result.tensor_id("T").unwrap();
    assert!(result.kernels()[0].read_writes.reads.is_empty());
    assert_eq!(
        result.kernels()[1].read_writes.reads,
        BTreeSet::from([tensor])
    );
    assert_eq!(result.kernels()[1].accesses.len(), 2);
    let first = result.access(result.kernels()[1].accesses[0]);
    let second = result.access(result.kernels()[1].accesses[1]);
    assert_eq!(first.kind, AccessKind::Read);
    assert_eq!(second.kind, AccessKind::Write);
}

#[test]
fn supports_direct_rust_input_and_retains_source_spans_for_text_input() {
    let node = IrNode::call(
        "store",
        [
            IrNode::call("output", [IrNode::atom("O")]),
            IrNode::atom("1.5"),
            IrNode::call("index", [IrNode::atom("fulltile")]),
        ],
    );
    let native = analyze(node.clone()).unwrap();
    assert_eq!(native.ir(), &node);
    assert_eq!(native.accesses()[0].source_span, None);
    let source = "(store (output O) (load (input A) fulltile) (index fulltile))";
    let result = analyze_text(source).unwrap();
    let span = result.accesses()[0].source_span.unwrap();
    assert_eq!(&source[span.start..span.end], "(load (input A) fulltile)");
}

#[test]
fn rejects_incomplete_ir_instead_of_returning_partial_analysis() {
    for source in [
        "dummydata74",
        "(seq dummy dummydata82)",
        "(store (tensor T) 1)",
        "(store (tensor T) 1 (index (tile k)))",
        "dummy dummy",
        "(seq dummy",
        "()",
        "(store (tensor T) (load (input A,B) fulltile) (index fulltile))",
        "(ploop 0 16 0 i (store (tensor T) 1 (index fulltile)))",
        "(store (tensor T) 1 (index (tile 0 0)))",
    ] {
        assert!(
            analyze_text(source).is_err(),
            "unexpectedly accepted {source}"
        );
    }
    assert!(
        analyze_text("(seq dummy (sloop 0 16 4 k dummy))")
            .unwrap()
            .kernels()
            .is_empty()
    );
}

#[test]
fn all_pinned_corpus_cases_produce_consistent_access_summaries() {
    for line in FFN.lines().chain(VANILLA.lines()) {
        let (_, source) = line.split_once(':').unwrap();
        let result = analyze_text(source).unwrap();
        let total: usize = result.kernels().iter().map(|k| k.accesses.len()).sum();
        assert_eq!(total, result.accesses().len());
        let direct: usize = result.scopes().iter().map(|s| s.accesses.len()).sum();
        assert_eq!(
            direct, total,
            "every access has exactly one containing scope"
        );
        for kernel in result.kernels() {
            let reads: BTreeSet<_> = kernel
                .accesses
                .iter()
                .map(|a| result.access(*a))
                .filter(|a| a.kind == AccessKind::Read)
                .map(|a| a.tensor)
                .collect();
            assert_eq!(reads, kernel.read_writes.reads);
        }
    }
}
