use super::execution::{Domain, Kernel, MAX_GRID_X, coordinates, launches};
use crate::{IndexExpr, LoopDomain};

#[test]
fn grid_decode_and_chunking_preserve_nonzero_coordinates() {
    let d = |name: &str, start, stop, step| {
        Domain::new(&LoopDomain {
            variable: name.into(),
            start: IndexExpr::Constant(start),
            stop: IndexExpr::Constant(stop),
            step: IndexExpr::Constant(step),
        })
        .unwrap()
    };
    let kernel = Kernel {
        domain: vec![d("r", 2, 6, 2), d("c", 3, 10, 3)],
        body: vec![],
        blocks: 6,
        threads: 128,
        shared: 0,
        alignment: 1,
    };
    let decoded: Vec<_> = (0..6)
        .map(|block| {
            let c = coordinates(&kernel, block);
            (c["r"], c["c"])
        })
        .collect();
    assert_eq!(decoded, [(2, 3), (2, 6), (2, 9), (4, 3), (4, 6), (4, 9)]);
    let chunks = launches(2, MAX_GRID_X + 3);
    assert_eq!(chunks.len(), 2);
    assert_eq!(
        (chunks[1].kernel, chunks[1].base, chunks[1].blocks),
        (2, MAX_GRID_X, 3)
    );
}

#[test]
fn existing_register_pipelines_emit_without_local_launch_slots() {
    for plan in crate::emit::native::combine::tests::pipelines() {
        let source = crate::emit(&plan).unwrap();
        assert_eq!(
            source.requirements().buffers.len(),
            plan.value_instances()
                .filter(|(_, v)| matches!(
                    v.storage(),
                    crate::Storage::External | crate::Storage::Global
                ))
                .count()
        );
        assert!(source.code().contains("body_0"));
    }
}
