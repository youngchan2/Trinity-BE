use std::collections::BTreeMap;

use trinity_lowering::{
    Action, CommunicationKind, CommunicationOperation, ComputeOperation, CudaTargetCapability,
    DType, FusionError, FusionRewrite, FusionRule, OperationId, OperationPayload,
    PhysicalInvariantError, PhysicalPlan, PhysicalPlanBuilder, Storage, TargetCapability,
    ValueInstanceId, all_gather_implementations, fuse, fusion_rules, gemm_implementations,
};

const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Hopper);

struct Graph {
    builder: PhysicalPlanBuilder,
    shapes: BTreeMap<ValueInstanceId, [usize; 2]>,
    world: usize,
}

impl Graph {
    fn new(world: usize) -> Self {
        Self {
            builder: PhysicalPlanBuilder::new(TARGET, world),
            shapes: BTreeMap::new(),
            world,
        }
    }

    fn value(&mut self, shape: [usize; 2], storage: Storage) -> ValueInstanceId {
        let id = self.builder.add_value(DType::Bf16, shape, storage);
        self.shapes.insert(id, shape);
        id
    }

    fn input(&mut self, name: &str, shape: [usize; 2]) -> ValueInstanceId {
        let id = self.value(shape, Storage::External);
        self.builder.bind_input(name, id);
        id
    }

    fn gemm(
        &mut self,
        lhs: ValueInstanceId,
        rhs: ValueInstanceId,
        storage: Storage,
    ) -> ValueInstanceId {
        let [m, _] = self.shapes[&lhs];
        let [_, n] = self.shapes[&rhs];
        let shape = [m, n];
        let implementation = gemm_implementations(TARGET)[0]
            .enumerate(
                [DType::Bf16; 3],
                [&self.shapes[&lhs], &self.shapes[&rhs], &shape],
            )
            .pop()
            .unwrap();
        let result = self.value(shape, storage);
        let op = self.builder.add_operation(
            [lhs, rhs],
            [result],
            OperationPayload::Compute(ComputeOperation::new(implementation)),
        );
        self.builder.add_action([op]);
        result
    }

    fn gather(
        &mut self,
        source: ValueInstanceId,
        axis: usize,
        id: &str,
        storage: Storage,
    ) -> ValueInstanceId {
        let mut shape = self.shapes[&source];
        shape[axis] *= self.world;
        let implementation = all_gather_implementations(TARGET)
            .iter()
            .find(|definition| definition.id().as_str() == id)
            .unwrap()
            .enumerate(
                DType::Bf16,
                [&self.shapes[&source], &shape],
                axis,
                self.world,
            )
            .pop()
            .unwrap();
        let result = self.value(shape, storage);
        let op = self.builder.add_operation(
            [source],
            [result],
            OperationPayload::Communication(CommunicationOperation::new(
                CommunicationKind::AllGather,
                implementation,
            )),
        );
        self.builder.add_action([op]);
        result
    }

    fn finish(self, output: ValueInstanceId) -> PhysicalPlan {
        self.builder.finalize("Y", output).unwrap()
    }
}

fn chain(pull: bool, push: bool) -> PhysicalPlan {
    let mut g = Graph::new(if pull || push { 2 } else { 1 });
    let x = g.input("X", [128, 128]);
    let w0 = g.input("W0", [128, 128]);
    let w1 = g.input("W1", [128, 128]);
    let x = if pull {
        g.gather(x, 0, "nvshmem.peer_pull", Storage::Global)
    } else {
        x
    };
    let first = g.gemm(x, w0, Storage::Global);
    let second = g.gemm(
        first,
        w1,
        if push {
            Storage::Global
        } else {
            Storage::External
        },
    );
    let output = if push {
        g.gather(second, 1, "nvshmem.peer_push", Storage::External)
    } else {
        second
    };
    g.finish(output)
}

fn alternatives(plan: &PhysicalPlan) -> Vec<PhysicalPlan> {
    fuse(plan, fusion_rules(TARGET)).unwrap()
}

fn internal_storages(plan: &PhysicalPlan) -> Vec<Storage> {
    plan.value_instances()
        .filter_map(|(_, value)| (value.storage() != Storage::External).then_some(value.storage()))
        .collect()
}

#[test]
fn gemm_chain_has_shared_and_bf16_register_candidates_with_derived_boundaries() {
    let source = chain(false, false);
    let original_hash = source.hash();
    let plans = alternatives(&source);
    assert_eq!(plans.len(), 3);
    assert!(plans[0].same_body(&source));
    assert_eq!(source.hash(), original_hash);
    assert_eq!(source.actions().len(), 2);
    assert_eq!(internal_storages(&source), [Storage::Global]);
    for (plan, storage) in plans[1..].iter().zip([Storage::Shared, Storage::Register]) {
        assert_eq!(plan.actions().len(), 1);
        assert_eq!(plan.operations().len(), 2);
        assert_eq!(internal_storages(plan), [storage]);
        let action = plan.actions().next().unwrap().1;
        assert_eq!(action.inputs().len(), 3);
        assert_eq!(action.outputs(), &[plan.output().value()]);
        let intermediate = plan.operations().next().unwrap().1.outputs()[0];
        let value = plan.value_instance(intermediate).unwrap();
        assert_eq!(
            value.dtype(),
            DType::Bf16,
            "fusion must preserve the rounding boundary"
        );
        assert_eq!(value.shape(), [128, 128]);
        assert!(!action.inputs().contains(&intermediate));
        assert!(!action.outputs().contains(&intermediate));
        assert_eq!(plan.inputs(), source.inputs());
        assert_eq!(plan.output(), source.output());
        assert_ne!(plan.hash(), original_hash);
    }
    assert!(!plans[1].same_body(&plans[2]));
    assert_ne!(plans[1].hash(), plans[2].hash());
}

#[test]
fn producer_fusion_supports_both_shard_axes_but_nvls_and_pull_stay_standalone() {
    for axis in 0..2 {
        for backend in [
            "nvls.one_shot_push_nbi",
            "nvshmem.peer_push",
            "nvshmem.peer_pull",
        ] {
            let mut g = Graph::new(2);
            let x = g.input("X", [256, 128]);
            let w = g.input("W", [128, 256]);
            let product = g.gemm(x, w, Storage::Global);
            let output = g.gather(product, axis, backend, Storage::External);
            let source = g.finish(output);
            let plans = alternatives(&source);
            assert_eq!(
                plans.len(),
                if backend == "nvshmem.peer_push" { 3 } else { 1 }
            );
            for plan in &plans[1..] {
                assert_eq!(plan.actions().len(), 1);
                assert_eq!(
                    plan.value_instance(plan.output().value())
                        .unwrap()
                        .storage(),
                    Storage::External
                );
                assert_eq!(internal_storages(plan).len(), 1);
            }
        }
    }
}

#[test]
fn consumer_fusion_stages_either_operand_in_shared_on_both_shard_axes() {
    for operand in 0..2 {
        for axis in 0..2 {
            for backend in [
                "nvls.one_shot_push_nbi",
                "nvshmem.peer_push",
                "nvshmem.peer_pull",
            ] {
                let mut g = Graph::new(2);
                let source = g.input("Shard", [128, 128]);
                let gathered = g.gather(source, axis, backend, Storage::Global);
                let shape = g.shapes[&gathered];
                let output = if operand == 0 {
                    let rhs = g.input("W", [shape[1], 128]);
                    g.gemm(gathered, rhs, Storage::External)
                } else {
                    let lhs = g.input("X", [128, shape[0]]);
                    g.gemm(lhs, gathered, Storage::External)
                };
                let source = g.finish(output);
                let plans = alternatives(&source);
                assert_eq!(
                    plans.len(),
                    if backend == "nvshmem.peer_pull" { 2 } else { 1 }
                );
                for plan in &plans[1..] {
                    assert_eq!(internal_storages(plan), [Storage::Shared]);
                    assert!(plan.inputs().iter().all(|binding| {
                        plan.value_instance(binding.value()).unwrap().storage() == Storage::External
                    }));
                }
            }
        }
    }
}

#[test]
fn computation_fusion_rejects_rhs_handoff_and_cross_cta_tiles() {
    for (middle_n, final_n, rhs_handoff) in [(256, 128, false), (128, 256, false), (128, 128, true)]
    {
        let mut g = Graph::new(1);
        let x = g.input("X", [128, 128]);
        let w0 = g.input("W0", [128, middle_n]);
        let w1 = g.input("W1", [middle_n, final_n]);
        let middle = g.gemm(x, w0, Storage::Global);
        let output = if rhs_handoff {
            g.gemm(w1, middle, Storage::External)
        } else {
            g.gemm(middle, w1, Storage::External)
        };
        assert_eq!(alternatives(&g.finish(output)).len(), 1);
    }
}

#[test]
fn a_previously_fused_body_is_revalidated_before_extending_it() {
    let mut g = Graph::new(2);
    let x = g.input("X", [128, 128]);
    let w = g.input("W", [128, 128]);
    let first = g.gemm(x, w, Storage::Global);
    let second = g.gemm(w, first, Storage::Global); // unsupported RHS handoff
    let output = g.gather(second, 1, "nvshmem.peer_push", Storage::External);
    let original = g.finish(output);
    let custom_plans = fuse(&original, &[&PromoteBridge]).unwrap();
    let partial = custom_plans
        .iter()
        .find(|plan| {
            plan.actions().len() == 2
                && plan.actions().any(|(_, action)| {
                    action.operations().len() == 2
                        && action.operations().iter().all(|id| {
                            matches!(
                                plan.operation(*id).unwrap().payload(),
                                OperationPayload::Compute(_)
                            )
                        })
                })
        })
        .unwrap();
    // The new boundary alone is a supported GEMM -> push, but the old
    // Shared GEMM -> GEMM edge is not a supported compute body.
    assert_eq!(alternatives(partial).len(), 1);
}

#[test]
fn two_input_pulls_remain_separate_partial_fusion_choices() {
    let mut g = Graph::new(2);
    let a = g.input("A", [64, 128]);
    let b = g.input("B", [128, 64]);
    let a = g.gather(a, 0, "nvshmem.peer_pull", Storage::Global);
    let b = g.gather(b, 1, "nvshmem.peer_pull", Storage::Global);
    let output = g.gemm(a, b, Storage::External);
    let plans = alternatives(&g.finish(output));
    assert_eq!(plans.len(), 3);
    assert_eq!(
        plans
            .iter()
            .filter(|plan| plan.actions().len() == 2)
            .count(),
        2
    );
    assert!(plans.iter().all(|plan| plan.actions().len() >= 2));
}

#[test]
fn a_pull_keeps_its_computed_remote_source_in_global_memory() {
    let mut g = Graph::new(2);
    let x = g.input("X", [128, 128]);
    let w = g.input("W", [128, 128]);
    let source = g.gemm(x, w, Storage::Global);
    let pulled = g.gather(source, 0, "nvshmem.peer_pull", Storage::Global);
    let output = g.gemm(pulled, w, Storage::External);
    let plans = alternatives(&g.finish(output));
    assert_eq!(plans.len(), 2);
    let fused = &plans[1];
    assert_eq!(fused.actions().len(), 2);
    let source = fused.operations().next().unwrap().1.outputs()[0];
    assert_eq!(
        fused.value_instance(source).unwrap().storage(),
        Storage::Global
    );
    assert_eq!(internal_storages(fused), [Storage::Global, Storage::Shared]);
}

#[test]
fn push_fusion_rejects_a_destination_in_cta_local_memory() {
    let mut g = Graph::new(2);
    let x = g.input("X", [128, 128]);
    let w = g.input("W", [128, 128]);
    let product = g.gemm(x, w, Storage::Global);
    g.gather(product, 1, "nvshmem.peer_push", Storage::Shared);
    // Physical finalization allows an unused local result; backend matching
    // must still reject it as a peer-write destination.
    let plan = g.finish(x);
    assert_eq!(alternatives(&plan).len(), 1);
}

#[test]
fn enumerates_all_partial_and_complete_chains_independently_of_rule_order() {
    let source = chain(true, true);
    let plans = alternatives(&source);
    // Each of the three edges is either unfused or promoted: 2 * 3 * 3.
    assert_eq!(plans.len(), 18);
    let mut counts = BTreeMap::new();
    for plan in &plans {
        *counts.entry(plan.actions().len()).or_insert(0usize) += 1;
    }
    assert_eq!(counts, BTreeMap::from([(1, 4), (2, 8), (3, 5), (4, 1)]));
    for plan in plans.iter().filter(|plan| plan.actions().len() == 1) {
        assert_eq!(plan.actions().next().unwrap().1.operations().len(), 4);
        assert_eq!(internal_storages(plan)[0], Storage::Shared);
        assert_eq!(alternatives(plan).len(), 1);
    }
    let reversed_rules = fusion_rules(TARGET)
        .iter()
        .copied()
        .rev()
        .collect::<Vec<_>>();
    let reordered = fuse(&source, &reversed_rules).unwrap();
    assert_same_candidates(&plans, &reordered);
    assert_same_candidates(&plans, &alternatives(&reverse_insertion(&source)));
    let repeated = alternatives(&source);
    assert!(
        plans
            .iter()
            .zip(&repeated)
            .all(|(a, b)| a.same_body(b) && a.hash() == b.hash())
    );
}

fn assert_same_candidates(lhs: &[PhysicalPlan], rhs: &[PhysicalPlan]) {
    assert_eq!(lhs.len(), rhs.len());
    assert!(
        lhs.iter()
            .all(|a| rhs.iter().any(|b| a.same_body(b) && a.hash() == b.hash()))
    );
}

fn reverse_insertion(plan: &PhysicalPlan) -> PhysicalPlan {
    let mut builder = PhysicalPlanBuilder::new(plan.target(), plan.world_size());
    let mut values = BTreeMap::new();
    for (id, value) in plan.value_instances().collect::<Vec<_>>().into_iter().rev() {
        values.insert(
            id,
            builder.add_value(
                value.dtype(),
                value.shape().iter().copied(),
                value.storage(),
            ),
        );
    }
    for binding in plan.inputs().iter().rev() {
        builder.bind_input(binding.tensor(), values[&binding.value()]);
    }
    let mut operations = BTreeMap::new();
    for (id, operation) in plan.operations().collect::<Vec<_>>().into_iter().rev() {
        operations.insert(
            id,
            builder.add_operation(
                operation.inputs().iter().map(|id| values[id]),
                operation.outputs().iter().map(|id| values[id]),
                operation.payload().clone(),
            ),
        );
    }
    for (_, action) in plan.actions().collect::<Vec<_>>().into_iter().rev() {
        builder.add_action(action.operations().iter().rev().map(|id| operations[id]));
    }
    builder
        .finalize(plan.output().tensor(), values[&plan.output().value()])
        .unwrap()
}

#[test]
fn shared_transition_fanout_and_abi_intermediates_prevent_promotion() {
    // A shared AllGather result consumed by two GEMMs must stay in Global.
    let mut g = Graph::new(2);
    let x = g.input("X", [128, 128]);
    let w0 = g.input("W0", [128, 128]);
    let w1 = g.input("W1", [128, 128]);
    let gathered = g.gather(x, 0, "nvshmem.peer_pull", Storage::Global);
    g.gemm(gathered, w0, Storage::Global);
    let output = g.gemm(gathered, w1, Storage::External);
    let plan = g.finish(output);
    assert_eq!(alternatives(&plan).len(), 1);
    assert!(matches!(
        fuse(&plan, &[&PromoteBridge]),
        Err(FusionError::InvalidRewrite { .. })
    ));

    // The ABI output can still be consumed internally, but cannot be promoted.
    let mut g = Graph::new(1);
    let x = g.input("X", [128, 128]);
    let w = g.input("W", [128, 128]);
    let output = g.gemm(x, w, Storage::External);
    g.gemm(output, w, Storage::Global);
    let plan = g.finish(output);
    assert_eq!(alternatives(&plan).len(), 1);
    assert!(matches!(
        fuse(&plan, &[&PromoteBridge]),
        Err(FusionError::InvalidRewrite { .. })
    ));
}

struct PromoteBridge;
impl FusionRule for PromoteBridge {
    fn apply(
        &self,
        _: &PhysicalPlan,
        producer: &Action,
        consumer: &Action,
    ) -> Result<Vec<FusionRewrite>, FusionError> {
        let bridge = *producer
            .outputs()
            .iter()
            .find(|id| consumer.inputs().contains(id))
            .unwrap();
        Ok(vec![FusionRewrite::new(
            producer
                .operations()
                .iter()
                .chain(consumer.operations())
                .copied(),
            [(bridge, Storage::Shared)],
        )])
    }
}

struct Merge;
impl FusionRule for Merge {
    fn apply(
        &self,
        _: &PhysicalPlan,
        producer: &Action,
        consumer: &Action,
    ) -> Result<Vec<FusionRewrite>, FusionError> {
        let operations = producer
            .operations()
            .iter()
            .chain(consumer.operations())
            .copied()
            .collect::<Vec<_>>();
        // Deliberate duplicate proposals must not duplicate candidates.
        Ok(vec![
            FusionRewrite::new(operations.iter().copied(), []),
            FusionRewrite::new(operations.into_iter().rev(), []),
        ])
    }
}

#[test]
fn contraction_skips_a_transitive_edge_before_invoking_rules() {
    let mut g = Graph::new(1);
    let x = g.input("X", [128, 128]);
    let w = g.input("W", [128, 128]);
    let first = g.gemm(x, w, Storage::Global);
    let middle = g.gemm(first, w, Storage::Global);
    let output = g.gemm(first, middle, Storage::External);
    let plan = g.finish(output);
    let ids = plan.operations().map(|(id, _)| id).collect::<Vec<_>>();
    struct AssertPair(OperationId, OperationId);
    impl FusionRule for AssertPair {
        fn apply(
            &self,
            plan: &PhysicalPlan,
            p: &Action,
            c: &Action,
        ) -> Result<Vec<FusionRewrite>, FusionError> {
            assert!(
                p.operations() != [self.0] || c.operations() != [self.1],
                "cycle-producing pair reached rule"
            );
            Merge.apply(plan, p, c)
        }
    }
    let plans = fuse(&plan, &[&AssertPair(ids[0], ids[2])]).unwrap();
    assert_eq!(plans.len(), 4); // original, two convex pairs, complete body
    assert_eq!(
        plans
            .iter()
            .filter(|plan| plan.actions().len() == 1)
            .count(),
        1
    );
    assert!(fuse(&plan, &[]).unwrap()[0].same_body(&plan));
}

#[test]
fn invalid_custom_rewrites_report_errors_without_changing_the_source() {
    let plan = chain(false, false);
    let before = plan.clone();
    struct Bad(u8);
    impl FusionRule for Bad {
        fn apply(
            &self,
            plan: &PhysicalPlan,
            p: &Action,
            c: &Action,
        ) -> Result<Vec<FusionRewrite>, FusionError> {
            let mut operations = p
                .operations()
                .iter()
                .chain(c.operations())
                .copied()
                .collect::<Vec<_>>();
            let bridge = p.outputs()[0];
            let updates = match self.0 {
                0 => {
                    operations.pop();
                    vec![]
                }
                1 => {
                    operations.push(operations[0]);
                    vec![]
                }
                2 => vec![(bridge, Storage::Shared), (bridge, Storage::Register)],
                3 => vec![(bridge, Storage::External)],
                4 => vec![(plan.inputs()[0].value(), Storage::Shared)],
                5 => vec![(bridge, Storage::Global)],
                _ => {
                    let mut foreign = PhysicalPlanBuilder::new(TARGET, 1);
                    let id = (0..32)
                        .map(|_| foreign.add_value(DType::Bf16, [128, 128], Storage::Global))
                        .last()
                        .unwrap();
                    vec![(id, Storage::Shared)]
                }
            };
            Ok(vec![FusionRewrite::new(operations, updates)])
        }
    }
    for case in 0..7 {
        assert!(matches!(
            fuse(&plan, &[&Bad(case)]),
            Err(FusionError::InvalidRewrite { .. })
        ));
        assert!(plan.same_body(&before));
        assert_eq!(plan.hash(), before.hash());
    }
}

#[test]
fn finalization_errors_are_not_hidden_as_nonmatches() {
    let mut g = Graph::new(1);
    let x = g.input("X", [128, 128]);
    let w = g.input("W", [128, 128]);
    // Initially distinct physical specs. The custom rule below creates a
    // duplicate operation by changing the second result to Shared as well.
    g.gemm(x, w, Storage::Shared);
    let second = g.gemm(x, w, Storage::Global);
    let output = g.gemm(second, w, Storage::External);
    let plan = g.finish(output);
    assert!(matches!(
        fuse(&plan, &[&PromoteBridge]),
        Err(FusionError::InvalidPlan(
            PhysicalInvariantError::DuplicateOperation { .. }
        ))
    ));
}
