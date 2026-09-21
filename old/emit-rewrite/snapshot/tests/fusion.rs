use std::collections::BTreeMap;

#[allow(dead_code)]
mod support;

use trinity_lowering::{
    CommunicationKind, CommunicationOperation, ComputeOperation, CudaTargetCapability, DType,
    OperationPayload, PhysicalPlan, PhysicalPlanBuilder, Storage, TargetCapability,
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
        self.builder
            .add_statement(trinity_lowering::Statement::Operation(op));
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
        self.builder
            .add_statement(trinity_lowering::Statement::Operation(op));
        result
    }

    fn finish(self, output: ValueInstanceId) -> PhysicalPlan {
        self.builder.finalize("Y", output).unwrap()
    }
}

fn chain(pull: bool, push: bool) -> PhysicalPlan {
    chain_world(pull, push, if pull || push { 2 } else { 1 })
}
fn chain_world(pull: bool, push: bool, world: usize) -> PhysicalPlan {
    let mut g = Graph::new(world);
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

#[test]
fn fusion_rebuilds_structured_statements_without_mutating_the_original() {
    for (pull, push) in [(false, false), (true, false), (false, true), (true, true)] {
        let plan = chain(pull, push);
        let hash = plan.hash();
        let candidates = fuse(&plan, fusion_rules(TARGET)).unwrap();
        assert!(candidates[0].same_body(&plan));
        assert!(candidates.len() >= 2);
        for candidate in &candidates {
            trinity_lowering::emit(candidate).unwrap();
        }
        assert_eq!(plan.hash(), hash);
    }
}

#[test]
fn wgmma_ss_rules_offer_only_supported_shared_handoffs() {
    let plan = chain(false, false);
    let statements = plan.statements();
    let rewrites: Vec<_> = fusion_rules(TARGET)
        .iter()
        .flat_map(|rule| rule.apply(&plan, &statements[0], &statements[1]).unwrap())
        .collect();
    assert_eq!(rewrites.len(), 1);
    assert!(rewrites.iter().all(|r| r.operations().len() == 2));
    let storages: Vec<_> = rewrites
        .iter()
        .flat_map(|r| r.storage_updates().iter().map(|(_, s)| *s))
        .collect();
    assert!(storages.contains(&Storage::Shared));
    assert!(!storages.contains(&Storage::Register));
}

#[test]
fn gemm_handoff_keeps_serial_scopes_and_rebuilds_task_readiness() {
    let plan = chain_world(false, false, 2);
    let candidates = fuse(&plan, fusion_rules(TARGET)).unwrap();
    let fused = candidates.last().unwrap();
    assert_eq!(fused.statements().len(), 1);
    let source = trinity_lowering::emit(fused).unwrap();
    let e = source.execution().as_persistent().unwrap();
    assert_eq!(e.tasks_per_rank, 1);
    assert_eq!(e.schedule[0].stages.len(), 4);
    assert_eq!(source.requirements().buffers.len(), 4);
    assert_eq!(
        source.requirements().shared_memory_bytes,
        65536 + 128 * 128 * 2
    );
    assert_eq!(source.code().matches("warpgroup_wait<0>").count(), 2);
    assert_eq!(source.code().matches("clear(accumulator_").count(), 2);
    assert!(!source.code().contains("${"));
    let internal = fused
        .value_instances()
        .find(|(_, v)| v.storage() == Storage::Shared)
        .unwrap()
        .0;
    assert!(e.work.iter().all(|w| {
        w.reads
            .iter()
            .chain(&w.writes)
            .chain(w.stages.iter().flatten())
            .all(|r| r.value != internal)
    }));
    let phase = source.bodies()[0].mainloop.as_ref().unwrap();
    let writer = phase.outputs.iter().find(|b| b.value == internal).unwrap();
    let reader = phase.inputs.iter().find(|b| b.value == internal).unwrap();
    assert_eq!(writer.symbol, reader.symbol);
}

fn pointwise_chain(world: usize) -> PhysicalPlan {
    let mut b = support::tensor::Builder::new(world);
    let x = b.input("X", DType::Fp32, &[3, 129]);
    let h = b.pointwise("mul", &[x, x], DType::Fp32, None, Storage::Global);
    let h = b.pointwise("add", &[h, x], DType::Bf16, None, Storage::Global);
    let y = b.pointwise("relu", &[h], DType::Fp32, None, Storage::External);
    b.finish(y)
}

#[test]
fn pointwise_chain_preserves_rounding_and_shares_one_body_with_the_tail() {
    let plan = pointwise_chain(1);
    let candidates = fuse(&plan, fusion_rules(TARGET)).unwrap();
    let fused = candidates
        .iter()
        .find(|p| p.statements().len() == 1)
        .unwrap();
    let source = trinity_lowering::emit(fused).unwrap();
    assert_eq!(source.execution().as_streamed().unwrap().launches.len(), 1);
    assert_eq!(source.execution().as_streamed().unwrap().tasks.len(), 1);
    assert_eq!(source.execution().as_streamed().unwrap().grids[0].blocks, 6);
    assert_eq!(source.bodies().len(), 1);
    assert_eq!(source.requirements().buffers.len(), 2);
    assert!(source.bodies().iter().all(|b| b.mainloop.is_none()));
    assert!(source.code().contains("__fmul_rn("));
    assert!(source.code().contains("=cutlass::bfloat16_t("));
    assert!(!source.code().contains("${"));
}

impl Graph {
    fn relu(&mut self, input: ValueInstanceId, dtype: DType, storage: Storage) -> ValueInstanceId {
        let shape = self.shapes[&input];
        let output = self.builder.add_value(dtype, shape, storage);
        self.shapes.insert(output, shape);
        let implementation = trinity_lowering::pointwise_implementations(TARGET)
            .iter()
            .find(|d| d.id().as_str() == "cuda.relu")
            .unwrap()
            .enumerate(&[DType::Bf16, dtype], &[&shape, &shape], None)
            .pop()
            .unwrap();
        let op = self.builder.add_operation(
            [input],
            [output],
            OperationPayload::Compute(ComputeOperation::new(implementation)),
        );
        self.builder
            .add_statement(trinity_lowering::Statement::Operation(op));
        output
    }
}

fn gemm_relu(world: usize, fanout: bool) -> PhysicalPlan {
    let mut g = Graph::new(world);
    let x = g.input("X", [256, 128]);
    let w = g.input("W", [128, 128]);
    let h = g.gemm(x, w, Storage::Global);
    let y = g.relu(
        h,
        DType::Fp32,
        if fanout {
            Storage::Global
        } else {
            Storage::External
        },
    );
    if fanout {
        let y = g.relu(h, DType::Bf16, Storage::External);
        g.finish(y)
    } else {
        g.finish(y)
    }
}

#[test]
fn gemm_relu_substitutes_fragment_binding_after_dtype_conversion() {
    for world in [1, 2] {
        let plan = gemm_relu(world, false);
        let candidates = fuse(&plan, fusion_rules(TARGET)).unwrap();
        assert_eq!(candidates.len(), 2);
        let fused = &candidates[1];
        let source = trinity_lowering::emit(fused).unwrap();
        match source.execution() {
            trinity_lowering::emit::Execution::Streamed(e) => {
                assert_eq!(e.tasks.len(), 1);
                assert_eq!(e.grids[0].blocks, 2);
                assert_eq!(e.launches.len(), 1);
            }
            trinity_lowering::emit::Execution::Persistent(e) => {
                assert_eq!(e.tasks_per_rank, 2);
            }
        }
        assert_eq!(source.bodies().len(), 1);
        assert!(source.bodies()[0].mainloop.is_some());
        assert_eq!(source.requirements().buffers.len(), 3);
        let internal = fused
            .value_instances()
            .find(|(_, v)| v.storage() == Storage::Register)
            .unwrap()
            .0;
        let epilogue = &source.bodies()[0].epilogue;
        let output = epilogue
            .outputs
            .iter()
            .find(|b| b.value == internal)
            .unwrap();
        let input = epilogue
            .inputs
            .iter()
            .find(|b| b.value == internal)
            .unwrap();
        assert_eq!(output.symbol, input.symbol);
        let code = source.code();
        assert!(code.contains("=cutlass::bfloat16_t(accumulator_"));
        assert!(code.contains("float(result_"));
        assert!(!source.bodies()[0].render().unwrap().contains("input_"));
        assert!(!code.contains("${"));
        if let Some(e) = source.execution().as_persistent() {
            assert_eq!(e.output_dependencies.len(), world);
            assert!(e.output_dependencies.iter().all(|d| d.len() == 2));
        }
    }
}

#[test]
fn externally_observed_intermediate_is_not_promoted() {
    let plan = gemm_relu(1, true);
    let candidates = fuse(&plan, fusion_rules(TARGET)).unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].same_body(&plan));
}

#[test]
#[ignore = "requires NVCC and NVSHMEM, never executes GPU work"]
fn fused_sources_compile_and_link() {
    for plan in [
        chain(false, false),
        chain(true, true),
        gemm_relu(1, false),
        gemm_relu(2, false),
        pointwise_chain(1),
    ] {
        let candidates = fuse(&plan, fusion_rules(TARGET)).unwrap();
        let source = trinity_lowering::emit(candidates.last().unwrap()).unwrap();
        let artifact = trinity_lowering::compile(source).unwrap_or_else(|e| panic!("{e}"));
        assert!(artifact.artifact_path().is_file());
    }
}
