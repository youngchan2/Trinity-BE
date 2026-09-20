#[allow(dead_code)]
mod support;
use std::collections::BTreeSet;
use trinity_lowering::{
    AsStr, CudaTargetCapability, OperationPayload, TargetCapability, emit, fuse, fusion_rules,
};

#[test]
fn public_api_source_is_deterministic_and_preserves_binding_names() {
    let plan = support::gemm(256, 256, 192, 1);
    let a = emit(&plan).unwrap();
    let b = emit(&plan).unwrap();
    assert_eq!(a.code(), b.code());
    let req = a.requirements();
    assert_eq!(req.target, CudaTargetCapability::Hopper);
    assert_eq!(req.target.as_str(), "sm_90a");
    assert_eq!(req.target.to_string(), "hopper");
    assert_eq!(plan.target(), TargetCapability::Cuda(req.target));
    assert_eq!(req.shared_memory_bytes, 65536);
    assert_eq!(req.block_threads, 128);
    assert!(!req.nvshmem);
    assert!(!req.cooperative_launch);
    assert_eq!(a.execution().as_streamed().unwrap().grids[0].blocks, 4);
    assert_eq!(a.execution().as_streamed().unwrap().tasks.len(), 1);
    assert_eq!(a.bodies().len(), 1);
    assert!(
        req.buffers
            .iter()
            .any(|b| b.input_names.iter().any(|s| s == "X with spaces\""))
    );

    // Names occur only inside the escaped ABI metadata string, never as C++ identifiers.
    let metadata = serde_json::to_string(&serde_json::to_string(req).unwrap()).unwrap();
    assert!(a.code().contains(&format!("metadata[] = {metadata}")));
    assert!(!a.code().contains("#include <nvshmem.h>"));
    for buffer in &req.buffers {
        assert_eq!(buffer.bytes, buffer.shape.iter().product::<usize>() * 2);
        assert_eq!(buffer.alignment, 16);
        assert_eq!(buffer.dtype, trinity_lowering::DType::Bf16);
        assert_eq!(buffer.strides, [buffer.shape[1], 1]);
    }
    assert_eq!(req.workspace_bytes, 0);
    assert_eq!(req.workspace_alignment, 1);
    assert!(!req.workspace_symmetric);
    assert_eq!(a.execution().workspace_bytes(), 0);
}

#[test]
fn streamed_kernels_use_stream_order_without_a_device_scheduler() {
    let source = emit(&support::gemm_chain()).unwrap();

    // The shared ABI descriptor declares both layouts; inspect only generated runtime code.
    let code = source
        .code()
        .split("return &descriptor;\n}")
        .nth(1)
        .unwrap();

    // This is a generated ABI/runtime boundary: streamed callers need only
    // bindings and a stream, including when one GEMM consumes another's output.
    for forbidden in [
        "struct Context",
        "struct Task",
        "struct Header",
        "kDependencies",
        "kStages",
        "kTiles",
        "body_0_args",
        "kOutputDependencies",
        "workspace",
        "worker_count",
        "epoch",
        "initialize<<<",
        "complete_task",
        "claim(",
        "__threadfence_system",
        "cuda::atomic",
        "nvshmem",
        "PersistentRuntime",
    ] {
        assert!(
            !code.contains(forbidden),
            "streamed execution contains {forbidden}"
        );
    }
    assert!(code.contains("trinity_status(void* stream, trinity::abi::ErrorInfo* error)"));
    assert!(code.contains("StreamedRuntime{}"));
    // Two domain launches preserve the producer/consumer execution boundary.
    let launches = &source.execution().as_streamed().unwrap().launches;
    assert_eq!(
        launches.iter().map(|l| l.blocks).collect::<Vec<_>>(),
        [4, 2]
    );
    let mut previous = None;
    for launch in launches {
        let call = format!(
            "kernel_{}<<<{}, kThreads, kSharedBytes, stream>>>(bindings, 0LL)",
            launch.task, launch.blocks
        );
        let position = code.find(&call).unwrap();
        assert!(previous.is_none_or(|p| p < position));
        previous = Some(position);
    }
}

#[test]
fn persistent_runtime_keeps_control_storage_and_stage_hooks() {
    for backend in ["peer_push", "peer_pull", "one_shot_push_nbi"] {
        let source = emit(&support::input_gather(backend, 0, 2, true)).unwrap();
        let req = source.requirements();
        assert!(req.workspace_bytes > 0);
        assert_eq!(req.workspace_bytes % req.workspace_alignment, 0);
        assert_eq!(req.workspace_alignment, 8);
        assert!(req.workspace_symmetric);
        assert_eq!(req.workspace_bytes, source.execution().workspace_bytes());
        for required in [
            "struct Task",
            "kDependencies",
            "kStages",
            "kOutputDependencies",
            "worker_count",
            "epoch",
            "initialize<<<",
            "complete_task",
            "PersistentRuntime{c, task}",
            "nvshmemx_collective_launch",
            "runtime.await_stage(stage_base_",
            "runtime.prefetch_stage(stage_base_",
        ] {
            assert!(source.code().contains(required), "missing {required}");
        }
        assert!(!source.code().contains("StreamedRuntime"));
        assert!(source.code().contains(
            "trinity_status(void const* workspace, void* stream, trinity::abi::ErrorInfo* error)"
        ));
    }
}

#[test]
fn identity_streamed_execution_has_no_allocation_or_kernel_launch() {
    use trinity_lowering::{DType, PhysicalPlanBuilder, Storage};
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
    let x = b.add_value(DType::Bf16, [128, 128], Storage::External);
    b.bind_input("X", x);
    let source = emit(&b.finalize("Y", x).unwrap()).unwrap();
    assert_eq!(source.requirements().workspace_bytes, 0);
    assert!(
        source
            .execution()
            .as_streamed()
            .unwrap()
            .launches
            .is_empty()
    );
    assert!(!source.code().contains("<<<"));
}

#[test]
fn peer_regions_match_an_elementwise_allgather_oracle_on_both_axes() {
    for backend in ["peer_push", "peer_pull", "one_shot_push_nbi"] {
        for axis in 0..2 {
            let plan = support::gather(backend, axis, 3);
            let (id, op) = plan.operations().next().unwrap();
            let OperationPayload::Communication(_comm) = op.payload() else {
                unreachable!()
            };

            let source = plan.value_instance(op.inputs()[0]).unwrap().shape();
            let target = plan.value_instance(op.outputs()[0]).unwrap().shape();
            let mut seen = vec![vec![false; target[0] * target[1]]; 3];
            let emission = support::operation_work(&plan, id);
            for rank_work in emission.work {
                for work in rank_work {
                    for write in &work.writes {
                        for row in write.origin[0]..write.origin[0] + write.extent[0] {
                            for col in write.origin[1]..write.origin[1] + write.extent[1] {
                                let mut local = [row, col];
                                let peer = local[axis] / source[axis];
                                local[axis] %= source[axis];
                                assert!(work.reads.iter().any(|r| r.rank == peer
                                    && (0..2).all(|a| local[a] >= r.origin[a]
                                        && local[a] < r.origin[a] + r.extent[a])));
                                let position = &mut seen[write.rank][row * target[1] + col];
                                assert!(!*position);
                                *position = true;
                            }
                        }
                    }
                }
            }
            assert!(seen.iter().flatten().all(|v| *v));
            let emission = emit(&plan).unwrap();
            assert!(emission.requirements().cooperative_launch);
            let output = plan.output().value().index();
            if backend == "peer_push" {
                assert!(emission.requirements().buffers[output].symmetric);
                assert!(
                    emission
                        .execution()
                        .as_persistent()
                        .unwrap()
                        .output_dependencies[0]
                        .iter()
                        .any(|d| d.rank == 2)
                );
            } else if backend == "peer_pull" {
                assert!(emission.requirements().buffers[op.inputs()[0].index()].symmetric);
                assert!(!emission.requirements().buffers[output].symmetric);
                assert!(
                    emission
                        .execution()
                        .as_persistent()
                        .unwrap()
                        .output_dependencies[0]
                        .iter()
                        .all(|d| d.rank == 0)
                );
            }
        }
    }
}

#[test]
fn gemm_only_waits_for_the_regions_read_by_each_k_stage() {
    for backend in ["peer_push", "peer_pull", "one_shot_push_nbi"] {
        for axis in 0..2 {
            let plan = support::input_gather(backend, axis, 2, false);
            let source = emit(&plan).unwrap();
            let execution = source.execution().as_persistent().unwrap();
            let (_, gemm) = plan
                .operations()
                .find(|(_, op)| matches!(op.payload(), OperationPayload::Compute(_)))
                .unwrap();
            let rhs = gemm.inputs()[1];
            for (task, task_work) in execution
                .schedule
                .iter()
                .zip(&execution.work)
                .filter(|(t, _)| !t.stages.is_empty())
            {
                for (stage, actual) in task.stages.iter().enumerate() {
                    let expected_region = ([stage * 64, task_work.coordinate[1] * 128], [64, 128]);
                    let mut expected = BTreeSet::new();
                    for (producer, work) in execution.schedule.iter().zip(&execution.work) {
                        for write in &work.writes {
                            if write.value == rhs
                                && write.rank == task.rank
                                && (0..2).all(|a| {
                                    write.origin[a] < expected_region.0[a] + expected_region.1[a]
                                        && expected_region.0[a] < write.origin[a] + write.extent[a]
                                })
                            {
                                expected.insert(trinity_lowering::emit::Dependency {
                                    rank: producer.rank,
                                    slot: producer.slot,
                                });
                            }
                        }
                    }
                    assert_eq!(
                        actual.dependencies.iter().copied().collect::<BTreeSet<_>>(),
                        expected
                    );
                }
                assert_eq!(task.dependencies, task.stages[0].dependencies);
            }
        }
    }
}

#[test]
fn compute_edges_wait_for_the_whole_producer_but_output_gather_uses_tiles() {
    let source = emit(&support::gemm_chain_world(2)).unwrap();
    let execution = source.execution().as_persistent().unwrap();
    for (_, t) in execution
        .tasks
        .iter()
        .zip(&execution.schedule)
        .filter(|(t, _)| t.body == 1)
    {
        assert_eq!(t.dependencies.len(), 4);
        assert!(t.stages.iter().all(|s| s.dependencies.is_empty()));
    }
    for backend in ["peer_push", "peer_pull"] {
        let source = emit(&support::output_gather(backend, 0, 2)).unwrap();
        for t in source
            .execution()
            .as_persistent()
            .unwrap()
            .schedule
            .iter()
            .filter(|t| t.stages.is_empty())
        {
            assert_eq!(t.dependencies.len(), 1);
        }
    }
}

#[test]
fn nvls_collectives_share_one_world_order_including_different_statements() {
    let source = emit(&support::peer_chain(
        "one_shot_push_nbi",
        "one_shot_push_nbi",
        2,
    ))
    .unwrap();
    let execution = source.execution().as_persistent().unwrap();
    for rank in 0..2 {
        let tasks: Vec<_> = execution
            .schedule
            .iter()
            .filter(|t| t.rank == rank && t.ordered_collective)
            .collect();
        assert_eq!(tasks.len(), 4);
        for pair in tasks.windows(2) {
            for peer in 0..2 {
                assert!(
                    pair[1]
                        .dependencies
                        .contains(&trinity_lowering::emit::Dependency {
                            rank: peer,
                            slot: pair[0].slot
                        })
                );
            }
        }
    }
}

#[test]
fn fusion_emits_a_single_task_set_for_compatible_gemms() {
    let mut b = support::Builder::new(1);
    let a = b.input("A", [128, 128]);
    let w = b.input("W", [128, 128]);
    let v = b.input("V", [128, 128]);
    let x = b.gemm(a, w, 128, 128, 128);
    let y = b.gemm(x, v, 128, 128, 128);
    let plan = b.finish(y);
    let result = fuse(
        &plan,
        fusion_rules(TargetCapability::Cuda(CudaTargetCapability::Hopper)),
    );
    let plans = result.unwrap();
    assert_eq!(plans.len(), 2);
    assert_eq!(
        emit(&plans[1])
            .unwrap()
            .execution()
            .as_streamed()
            .unwrap()
            .launches
            .len(),
        1
    );
}

#[test]
fn templates_reject_missing_fields() {
    assert!(trinity_lowering::emit::render_template("{{ missing }}", &()).is_err());
}

#[test]
fn public_builder_with_inapplicable_instance_returns_error() {
    use trinity_lowering::*;
    let valid = support::gemm(128, 128, 64, 1);
    let payload = valid.operations().next().unwrap().1.payload().clone();
    for dtype in [DType::Bf16, DType::Fp32] {
        let mut b =
            PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
        let a = b.add_value(dtype, [128, 65], Storage::External);
        b.bind_input("A", a);
        let w = b.add_value(dtype, [65, 128], Storage::External);
        b.bind_input("W", w);
        let out = b.add_value(dtype, [128, 128], Storage::External);
        let op = b.add_operation([a, w], [out], payload.clone());
        b.add_statement(trinity_lowering::Statement::Operation(op));
        assert!(matches!(
            b.finalize("Y", out),
            Err(trinity_lowering::PhysicalInvariantError::InvalidProgram(_))
        ));
    }
}

#[test]
fn nvls_rejects_unpacked_bf16_columns_before_cuda_generation() {
    use trinity_lowering::*;
    let mut b = PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 2);
    let x = b.add_value(DType::Bf16, [128, 3], Storage::External);
    b.bind_input("X", x);
    let y = b.add_value(DType::Bf16, [256, 3], Storage::External);
    let instance = all_gather_implementations(b_target())[0]
        .enumerate(DType::Bf16, [&[128, 3], &[256, 3]], 0, 2)
        .pop()
        .unwrap();
    let id = b.add_operation(
        [x],
        [y],
        OperationPayload::Communication(CommunicationOperation::new(
            CommunicationKind::AllGather,
            instance,
        )),
    );
    b.add_statement(Statement::Operation(id));
    assert!(
        b.finalize("Y", y)
            .err()
            .unwrap()
            .to_string()
            .contains("even row-major")
    );
    fn b_target() -> TargetCapability {
        TargetCapability::Cuda(CudaTargetCapability::Hopper)
    }
}

#[test]
fn gathered_lhs_stage_readiness_tracks_m_rows_and_k_columns() {
    for backend in ["peer_push", "peer_pull", "one_shot_push_nbi"] {
        for axis in 0..2 {
            let plan = support::lhs_gather(backend, axis, 2);
            let (id, op) = plan
                .operations()
                .find(|(_, op)| matches!(op.payload(), OperationPayload::Compute(_)))
                .unwrap();
            let OperationPayload::Compute(_gemm) = op.payload() else {
                unreachable!()
            };
            let emission = support::operation_work(&plan, id);
            for work in &emission.work[0] {
                for (k, reads) in work.stages.iter().enumerate() {
                    let lhs = reads.iter().find(|r| r.value == op.inputs()[0]).unwrap();
                    assert_eq!(lhs.origin, [work.coordinate[0] * 128, k * 64, 0]);
                    assert_eq!(lhs.extent, [128, 64, 1]);
                }
            }
            emit(&plan).unwrap();
        }
    }
}

#[test]
fn emits_tensor_spans_beyond_32_bit_indexing() {
    use trinity_lowering::{DType, PhysicalPlanBuilder, Storage};

    // Metadata needs no device allocation or enumeration of billions of elements.
    for world in [1, 2] {
        for shape in [
            [i32::MAX as usize, 1],
            [i32::MAX as usize + 1, 1],
            [32768, 65536],
            [65536, 65536],
        ] {
            let mut b = PhysicalPlanBuilder::new(
                TargetCapability::Cuda(CudaTargetCapability::Hopper),
                world,
            );
            let value = b.add_value(DType::Bf16, shape, Storage::External);
            b.bind_input("X", value);
            let source = emit(&b.finalize("Y", value).unwrap()).unwrap();
            let buffer = &source.requirements().buffers[0];
            assert_eq!(buffer.bytes, shape.iter().product::<usize>() * 2);
            assert_eq!(buffer.strides, [shape[1], 1]);
        }
    }
}

#[test]
fn simt_and_gemm_access_tiles_beyond_32_bit_offsets() {
    for world in [1, 2] {
        for gemm in [false, true] {
            let source = emit(&support::loop_ir::large_offset(world, gemm)).unwrap();
            assert_eq!(source.execution().tasks().len(), world);
            let input = source
                .requirements()
                .buffers
                .iter()
                .find(|b| b.input_names.iter().any(|n| n == "X"))
                .unwrap();
            assert!(input.bytes > (1usize << 32) * 2);
            assert!(source.code().contains("std::int64_t(lv0)"));
            if !gemm {
                assert!(source.code().contains("4294967296"));
                assert!(!source.code().contains("body_0_args"));
            }
        }
    }
}

#[test]
fn persistent_epoch_is_gpu_owned_and_failure_is_sticky() {
    let source = emit(&support::gemm(128, 128, 64, 2)).unwrap();
    let code = source.code();

    assert!(code.contains("h->epoch = previous + 1"));
    assert!(code.contains("c.epoch = header(c)->epoch"));
    assert!(code.contains("previous == ~0ULL"));
    assert!(!code.contains("p->epoch"));
    assert!(!code.contains("h->error = 0"));
}
