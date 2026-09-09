#[allow(dead_code)]
mod support;
use std::collections::BTreeSet;
use trinity_lowering::{
    CudaTargetCapability, EmitError, OperationPayload, TargetCapability, emit, fuse, fusion_rules,
};

#[test]
fn public_api_source_is_deterministic_and_preserves_binding_names() {
    let plan = support::gemm(256, 256, 192, 1);
    let a = emit(&plan).unwrap();
    let b = emit(&plan).unwrap();
    assert_eq!(a.code(), b.code());
    let req = a.requirements();
    assert_eq!(req.target, CudaTargetCapability::Hopper);
    assert_eq!(req.cuda_arch, "sm_90a");
    assert_eq!(plan.target(), TargetCapability::Cuda(req.target));
    assert_eq!(req.shared_memory_bytes, 65536);
    assert_eq!(req.block_threads, 128);
    assert!(!req.nvshmem);
    assert!(!req.cooperative_launch);
    assert_eq!(req.minimum_workers, 1);
    assert_eq!(a.execution().tasks_per_rank, 4);
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
    // Coordinate tables cover four producer tiles followed by two consumer
    // tiles. Streamed launch order must preserve that producer/consumer edge.
    let launches = &source.execution().launches;
    assert_eq!(launches.iter().map(|l| l.count).collect::<Vec<_>>(), [4, 2]);
    let mut previous = None;
    for launch in launches {
        let call = format!(
            "kernel_{}<<<{}, kThreads, kSharedBytes, stream>>>(bindings)",
            launch.operation, launch.count
        );
        let position = code.find(&call).unwrap();
        assert!(previous.is_none_or(|p| p < position));
        previous = Some(position);
        assert!(code.contains(&format!("kTiles[{} + blockIdx.x]", launch.begin)));
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
            "runtime.await_stage(0)",
            "runtime.prefetch_stage(k + 1, K / 64)",
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
    assert!(source.execution().launches.is_empty());
    assert!(!source.code().contains("<<<"));
}

#[test]
fn peer_regions_match_an_elementwise_allgather_oracle_on_both_axes() {
    for backend in ["peer_push", "peer_pull", "one_shot_push_nbi"] {
        for axis in 0..2 {
            let plan = support::gather(backend, axis, 3);
            let (id, op) = plan.operations().next().unwrap();
            let OperationPayload::Communication(comm) = op.payload() else {
                unreachable!()
            };
            let implementation = comm.implementation().definition().cuda().unwrap();
            let source = plan.value_instance(op.inputs()[0]).unwrap().shape();
            let target = plan.value_instance(op.outputs()[0]).unwrap().shape();
            let mut seen = vec![vec![false; target[0] * target[1]]; 3];
            let emission = implementation.specialize(&plan, id).unwrap();
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
                    emission.execution().output_dependencies[0]
                        .iter()
                        .any(|d| d.rank == 2)
                );
            } else if backend == "peer_pull" {
                assert!(emission.requirements().buffers[op.inputs()[0].index()].symmetric);
                assert!(!emission.requirements().buffers[output].symmetric);
                assert!(
                    emission.execution().output_dependencies[0]
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
            let execution = source.execution();
            let (_, gemm) = plan
                .operations()
                .find(|(_, op)| matches!(op.payload(), OperationPayload::Compute(_)))
                .unwrap();
            let rhs = gemm.inputs()[1];
            for task in execution.tasks.iter().filter(|t| !t.stages.is_empty()) {
                for (stage, actual) in task.stages.iter().enumerate() {
                    let expected_region = ([stage * 64, task.coordinate[1] * 128], [64, 128]);
                    let mut expected = BTreeSet::new();
                    for rank in 0..2 {
                        let mut slot = 0;
                        for (id, op) in plan.operations() {
                            let definition = match op.payload() {
                                OperationPayload::Compute(x) => x.implementation().definition(),
                                OperationPayload::Communication(x) => {
                                    x.implementation().definition()
                                }
                            };
                            for work in definition
                                .cuda()
                                .unwrap()
                                .specialize(&plan, id)
                                .unwrap()
                                .work[rank]
                                .iter()
                            {
                                for write in &work.writes {
                                    if write.value == rhs
                                        && write.rank == task.rank
                                        && (0..2).all(|a| {
                                            write.origin[a]
                                                < expected_region.0[a] + expected_region.1[a]
                                                && expected_region.0[a]
                                                    < write.origin[a] + write.extent[a]
                                        })
                                    {
                                        expected.insert(trinity_lowering::emit::Dependency {
                                            rank,
                                            slot,
                                        });
                                    }
                                }
                                slot += 1;
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
            assert_eq!(
                source.requirements().minimum_workers,
                if backend == "one_shot_push_nbi" && axis == 1 {
                    1
                } else {
                    2
                }
            );
        }
    }
}

#[test]
fn compute_edges_wait_for_the_whole_producer_but_output_gather_uses_tiles() {
    let source = emit(&support::gemm_chain()).unwrap();
    for t in source.execution().tasks.iter().filter(|t| t.operation == 1) {
        assert_eq!(t.dependencies.len(), 4);
        assert!(t.stages.iter().all(|s| s.dependencies.is_empty()));
    }
    for backend in ["peer_push", "peer_pull"] {
        let source = emit(&support::output_gather(backend, 0, 2)).unwrap();
        for t in source
            .execution()
            .tasks
            .iter()
            .filter(|t| t.stages.is_empty())
        {
            assert_eq!(t.dependencies.len(), 1);
        }
    }
}

#[test]
fn nvls_collectives_share_one_world_order_including_different_actions() {
    let source = emit(&support::peer_chain(
        "one_shot_push_nbi",
        "one_shot_push_nbi",
        2,
    ))
    .unwrap();
    let execution = source.execution();
    for rank in 0..2 {
        let tasks: Vec<_> = execution
            .tasks
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
fn promoted_storage_is_rejected() {
    let mut b = support::Builder::new(1);
    let a = b.input("A", [128, 128]);
    let w = b.input("W", [128, 128]);
    let v = b.input("V", [128, 128]);
    let x = b.gemm(a, w, 128, 128, 128);
    let y = b.gemm(x, v, 128, 128, 128);
    let plan = b.finish(y);
    let plans = fuse(
        &plan,
        fusion_rules(TargetCapability::Cuda(CudaTargetCapability::Hopper)),
    )
    .unwrap();
    assert!(plans.len() > 1);
    for plan in &plans[1..] {
        assert!(matches!(emit(plan), Err(EmitError::Unsupported(_))));
    }
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
        b.add_action([op]);
        assert!(matches!(
            emit(&b.finalize("Y", out).unwrap()),
            Err(EmitError::Unsupported(_))
        ));
    }
}

#[test]
fn nvls_rejects_unpacked_bf16_columns_before_cuda_generation() {
    let mut b = support::Builder::new(2);
    let x = b.input("X", [128, 3]);
    let y = b.gather(x, [128, 3], 0, 2, "one_shot_push_nbi");
    assert!(matches!(emit(&b.finish(y)), Err(EmitError::Unsupported(_))));
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
            let OperationPayload::Compute(gemm) = op.payload() else {
                unreachable!()
            };
            let emission = gemm
                .implementation()
                .definition()
                .cuda()
                .unwrap()
                .specialize(&plan, id)
                .unwrap();
            for work in &emission.work[0] {
                for (k, reads) in work.stages.iter().enumerate() {
                    let lhs = reads.iter().find(|r| r.value == op.inputs()[0]).unwrap();
                    assert_eq!(lhs.origin, [work.coordinate[0] * 128, k * 64]);
                    assert_eq!(lhs.extent, [128, 64]);
                }
            }
            let source = emit(&plan).unwrap();
            let expected = if backend == "one_shot_push_nbi" && axis == 0 {
                1
            } else {
                2
            };
            assert_eq!(source.requirements().minimum_workers, expected);
        }
    }
}

#[test]
fn rejects_tensor_spans_that_exceed_cute_index_width() {
    use trinity_lowering::{DType, PhysicalPlanBuilder, Storage};

    let plan = support::gemm(65536, 128, 65536, 1);
    assert!(matches!(emit(&plan), Err(EmitError::Unsupported(_))));

    // Identity plans exercise the size boundary without enumerating GEMM work.
    for shape in [
        [i32::MAX as usize, 1],
        [i32::MAX as usize + 1, 1],
        [2, i32::MAX as usize / 2 + 1],
        [usize::MAX, 2],
        [1, usize::MAX],
    ] {
        let mut b =
            PhysicalPlanBuilder::new(TargetCapability::Cuda(CudaTargetCapability::Hopper), 1);
        let value = b.add_value(DType::Bf16, shape, Storage::External);
        b.bind_input("X", value);
        let result = emit(&b.finalize("Y", value).unwrap());
        if shape == [i32::MAX as usize, 1] {
            assert_eq!(
                result.unwrap().requirements().buffers[0].bytes,
                i32::MAX as usize * 2
            );
        } else {
            assert!(matches!(result, Err(EmitError::Unsupported(_))));
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
