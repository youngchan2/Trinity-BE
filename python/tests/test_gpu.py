import sys
import threading
from pathlib import Path
import pytest
import torch
import trinity_lowering as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from plans import gemm, gemm_relu
from test_compiler import identity

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("m,n,k", [(128, 128, k) for k in (64, 128, 192, 320)] + [(256, 256, 192)])
@torch.inference_mode()
def test_gemm_bias_repeated_side_stream_and_allocator(m, n, k):
    artifact = tl.compile(tl.emit(gemm(m, n, k)))
    module = tl.load(artifact, "cuda:0")
    artifact.close()

    producer, runner, consumer = [torch.cuda.Stream(device=module.device) for _ in range(3)]
    with torch.cuda.stream(producer):
        x = torch.randn(m, k, device=module.device, dtype=torch.bfloat16)
        w = torch.randn(k, n, device=module.device, dtype=torch.bfloat16)
        bias = torch.randn(m, n, device=module.device, dtype=torch.bfloat16)
        reference = (x.float() @ w.float()).bfloat16() + bias
        produced = producer.record_event()

    execution = module.prepare({"X": x, "W": w, "bias": bias})

    with pytest.raises(tl.ResourceBusy):
        module.close()

    with torch.cuda.stream(runner):
        execution.run(wait_for=[produced])
        first = execution.output.clone()
        for _ in range(16):
            execution.run()
        done = runner.record_event()

    with pytest.raises(tl.ResourceBusy):
        execution.run(stream=consumer)

    del x, w, bias
    with torch.cuda.stream(consumer):
        consumer.wait_event(done)
        copy = first.clone()
        _pressure = [
            torch.empty(m, max(k, n), device=module.device, dtype=torch.bfloat16) for _ in range(64)
        ]

    execution.wait(timeout=30)
    consumer.synchronize()
    torch.testing.assert_close(copy, reference, atol=0.125, rtol=0.02)
    execution.run(stream=consumer)

    execution.wait(timeout=30)
    execution.close()
    module.close()


@torch.inference_mode()
def test_graph_with_lowered_relu_and_static_metadata():
    artifact = tl.compile(tl.emit(gemm_relu()))
    module = tl.load(artifact, "cuda:0")
    io = module.allocate_io()
    io.inputs["X"].normal_()
    io.inputs["W"].normal_()
    io.inputs["bias"].normal_()

    execution = module.prepare(io.inputs, out=io.output)
    stream = torch.cuda.Stream(device=module.device)
    stream.wait_stream(torch.cuda.current_stream(module.device))

    graph = tl.Graph.capture(execution.run, executions=[execution], stream=stream)

    with pytest.raises(tl.ResourceBusy):
        execution.close()
    with pytest.raises(tl.ResourceBusy):
        execution.run(stream=stream)

    with torch.cuda.stream(stream):
        for _ in range(10):
            graph.replay()
        snapshot = graph.output.clone()
    graph.wait(timeout=30)
    reference = torch.relu(
        (io.inputs["X"].float() @ io.inputs["W"].float()).bfloat16() + io.inputs["bias"]
    )
    torch.testing.assert_close(snapshot, reference, atol=0.125, rtol=0.02)

    with torch.cuda.stream(stream):
        graph.replay()

    # Dropping a Graph from a different thread must retain its pending native state.
    holder = [graph]
    del graph
    thread = threading.Thread(target=lambda: holder.clear())
    thread.start()
    thread.join()
    stream.synchronize()
    from trinity_lowering.runtime import native

    native().collect_local()
    execution.close()
    module.close()
    artifact.close()


@torch.inference_mode()
def test_identity_and_storage_mutation():
    artifact = tl.compile(tl.emit(identity()))
    module = tl.load(artifact, "cuda:0")
    x = torch.zeros(128, 128, dtype=torch.bfloat16, device=module.device)
    names = artifact.requirements.buffers[0].input_names

    execution = module.prepare(dict.fromkeys(names, x))

    assert execution.run().data_ptr() == x.data_ptr()

    execution.wait(timeout=30)

    x.set_(torch.ones_like(x))

    with pytest.raises(ValueError, match="storage"):
        execution.run()

    execution.close()
    module.close()
    artifact.close()
