"""Real GPU stream/Graph lifetime tests using a portable COPY host-ABI fixture.

This deliberately bypasses only the public Hopper artifact admission step; it
never recompiles or claims to validate the generated sm_90a/WGMMA backend.
"""

import gc
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import pytest
import torch
import trinity_lowering as tl
from trinity_lowering.metadata import requirements
from trinity_lowering.runtime import LoadedModule, device_of, native

pytestmark = pytest.mark.cuda_runtime


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    temp = tmp_path_factory.mktemp("cuda-runtime")
    buffers = [
        dict(
            value=i,
            input_names=["X"] if i == 0 else [],
            output_name="Y" if i else None,
            shape=[128, 128],
            strides=[128, 1],
            dtype="bf16",
            bytes=32768,
            alignment=16,
            external=True,
            symmetric=False,
        )
        for i in (0, 1)
    ]
    data = dict(
        target="hopper",
        cuda_arch="sm_90a",
        world_size=1,
        buffers=buffers,
        workspace_bytes=0,
        workspace_alignment=1,
        workspace_symmetric=False,
        cooperative_launch=False,
        shared_memory_bytes=0,
        block_threads=128,
        minimum_workers=1,
        nvshmem=False,
        nvls=False,
    )
    (temp / "fixture_metadata.h").write_text(
        "static constexpr char metadata[] = " + json.dumps(json.dumps(data)) + ";\n"
    )

    capability = torch.cuda.get_device_capability(0)
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0"))
    path = temp / "program.so"
    subprocess.run(
        [
            str(cuda / "bin/nvcc"),
            "--shared",
            "-Xcompiler=-fPIC",
            "--cudart=shared",
            "-std=c++17",
            f"-arch=sm_{capability[0]}{capability[1]}",
            "-I",
            str(root.parent / "src/emit/cuda"),
            "-I",
            str(temp),
            str(root / "tests/fixtures/cuda_library.cu"),
            "-o",
            str(path),
        ],
        check=True,
        timeout=120,
    )

    return path, data


def module(library):
    path, data = library
    handle = native().Module.load(str(path), 0, 1, None)
    return LoadedModule(
        handle,
        {"fixture": "portable copy"},
        requirements(data),
        torch.device("cuda:0"),
        None,
        "fixture",
    )


def collect_until(fn):
    deadline = time.monotonic() + 10
    while True:
        native().collect_local()
        try:
            fn()
            return
        except tl.ResourceBusy:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def test_real_device_admission():
    if torch.cuda.get_device_capability(0) != (9, 0):
        with pytest.raises(ValueError, match="Hopper"):
            device_of("cuda:0")
    else:
        assert device_of("cuda:0") == torch.device("cuda:0")


@torch.inference_mode()
def test_streams_repeat_and_gc_allocator_pressure(library):
    m = module(library)
    producer, execution_stream, consumer = [torch.cuda.Stream() for _ in range(3)]

    with torch.cuda.stream(producer):
        x = torch.randn(128, 128, device="cuda:0", dtype=torch.bfloat16)
        reference = x.clone()
        produced = producer.record_event()

    e = m.prepare({"X": x})

    with torch.cuda.stream(execution_stream):
        e.run(wait_for=[produced])
        first = e.output.clone()
        for _ in range(8):
            assert e.run().data_ptr() == e.output.data_ptr()
        done = execution_stream.record_event()

    with pytest.raises(tl.ResourceBusy):
        e.run(stream=consumer)
    with pytest.raises(tl.ResourceBusy):
        e.close(wait=False)

    del x
    holder = [e]
    del e
    thread = threading.Thread(target=holder.clear)
    thread.start()
    thread.join()
    gc.collect()

    with torch.cuda.stream(consumer):
        _pressure = [
            torch.empty(128, 128, device="cuda:0", dtype=torch.bfloat16) for _ in range(128)
        ]
        consumer.wait_event(done)
        actual = first.clone()

    consumer.synchronize()

    assert torch.equal(actual, reference)

    collect_until(m.close)


@torch.inference_mode()
def test_graph_replay_ownership_and_mutation(library):
    m = module(library)
    x = torch.randn(128, 128, device="cuda:0", dtype=torch.bfloat16)
    bias = torch.ones_like(x)

    e = m.prepare({"X": x})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = tl.Graph.capture(
        lambda: torch.relu(e.run() + bias), executions=[e], keepalive=[bias], stream=stream
    )

    with pytest.raises(tl.ResourceBusy):
        e.close()
    with pytest.raises(tl.ResourceBusy):
        m.close()

    with torch.cuda.stream(stream):
        for _ in range(8):
            g.replay()
        actual = g.output.clone()
    g.wait(30)

    assert torch.equal(actual, torch.relu(x + bias))

    with torch.cuda.stream(stream):
        x.fill_(2)
        g.replay()
    g.wait(30)

    assert torch.equal(g.output, torch.full_like(x, 3))

    x.set_(torch.zeros_like(x))

    with pytest.raises(ValueError, match="storage"):
        g.replay(stream)

    g.close()
    e.close()
    m.close()


@pytest.mark.parametrize("failure_call", [2, 4])
@torch.inference_mode()
def test_capture_failure_and_graph_gc(library, failure_call):
    m = module(library)
    x = torch.ones(128, 128, device="cuda:0", dtype=torch.bfloat16)

    e = m.prepare({"X": x})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    calls = 0

    def broken():
        nonlocal calls
        calls += 1
        result = e.run()
        if calls == failure_call:
            raise ValueError("callback failed after submission")
        return result

    with pytest.raises(ValueError, match="callback failed"):
        tl.Graph.capture(broken, executions=[e], stream=stream)

    stream.synchronize()
    collect_until(e.close)

    e = m.prepare({"X": x})
    graph = tl.Graph.capture(lambda: e.run().clone(), executions=[e], stream=stream)
    graph.replay(stream)

    holder = [graph]
    del graph
    thread = threading.Thread(target=holder.clear)
    thread.start()
    thread.join()

    with pytest.raises(tl.ResourceBusy):
        e.close(wait=False)

    stream.synchronize()
    collect_until(e.close)
    m.close()


@torch.inference_mode()
def test_partial_enqueue_error_and_release_retry(library, monkeypatch):
    m = module(library)
    x = torch.ones(128, 128, device="cuda:0", dtype=torch.bfloat16)

    e = m.prepare({"X": x})

    monkeypatch.setenv("TRINITY_FIXTURE_FAILURE", "launch")

    with pytest.raises(tl.RuntimeFailure) as error:
        e.run()
    assert error.value.code == 719 and error.value.submitted
    with pytest.raises(tl.RuntimeFailure):
        _ = e.output

    monkeypatch.delenv("TRINITY_FIXTURE_FAILURE")

    with pytest.raises(tl.RuntimeFailure):
        e.run()
    with pytest.raises(tl.RuntimeFailure):
        e.close()
    assert e._closed

    monkeypatch.setenv("TRINITY_FIXTURE_FAILURE", "release")

    with pytest.raises(tl.RuntimeFailure) as error:
        m.close()
    assert error.value.code == 19

    monkeypatch.delenv("TRINITY_FIXTURE_FAILURE")
    m.close()


@torch.inference_mode()
def test_private_module_preparation_state(library):
    a, b = module(library), module(library)
    x = torch.ones(128, 128, device="cuda:0", dtype=torch.bfloat16)
    ea, eb = a.prepare({"X": x}), b.prepare({"X": x})

    ea.run()
    eb.run()
    ea.wait(30)
    eb.wait(30)

    assert torch.equal(ea.output, eb.output)

    ea.close()
    eb.close()
    a.close()
    b.close()


@torch.inference_mode()
def test_tensor_validation_views_and_stream_restore(library):
    m = module(library)
    storage = torch.zeros(2, 128, 128, device="cuda:0", dtype=torch.bfloat16)
    x, y = storage[0], storage[1]

    with pytest.raises(ValueError, match="overlap"):
        m.prepare({"X": x}, out=x)
    with pytest.raises(ValueError, match="dtype/shape/stride"):
        m.prepare({"X": x.T})

    misaligned = torch.empty(128 * 128 + 1, device="cuda:0", dtype=torch.bfloat16)[1:].view(
        128, 128
    )

    with pytest.raises(ValueError, match="alignment"):
        m.prepare({"X": misaligned})

    e = m.prepare({"X": x}, out=y)
    caller, runner = torch.cuda.Stream(), torch.cuda.Stream()
    runner.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(caller):
        e.run(stream=runner)

        assert torch.cuda.current_stream() == caller

        e.wait(30)

        assert torch.cuda.current_stream() == caller
    e.close()
    m.close()


def test_prepare_error_and_autograd_contract(library, monkeypatch):
    m = module(library)
    x = torch.ones(128, 128, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)

    with pytest.raises(ValueError, match="no_grad"):
        m.prepare({"X": x})

    with torch.no_grad():
        monkeypatch.setenv("TRINITY_FIXTURE_FAILURE", "prepare")

        with pytest.raises(tl.RuntimeFailure) as error:
            m.prepare({"X": x})
        assert error.value.code == 17 and not error.value.submitted

        monkeypatch.delenv("TRINITY_FIXTURE_FAILURE")
    m.close()


def test_invalidated_capture_pins_owners_in_subprocess(library):
    import sys

    path, data = library
    code = """
import json, sys
from pathlib import Path
import torch
import trinity_lowering as tl
from test_cuda_runtime import module
library = (Path(sys.argv[1]), json.loads(sys.argv[2]))
with torch.inference_mode():
    m = module(library)
    e = m.prepare({"X": torch.ones(128, 128, device="cuda:0", dtype=torch.bfloat16)})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    count = 0
    def step():
        global count
        count += 1
        result = e.run()
        if count == 4:
            torch.cuda.current_stream().synchronize()
        return result
    try:
        tl.Graph.capture(step, executions=[e], stream=stream)
        raise AssertionError("capture should fail")
    except RuntimeError as error:
        assert "restart the process" in str(error.cleanup_error)
    try:
        module(library)
        raise AssertionError("device should be poisoned")
    except tl.RuntimeFailure as error:
        assert "restart the process" in str(error)
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-c", code, str(path), json.dumps(data)], check=True, timeout=30, env=env
    )
