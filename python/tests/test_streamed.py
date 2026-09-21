"""Public emission, compiler target admission, and generated GPU execution."""
import sys
from pathlib import Path

import pytest
import torch
import trinity_lowering as tl
from trinity_lowering.metadata import TARGET_CAPABILITIES
from trinity_lowering.runtime import device_of

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from plan_syntax import index, load, store
from tensor_ops import Builder


def pointwise(target, dtype, storage="global"):
    b = Builder(target=target)
    x = b.input("X", dtype, [3, 131])
    # Both operations in one explicit tile scope, including Register continuation.
    shape = [3, 131]
    c = b.value(dtype, shape, storage)
    y = b.value(dtype, shape, "external")
    access = index("(tile row 1)", "(clipped_tile col 64)")
    first = b.builder.add_operation([x], [c], expression=store(c, shape, f"(sqr {load(x, shape, access)})", access))
    second = b.builder.add_operation([c], [y], expression=store(y, shape, f"(sigmoid {load(c, shape, access)})", access))
    cols = b.builder.add_loop("parallel", "col", 0, 131, 64, [first, second])
    rows = b.builder.add_loop("parallel", "row", 0, 3, 1, [cols])
    return b.builder.build([rows], "Y", y)


def gemm(target, storage):
    b = tl.PhysicalPlanBuilder(target_name=target)
    x = b.add_value("bf16", [16, 128], "external")
    w = b.add_value("bf16", [128, 256], "external")
    c = b.add_value("bf16", [16, 256], storage)
    y = b.add_value("bf16", [16, 256], "external")
    b.bind_input("X", x)
    b.bind_input("W", w)
    ci = index("fulltile", "(tile n 128)")
    lhs = load(x, [16, 128], index("fulltile", "(tile k 64)"))
    rhs = load(w, [128, 256], index("(tile k 64)", "(tile n 128)"))
    op = b.add_operation([x, w], [c], expression=store(c, [16, 256], f"(+ {load(c, [16, 256], ci)} (@ {lhs} {rhs}))", ci))
    serial = b.add_loop("sequential", "k", 0, 128, 64, [op])
    relu = b.add_operation([c], [y], expression=store(y, [16, 256], f"(relu {load(c, [16, 256], ci)})", ci))
    cols = b.add_loop("parallel", "n", 0, 256, 128, [serial, relu])
    return b.build([cols], "Y", y)


@pytest.mark.parametrize("target", list(TARGET_CAPABILITIES))
def test_targets_metadata_compiler_flags_and_device_admission(target, sdk, monkeypatch):
    _, config = sdk
    source = tl.emit(pointwise(target, "bf16", "register"))
    assert source.requirements.target == target
    assert len(source.requirements.buffers) == 2
    artifact = tl.compile(source, config)
    arch = {"hopper": "sm_90a", "sm89": "sm_89", "sm120": "sm_120"}[target]
    assert f"-arch={arch}" in artifact.diagnostics["arguments"]
    artifact.close()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: TARGET_CAPABILITIES[target])
    assert device_of("cuda:0", target) == torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (0, 0))
    with pytest.raises(ValueError, match="requires"):
        device_of("cuda:0", target)


@pytest.mark.parametrize("target", ["sm_90a", "sm_89", "sm_120"])
def test_target_aliases_apply_to_builder_and_ir(target):
    b = tl.PhysicalPlanBuilder(target_name=target)
    x = b.add_value("fp32", [128], "external")
    b.bind_input("X", x)
    assert tl.emit(b.build([], "Y", x)).requirements.target in TARGET_CAPABILITIES
    text = "(store (view (output Y) (layout (axis a 128))) (load (view (input X) (layout (axis a 128))) (keyed_index (slot a fulltile))) (keyed_index (slot a fulltile)))"
    plan, = tl.lower_ir(text, {}, {"X": "fp32", "Y": "fp32"}, target_name=target)
    assert tl.emit(plan).requirements.target in TARGET_CAPABILITIES


@pytest.mark.gpu
@pytest.mark.parametrize("target", list(TARGET_CAPABILITIES))
@pytest.mark.parametrize("case", ["pointwise_fp32", "pointwise_bf16", "gemm_global", "gemm_register", "reduction"])
@torch.inference_mode()
def test_generated_kernels_repeat_on_side_stream_and_graph(target, case):
    if case.startswith("gemm"):
        plan = gemm(target, case.split("_")[1])
    elif case == "reduction":
        b = Builder(target=target)
        x = b.input("X", "fp32", [3, 131])
        plan = b.finish(b.reduce(x, "external"))
    else:
        plan = pointwise(target, case.split("_")[1], "register")
    artifact = tl.compile(tl.emit(plan))
    module = tl.load(artifact, "cuda:0")
    io = module.allocate_io()
    for x in io.inputs.values():
        x.normal_(std=0.2)
    x = io.inputs["X"]
    if case.startswith("gemm"):
        expected = torch.relu((x.float() @ io.inputs["W"].float()).bfloat16())
    elif case == "reduction":
        expected = x.sum(dim=1)
    else:
        squared = x.float().square().to(x.dtype).float()
        expected = torch.sigmoid(squared).to(x.dtype)
    execution = module.prepare(io.inputs, out=io.output)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for _ in range(5):
        execution.run(stream=stream)
    execution.wait(timeout=30)
    atol, rtol = (0.125, 0.02) if x.dtype == torch.bfloat16 else (1e-5, 1e-5)
    torch.testing.assert_close(io.output, expected, atol=atol, rtol=rtol)
    graph = tl.Graph.capture(execution.run, executions=[execution], stream=stream)
    for _ in range(5):
        graph.replay(stream=stream)
    graph.wait(timeout=30)
    torch.testing.assert_close(io.output, expected, atol=atol, rtol=rtol)
    graph.close()
    execution.close()
    module.close()
    artifact.close()
