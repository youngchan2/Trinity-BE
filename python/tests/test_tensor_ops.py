import json
from pathlib import Path
import sys

import pytest
import torch
import trinity_lowering as tl
from trinity_lowering.metadata import requirements

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tensor_ops import Builder, gated_silu, normalization, pointwise, silu
from plans import gemm_relu


def definition(name):
    return next(d for d in tl.pointwise_implementations() if d.id == f"cuda.{name}")


@pytest.mark.parametrize("shape", [(1,), (127,), (128,), (129,), (16, 4096), (16, 16384), (3, 257)])
def test_primitive_plans_and_metadata(shape):
    for name in ("add", "mul", "div", "scalar_div", "square", "sqrt", "sigmoid", "relu"):
        dtypes = ["bf16", "fp32", "bf16"] if name in ("add", "mul", "div") else ["bf16", "fp32"]
        attrs = {"scalar": 4096.0} if name == "scalar_div" else {}
        for world in (1, 2):
            source = tl.emit(pointwise(name, shape, dtypes, world, **attrs))
            buffers = source.requirements.buffers
            assert all(b.shape == shape for b in buffers)
            for buffer in buffers:
                assert buffer.bytes == torch.Size(shape).numel() * (
                    2 if buffer.dtype == "bf16" else 4
                )
            assert source.requirements.nvshmem == (world == 2)


def test_normalization_and_silu_preserve_precision_boundaries():
    source = tl.emit(normalization())
    vectors = [b for b in source.requirements.buffers if b.shape == (16,)]
    assert len(vectors) == 3
    assert all(b.dtype == "fp32" and b.bytes == 64 and b.strides == (1,) for b in vectors)
    assert source.requirements.buffers[-1].dtype == "bf16"
    for plan in (silu(), gated_silu()):
        buffers = tl.emit(plan).requirements.buffers
        assert sum(b.dtype == "fp32" for b in buffers) == 1
        assert buffers[-1].dtype == "bf16"


def test_graph_example_lowers_gemm_bias_and_relu():
    plan = gemm_relu()
    metadata = json.loads(plan.metadata_json())
    assert {b["name"] for b in metadata["inputs"]} == {"X", "W", "bias"}
    assert len(metadata["operations"]) == len(metadata["actions"]) == 3
    source = tl.emit(plan)
    assert sum(not b.external for b in source.requirements.buffers) == 2
    assert all(b.dtype == "bf16" for b in source.requirements.buffers)


def test_invalid_operands_and_arity_are_rejected():
    with pytest.raises(ValueError):
        definition("add").enumerate(["fp32"] * 3, [[16], [17], [16]])
    with pytest.raises(ValueError):
        definition("square").enumerate(["fp32"] * 3, [[16]] * 3)
    with pytest.raises(ValueError):
        definition("scalar_div").enumerate(["fp32"] * 2, [[16]] * 2)
    with pytest.raises(ValueError):
        definition("sqrt").enumerate(["fp32"] * 2, [[16]] * 2, scalar=2.0)
    with pytest.raises(ValueError):
        tl.reduce_sum_implementations()[0].enumerate(["fp32"] * 2, [[16, 32], [32]])
    with pytest.raises(ValueError):
        tl.broadcast_implementations()[0].enumerate(["fp32", "bf16"], [[16], [16, 32]])
    assert tl.reduce_sum_implementations()[0].enumerate(["bf16"] * 2, [[16, 32], [16]]) == []
    assert (
        tl.reduce_sum_implementations()[0].enumerate(["fp32"] * 2, [[16, 32], [32]], axis=0) == []
    )
    instance = definition("sqrt").enumerate(["fp32"] * 2, [[16]] * 2)[0]
    b = tl.PhysicalPlanBuilder()
    x = b.add_value("fp32", [16], "external")
    y = b.add_value("fp32", [16], "external")
    with pytest.raises(ValueError):
        b.add_operation([x, x], [y], instance)
    wrong = b.add_value("bf16", [16], "external")
    with pytest.raises(ValueError):
        b.add_operation([x], [wrong], instance)


def test_fp32_metadata_rejects_wrong_bytes_stride_rank():
    data = json.loads(
        tl.emit(pointwise("square", [129], ["bf16", "fp32"]))._native.requirements_json()
    )
    assert requirements(data).buffers[-1].bytes == 516
    for update in (
        {"bytes": 258},
        {"strides": [2]},
        {"shape": [1, 1, 129], "strides": [129, 129, 1]},
        {"dtype": "fp64"},
    ):
        corrupted = json.loads(json.dumps(data))
        corrupted["buffers"][-1].update(update)
        with pytest.raises(ValueError):
            requirements(corrupted)


def assert_result(actual, reference):
    torch.testing.assert_close(torch.isnan(actual), torch.isnan(reference))
    torch.testing.assert_close(torch.isposinf(actual), torch.isposinf(reference))
    torch.testing.assert_close(torch.isneginf(actual), torch.isneginf(reference))
    finite = torch.isfinite(reference)
    if actual.dtype == torch.bfloat16:

        def ordered(t):
            bits = t.view(torch.int16).to(torch.int32)
            return torch.where(bits < 0, -32768 - bits, bits)

        distance = (ordered(actual) - ordered(reference)).abs()
        assert torch.all(distance[finite] <= 1), distance[finite].max().item()
    else:
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1,), (127,), (128,), (129,), (3, 257), (16, 4096)])
@torch.inference_mode()
def test_gpu_pointwise(shape):
    for name in ("add", "mul", "div", "scalar_div", "square", "sqrt", "sigmoid", "relu"):
        binary = name in ("add", "mul", "div")
        dtypes = ["bf16", "fp32", "bf16"] if binary else ["bf16", "fp32"]
        attrs = {"scalar": 4096.0} if name == "scalar_div" else {}
        artifact = tl.compile(tl.emit(pointwise(name, shape, dtypes, **attrs)))
        module = tl.load(artifact, "cuda:0")
        x = (
            torch.linspace(-8, 8, torch.Size(shape).numel(), device="cuda")
            .reshape(shape)
            .bfloat16()
        )
        y = torch.randn(shape, device="cuda").abs() + 0.25
        inputs = {"X0": x, **({"X1": y} if binary else {})}
        reference = {
            "add": lambda: x.float() + y,
            "mul": lambda: x.float() * y,
            "div": lambda: x.float() / y,
            "scalar_div": lambda: x.float() / 4096.0,
            "square": lambda: x.float().square(),
            "sqrt": lambda: x.float().sqrt(),
            "sigmoid": lambda: 1.0 / (1.0 + torch.exp(-x.float())),
            "relu": lambda: torch.relu(x.float()),
        }[name]()
        if binary:
            reference = reference.bfloat16()
        execution = module.prepare(inputs)
        for _ in range(3):
            execution.run()
        execution.wait(30)
        assert_result(execution.output, reference)
        execution.close()
        module.close()
        artifact.close()


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(129,), (3, 257)])
@pytest.mark.parametrize("input_dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("output_dtype", ["bf16", "fp32"])
@torch.inference_mode()
def test_gpu_relu_special_values_and_casts(shape, input_dtype, output_dtype):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}
    values = torch.tensor(
        [-float("inf"), -2.0, -1e-8, -0.0, 0.0, 1e-8, 2.0, float("inf"), float("nan")],
        device="cuda",
        dtype=dtype[input_dtype],
    )
    count = torch.Size(shape).numel()
    x = values.repeat((count + values.numel() - 1) // values.numel())[:count].reshape(shape)
    reference = torch.relu(x.float()).to(dtype[output_dtype])
    artifact = tl.compile(tl.emit(pointwise("relu", shape, [input_dtype, output_dtype])))
    module = tl.load(artifact, "cuda:0")
    execution = module.prepare({"X0": x})
    execution.run()
    execution.wait(30)
    torch.testing.assert_close(execution.output, reference, rtol=0, atol=0, equal_nan=True)
    zeros = reference == 0
    assert torch.equal(torch.signbit(execution.output[zeros]), torch.signbit(reference[zeros]))
    execution.close()
    module.close()
    artifact.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "name,shape",
    [
        ("normalization", (16, 4096)),
        ("normalization", (3, 129)),
        ("silu", (16, 16384)),
        ("gate", (16, 16384)),
    ],
)
@torch.inference_mode()
def test_gpu_compositions_repeat_and_graph(name, shape):
    plan = {"normalization": normalization, "silu": silu, "gate": gated_silu}[name](shape)
    artifact = tl.compile(tl.emit(plan))
    module = tl.load(artifact, "cuda:0")
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()  # epsilon-free normalization must retain 0/0 NaNs
    if name == "gate":
        a = torch.randn_like(x)
        inputs = {"A": a, "B": x}
    else:
        inputs = {"X": x}

    def reference():
        v = x.float()
        if name == "normalization":
            return (v / torch.sqrt(v.square().sum(1, keepdim=True) / shape[1])).bfloat16()
        activation = (v * (1.0 / (1.0 + torch.exp(-v)))).bfloat16()
        return (a.float() * activation.float()).bfloat16() if name == "gate" else activation

    execution = module.prepare(inputs)
    for _ in range(5):
        execution.run()
    execution.wait(30)
    assert_result(execution.output, reference())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = tl.Graph.capture(lambda: execution.run().clone(), executions=[execution], stream=stream)
    for _ in range(3):
        with torch.cuda.stream(stream):
            x.mul_(0.5)
            graph.replay()
        graph.wait(30)
        assert_result(graph.output, reference())
    graph.close()
    execution.close()
    module.close()
    artifact.close()


@pytest.mark.gpu
@torch.inference_mode()
def test_gpu_reduce_broadcast_and_special_values():
    b = Builder()
    x = b.input("X", "fp32", [3, 129])
    result = b.broadcast(b.reduce(x), 129, "external")
    artifact = tl.compile(tl.emit(b.finish(result)))
    module = tl.load(artifact, "cuda:0")
    x = torch.randn(3, 129, device="cuda")
    x[0, 0] = float("inf")
    x[1, 0] = float("-inf")
    x[2, 0] = float("nan")
    execution = module.prepare({"X": x})
    execution.run()
    execution.wait(30)
    assert_result(execution.output, x.sum(1, keepdim=True).expand_as(x))
    execution.close()
    module.close()
    artifact.close()
