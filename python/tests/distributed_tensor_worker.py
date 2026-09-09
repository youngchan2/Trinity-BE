"""Rank-local FP32 statistics and communication consumers, run under torchrun."""

import gc
from pathlib import Path
import sys

import torch
import trinity_lowering as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tensor_ops import gather_sum_squares, normalization, silu
from test_tensor_ops import assert_result


def run_case(world, name, plan):
    artifact = tl.compile(tl.emit(plan))
    module = tl.load(artifact, world.device, world)
    io = module.allocate_io()
    x = io.inputs["X"]
    values = torch.arange(x.numel(), device=world.device).reshape(x.shape) % 13 - 6
    x.copy_(values + world.rank)
    world.record_stream(x)

    def reference():
        v = x.float()
        if name == "normalization":
            return (v / torch.sqrt(v.square().mean(1, keepdim=True))).bfloat16()
        if name == "silu":
            return (v / (1.0 + torch.exp(-v))).bfloat16()
        return (
            torch.cat([(values + r).bfloat16().float() for r in range(world.size)], 1)
            .square()
            .sum(1)
        )

    execution = module.prepare(io.inputs, out=io.output)
    for _ in range(5):
        execution.run()
    execution.wait(30)
    assert_result(execution.output, reference())
    stream = torch.cuda.Stream(device=world.device)
    stream.wait_stream(torch.cuda.current_stream(world.device))
    graph = tl.Graph.capture(lambda: execution.run().clone(), executions=[execution], stream=stream)
    with torch.cuda.stream(stream):
        for _ in range(5):
            graph.replay()
    graph.wait(30)
    assert_result(graph.output, reference())
    graph.close()
    execution.close()
    module.close()
    artifact.close()


world = tl.World.from_torchrun(timeout=30)
with torch.inference_mode():
    for name, plan in [
        ("normalization", normalization((16, 4096), world.size)),
        ("silu", silu((16, 16384), world.size)),
        ("gather", gather_sum_squares(world.size)),
    ]:
        run_case(world, name, plan)
        gc.collect()
        world.collect()

    # Directly exercise typed symmetric FP32 allocation and its Tensor lease.
    statistic = world._native.allocate(16 * 4, 16, [16], "fp32")
    assert statistic.shape == (16,) and statistic.dtype == torch.float32
    assert world._native.owns(statistic, 64)
    del statistic
    gc.collect()
    world.collect()
world.close()
