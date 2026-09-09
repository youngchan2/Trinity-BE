"""Always run under torchrun plus an external process timeout."""

import gc
import sys
from pathlib import Path
import torch
import trinity_lowering as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from plans import gather

world = tl.World.from_torchrun(timeout=30)
with torch.inference_mode():
    for backend in ("peer_push", "peer_pull", "one_shot_push_nbi"):
        for axis in (0, 1):
            artifact = tl.compile(tl.emit(gather(backend, axis, world.size)))
            module = tl.load(artifact, world.device, world)
            io = module.allocate_io()
            x = io.inputs["X"]
            x.copy_(
                torch.arange(x.numel(), device=world.device).reshape(x.shape) % 64 + world.rank * 64
            )
            world.record_stream(x)

            execution = module.prepare(io.inputs, out=io.output)

            expected = torch.cat(
                [
                    (
                        (torch.arange(x.numel(), device=world.device).reshape(x.shape) % 64)
                        + r * 64
                    ).bfloat16()
                    for r in range(world.size)
                ],
                dim=axis,
            )

            for _ in range(10):
                execution.run()
            execution.wait(30)
            assert torch.equal(execution.output, expected)

            stream = torch.cuda.Stream(device=world.device)
            stream.wait_stream(torch.cuda.current_stream(world.device))
            graph = tl.Graph.capture(
                lambda: execution.run().clone(), executions=[execution], stream=stream
            )
            with torch.cuda.stream(stream):
                for _ in range(20):
                    graph.replay()
            graph.wait(30)
            assert torch.equal(graph.output, expected)

            graph.close()
            execution.close()
            module.close()
            artifact.close()

            view = x.view(-1)
            del x, io
            world.collect()
            try:
                world.close()
                raise AssertionError("live symmetric view must block close")
            except tl.ResourceBusy:
                pass

            del view
            gc.collect()
            world.collect()

world.close()
