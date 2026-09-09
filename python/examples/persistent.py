# torchrun --standalone --nproc-per-node=2 examples/persistent.py
import torch
import trinity_lowering as tl
from plans import gather

world = tl.World.from_torchrun()
artifact = tl.compile(tl.emit(gather(world_size=world.size)))
module = tl.load(artifact, world.device, world)

with torch.inference_mode():
    io = module.allocate_io()
    io.inputs["X"].fill_(world.rank)

    world.record_stream(io.inputs["X"])

    execution = module.prepare(io.inputs, out=io.output)

    execution.run()
    execution.run()
    execution.wait()

    execution.close()
    del io

module.close()
world.collect()
world.close()
artifact.close()
