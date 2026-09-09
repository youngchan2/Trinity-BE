import torch
import trinity_lowering as tl
from plans import gemm_relu

artifact = tl.compile(tl.emit(gemm_relu()))
module = tl.load(artifact, device="cuda:0")

with torch.inference_mode():
    io = module.allocate_io()

    io.inputs["X"].normal_()
    io.inputs["W"].normal_()
    io.inputs["bias"].normal_()

    execution = module.prepare(io.inputs, out=io.output)
    stream = torch.cuda.Stream(device=module.device)
    stream.wait_stream(torch.cuda.current_stream(module.device))

    graph = tl.Graph.capture(execution.run, executions=[execution], stream=stream)

    with torch.cuda.stream(stream):
        graph.replay()
        saved = graph.output.clone()
        graph.replay()

    graph.wait()
    graph.close()
    execution.close()

module.close()
artifact.close()
