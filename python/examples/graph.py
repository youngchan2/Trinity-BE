import torch
import trinity_lowering as tl
from plans import gemm

artifact = tl.compile(tl.emit(gemm()))
module = tl.load(artifact, device="cuda:0")

with torch.inference_mode():
    io = module.allocate_io()

    io.inputs["X"].normal_()
    io.inputs["W"].normal_()

    bias = torch.randn(128, 128, device=module.device, dtype=torch.bfloat16)

    execution = module.prepare(io.inputs, out=io.output)
    stream = torch.cuda.Stream(device=module.device)
    stream.wait_stream(torch.cuda.current_stream(module.device))

    def step():
        execution.run()
        return torch.relu(execution.output + bias)

    graph = tl.Graph.capture(step, executions=[execution], keepalive=[bias], stream=stream)

    with torch.cuda.stream(stream):
        graph.replay()
        saved = graph.output.clone()
        graph.replay()

    graph.wait()
    graph.close()
    execution.close()

module.close()
artifact.close()
