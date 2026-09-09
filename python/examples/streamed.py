import torch
import trinity_lowering as tl
from plans import gemm

artifact = tl.compile(tl.emit(gemm()))
module = tl.load(artifact, device="cuda:0")
artifact.close()  # native loader owns a private library copy

with torch.inference_mode():
    x = torch.randn(128, 64, dtype=torch.bfloat16, device=module.device)
    weight = torch.randn(64, 128, dtype=torch.bfloat16, device=module.device)
    bias = torch.randn(128, 128, dtype=torch.bfloat16, device=module.device)

    execution = module.prepare({"X": x, "W": weight, "bias": bias})

    stream = torch.cuda.Stream(device=module.device)
    stream.wait_stream(torch.cuda.current_stream(module.device))

    with torch.cuda.stream(stream):
        execution.run()
        saved = execution.output.clone()
        execution.run()

    execution.wait()
    torch.testing.assert_close(
        saved, (x.float() @ weight.float()).bfloat16() + bias, atol=0.125, rtol=0.02
    )

execution.close()
module.close()
