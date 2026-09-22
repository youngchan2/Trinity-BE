"""Numerical tests invoked by the ignored Rust precision integration test."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

root = Path(sys.argv[1])
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False


def module(name):
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


results = []


def reference(x, dtype, b=None):
    # Each named intermediate in these fixtures crosses a kernel boundary.
    # Match its declared storage rounding, then use FP32 compute/accumulation.
    e = x.float().exp().to(dtype)
    denom = e.float().sum(1, keepdim=True).to(dtype)
    numerator = e if b is None else (e.float() @ b.float()).to(dtype)
    return (numerator.float() / denom.float()).to(dtype)


for name, dtype in [("softmax_fp16", torch.float16), ("softmax_bf16", torch.bfloat16),
                    ("exp_dot_fp16", torch.float16), ("exp_dot_bf16", torch.bfloat16)]:
    x = torch.linspace(-2, 2, 256, device="cuda").reshape(16, 16).to(dtype)
    b = (torch.linspace(-1, 1, 256, device="cuda").reshape(16, 16).to(dtype)
         if name.startswith("exp_dot") else None)
    expected = reference(x, dtype, b)
    kernel = module(name)
    actual = kernel.forward(x) if b is None else kernel.forward(X=x, B=b)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.0001)
    results.append({"test": name, "all_finite": True,
                    "reference": "FP32 compute with declared storage rounding",
                    "max_abs_error": (actual.float() - expected.float()).abs().max().item()})

    # Preserve the old out-of-range cases as explicit limitations, rather than
    # deleting them or silently promoting the IR to obtain finite outputs.
    high = 30 if b is not None else 40
    x = torch.linspace(20, high, 256, device="cuda").reshape(16, 16).to(dtype)
    expected = reference(x, dtype, b)
    actual = kernel.forward(x) if b is None else kernel.forward(X=x, B=b)
    torch.cuda.synchronize()
    if dtype == torch.float16:
        assert not torch.isfinite(expected).all()
        assert not torch.isfinite(actual).all()
        torch.testing.assert_close(torch.isfinite(actual), torch.isfinite(expected))
        results.append({"test": name + "_raw_exp_overflow", "all_finite": False,
                        "status": "expected numerical limitation; not an accuracy pass"})
    else:
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.001)
        results.append({"test": name + "_large_exp", "all_finite": True,
                        "max_abs_error": (actual.float() - expected.float()).abs().max().item()})

x = torch.full((16, 16), 40000.0, device="cuda", dtype=torch.float16)
for name in ("cast_opmath", "register_opmath"):
    # 40000 + 40000 exceeds FP16, but /2 brings the result back into range.
    # An ordinary register assignment must not insert a half rounding here.
    actual = module(name).forward(x)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)
    results.append({"test": name, "all_finite": True, "max_abs_error": 0.0})

# BF16 inputs above the FP16 finite range must never be narrowed to FP16 at dot.
a = torch.full((16, 16), 100000.0, device="cuda", dtype=torch.bfloat16)
b = torch.full((16, 16), 0.001, device="cuda", dtype=torch.bfloat16)
actual = module("gemm_bf16").forward(a, b)
expected = (a.float() @ b.float()).to(torch.bfloat16)
torch.cuda.synchronize()
assert torch.isfinite(actual).all()
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
results.append({"test": "gemm_bf16_large_input", "all_finite": True, "max_abs_error": 0.0})

# The producer and consumer have different tile/lane layouts inside one CTA.
# An explicit barrier must publish the tiled global stores before the full read.
x = torch.linspace(0, 4, 16 * 1024, device="cuda").reshape(16, 1024).to(torch.bfloat16)
expected = x.float().exp().to(torch.bfloat16).float().sum(1).to(torch.bfloat16)
kernel = module("tiled_exp_sum")
max_error = 0.0
for _ in range(64):
    actual = kernel.forward(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    max_error = max(max_error, (actual.float() - expected.float()).abs().max().item())
results.append({"test": "tiled_store_full_read", "repetitions": 64,
                "all_finite": True, "max_abs_error": max_error})
(root / "gpu_results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
