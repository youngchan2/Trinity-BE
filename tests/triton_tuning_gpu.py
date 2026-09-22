"""Exercise every tuning candidate, numerical tails, and repeated input mutation."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
torch.set_num_threads(1)
torch.manual_seed(123)

def module(name):
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def config(c):
    return {**c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}

g = module("gemm")
a = torch.randn((64, 128), device="cuda", dtype=torch.float16) * .1
b = torch.randn((128, 96), device="cuda", dtype=torch.float16) * .1
expected = (a.float() @ b.float()).half()
y = g.forward(a, b)
torch.cuda.synchronize()
torch.testing.assert_close(y, expected, rtol=.002, atol=.0005)
best = g.kernel_0.best_config
# Check every emitted candidate, not only the winning tile. The wrapper reads
# META tile sizes from each config when building its grid.
for c in g.KERNEL_0_CONFIGS:
    g.kernel_0.configs = [c]
    actual = g.forward(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=.002, atol=.0005)

m = module("mutate")
x = torch.zeros(256, device="cuda", dtype=torch.float16)
m.forward(x)
torch.cuda.synchronize()
torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)
m.forward(x)
torch.cuda.synchronize()
torch.testing.assert_close(x, torch.full_like(x, 2), rtol=0, atol=0)
split = module("split_sum")
x_split = torch.randn((128, 16), device="cuda", dtype=torch.float16) * .1
expected_split = x_split.float().sum(0).half()
actual = split.forward(x_split)
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected_split, rtol=.002, atol=.001)
split_best = split.kernel_0.best_config
for c in split.KERNEL_0_CONFIGS:
    # Preserve the mloop whole-chunk contract for the split count and serial tile.
    ns, tile = c.kwargs["META_num_splits"], c.kwargs["META_tile_k"]
    if ns != 1 and 128 % (ns * tile):
        continue
    split.kernel_0.configs = [c]
    actual = split.forward(x_split)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected_split, rtol=.002, atol=.001)

results = {"split_sum": {"all_legal_candidates_correct": True, "best_config": config(split_best)}, "gemm": {"candidates": len(g.KERNEL_0_CONFIGS), "all_candidates_correct": True,
                    "best_config": config(best), "timings_ms": [{"config": config(c), "ms": v} for c,v in g.kernel_0.configs_timings.items()]},
           "mutation": {"candidates": len(m.KERNEL_0_CONFIGS), "cold_and_cached_calls_correct": True,
                        "best_config": config(m.kernel_0.best_config)}}
(root / "gpu_results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({k: {a:b for a,b in v.items() if a != "timings_ms"} for k,v in results.items()}, indent=2))
