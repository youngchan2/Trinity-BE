import importlib.util
from pathlib import Path

import pytest
import trinity_lowering as tl

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("ffn_loop_example", ROOT / "examples/ffn_v3/run.py")
ffn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ffn)


@pytest.mark.parametrize("candidate", ["fused", "split-k"])
@pytest.mark.parametrize("world_size", [1, 2])
def test_python_loop_reader_and_rank_three_metadata(candidate, world_size):
    text, symbols, dtypes, _, _ = ffn.fixture(candidate, "reduced", 4)
    (plan,) = tl.lower_ir(text, symbols, dtypes, world_size=world_size)
    source = tl.emit(plan)
    assert source.requirements.world_size == world_size
    if candidate == "split-k":
        (scratch,) = [b for b in source.requirements.buffers if len(b.shape) == 3]
        assert scratch.dtype == "fp32"
        assert scratch.shape == (4, 16, 256)
        assert scratch.strides == (4096, 256, 1)


def test_reader_reports_missing_symbols():
    text, symbols, dtypes, _, _ = ffn.fixture()
    del symbols["tile_k"]
    with pytest.raises(ValueError, match=r"byte \d+.*tile_k"):
        tl.lower_ir(text, symbols, dtypes)


@pytest.mark.gpu
@pytest.mark.parametrize("size", ["reduced", "full"])
@pytest.mark.parametrize(
    "candidate,split", [("fused", 1), ("split-k", 1), ("split-k", 2), ("split-k", 4)]
)
def test_ffn_accuracy_repeated_execution_and_benchmark(size, candidate, split, tmp_path):
    import json

    result = ffn.run_case(candidate, size, split)
    (tmp_path / "results.json").write_text(json.dumps(result, indent=2))
    assert result["gpu_validated"]
