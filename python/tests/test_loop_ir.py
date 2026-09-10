import importlib.util
import json
from pathlib import Path

import pytest
import trinity_lowering as tl

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/loop_ir"


def ffn_fixture(candidate="fused", split=4):
    """Use the same checked-in IR and reduced dimensions as the Rust tests."""
    filename = {"fused": "IR.v3.txt", "split-k": "IR.v3.split_k.txt"}[candidate]
    text = (FIXTURES / filename).read_text().replace("16384", "512").replace("4096", "256")
    metadata = json.loads((FIXTURES / "IR.v3.meta.json").read_text())
    symbols = dict(metadata["example_bindings"])
    symbols["__nsplit_fdba4cffaeb51ebe"] = split
    dtypes = {name: "fp32" if name == "attn_O3" else "bf16" for name in metadata["tensor_shapes"]}
    dtypes["__split_fdba4cffaeb51ebe"] = "fp32"
    return text, symbols, dtypes


@pytest.fixture
def ffn_gpu_example():
    """Load the optional enclosing-workspace GPU harness only for GPU tests."""
    for parent in Path(__file__).resolve().parents:
        path = parent / "examples/ffn_v3/run.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("ffn_loop_example", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip("requires the enclosing workspace's examples/ffn_v3/run.py GPU harness")


@pytest.mark.parametrize("candidate", ["fused", "split-k"])
@pytest.mark.parametrize("world_size", [1, 2])
def test_python_loop_reader_and_rank_three_metadata(candidate, world_size):
    text, symbols, dtypes = ffn_fixture(candidate, 4)
    (plan,) = tl.lower_loop_ir(text, symbols, dtypes, world_size=world_size)
    source = tl.emit(plan)
    assert source.requirements.world_size == world_size
    if candidate == "split-k":
        (scratch,) = [b for b in source.requirements.buffers if len(b.shape) == 3]
        assert scratch.dtype == "fp32"
        assert scratch.shape == (4, 16, 256)
        assert scratch.strides == (4096, 256, 1)


def test_reader_reports_missing_symbols():
    text, symbols, dtypes = ffn_fixture()
    del symbols["tile_k"]
    with pytest.raises(ValueError, match=r"byte \d+.*tile_k"):
        tl.lower_loop_ir(text, symbols, dtypes)


@pytest.mark.gpu
@pytest.mark.parametrize("size", ["reduced", "full"])
@pytest.mark.parametrize(
    "candidate,split", [("fused", 1), ("split-k", 1), ("split-k", 2), ("split-k", 4)]
)
def test_ffn_accuracy_repeated_execution_and_benchmark(
    size, candidate, split, tmp_path, ffn_gpu_example
):
    result = ffn_gpu_example.run_case(candidate, size, split)
    (tmp_path / "results.json").write_text(json.dumps(result, indent=2))
    assert result["gpu_validated"]
