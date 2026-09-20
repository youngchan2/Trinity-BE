import concurrent.futures
import json
import shutil
import sys
from pathlib import Path
import pytest
import trinity_lowering as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from plan_syntax import index, load, store
from plans import gather


def identity():
    b = tl.PhysicalPlanBuilder()
    x = b.add_value("bf16", [128, 128], "external")
    b.bind_input('X with "quotes"', x)
    b.bind_input("alias", x)
    return b.build([], "Y", x)


def test_compiler_ownership_and_manifest(sdk):
    _, config = sdk
    source = tl.emit(identity())

    assert source.requirements.buffers[0].strides == (128, 1)

    artifact = tl.compile(source, config)
    path = artifact.directory
    data = json.loads((path / "manifest.json").read_text())

    assert data["host_abi_version"] == 1
    assert data["requirements"]["buffers"][0]["dtype"] == "bf16"
    assert artifact.diagnostics["exit_code"] == 0
    assert b"fixture compiler warning" in artifact.diagnostics["stderr"]
    with pytest.raises(TypeError):
        artifact.manifest["schema_version"] = 2

    artifact.close()

    assert not path.exists()
    with pytest.raises(RuntimeError):
        artifact.directory

    artifact = tl.compile(source, config)
    path = artifact.persist()
    artifact.close()

    assert path.exists() and (path / "diagnostics.json").is_file()

    shutil.rmtree(path)


def test_compile_failure_preserves_diagnostics(sdk):
    root, config = sdk
    (root / "mode").write_text("failure")

    with pytest.raises(tl.CompileError) as result:
        tl.compile(tl.emit(identity()), config)
    assert result.value.diagnostics["exit_code"] == 42
    assert b"fixture compiler stdout" in result.value.diagnostics["stdout"]


def test_concurrent_compilation(sdk):
    _, config = sdk
    source = tl.emit(identity())
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        artifacts = list(pool.map(lambda _: tl.compile(source, config), range(2)))

    assert artifacts[0].directory != artifacts[1].directory

    for a in artifacts:
        a.close()


def test_builder_checks_and_metadata():
    plan = identity()
    data = json.loads(plan.metadata_json())

    assert data["inputs"][0]["value"] == data["output"]["value"]
    with pytest.raises(ValueError):
        tl.gemm_implementations()[0].enumerate(["bf16"] * 3, [[128, 64], [65, 128], [128, 128]])

    shapes = [[128, 64], [64, 128], [128, 128]]
    implementations = tl.gemm_implementations()[0].enumerate(["bf16"] * 3, shapes)

    assert implementations

    b = tl.PhysicalPlanBuilder()
    ids = [b.add_value("bf16", s, "external") for s in shapes]
    b.bind_input("X", ids[0])
    b.bind_input("W", ids[1])
    a, w, y = ids
    ai = index("(tile m 128)", "(tile k 64)")
    bi = index("(tile k 64)", "(tile n 128)")
    ci = index("(tile m 128)", "(tile n 128)")
    rhs = f"(+ {load(y, shapes[2], ci)} (@ {load(a, shapes[0], ai)} {load(w, shapes[1], bi)}))"
    op = b.add_operation(
        ids[:2], ids[2:], expression=store(y, shapes[2], rhs, ci)
    )
    inner = b.add_loop("sequential", "k", 0, 64, 64, [op])
    cols = b.add_loop("parallel", "n", 0, 128, 128, [inner])
    rows = b.add_loop("parallel", "m", 0, 128, 128, [cols])

    assert (
        json.loads(b.build([rows], "Y", y).metadata_json())["statements"][0]["loop"]["kind"]
        == "parallel"
    )
    with pytest.raises(ValueError):
        b.add_value("bf16", [128, 128], "external")


def test_all_gather_bindings_on_both_axes():
    for world_size in [2, 3]:
        for axis in [0, 1]:
            data = json.loads(gather(axis=axis, world_size=world_size).metadata_json())
            assert data["world_size"] == world_size
            assert "loop" in data["statements"][0]
            shape = list(data["values"][data["output"]["value"]]["shape"])
            assert shape[axis] == 128 * world_size
