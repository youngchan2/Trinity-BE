import concurrent.futures
import json
import shutil
import pytest
import trinity_lowering as tl


def identity():
    b = tl.PhysicalPlanBuilder()
    x = b.add_value("bf16", [128, 128], "external")
    b.bind_input('X with "quotes"', x)
    b.bind_input("alias", x)
    return b.finalize("Y", x)


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
    op = b.add_operation(ids[:2], ids[2:], implementations[0])
    b.add_statement([op])

    assert "trinity_abi" in tl.emit(b.finalize("Y", ids[2])).code
    with pytest.raises(ValueError):
        b.add_value("bf16", [128, 128], "external")


def test_all_gather_bindings_on_both_axes_and_world_validation():
    for definition in tl.all_gather_implementations():
        for axis in (0, 1):
            shapes = [[128, 128], [128, 128]]
            shapes[1][axis] *= 2
            instances = definition.enumerate("bf16", shapes, axis, 2)

            assert instances

            builder = tl.PhysicalPlanBuilder(2)
            x, y = [builder.add_value("bf16", s, "external") for s in shapes]
            builder.bind_input("X", x)
            op = builder.add_operation([x], [y], instances[0])
            builder.add_statement([op])
            req = tl.emit(builder.finalize("Y", y)).requirements

            assert req.world_size == 2 and req.workspace_symmetric
            assert req.nvls == ("one_shot_push_nbi" in definition.id)

            other = tl.PhysicalPlanBuilder(3)
            a, b = [other.add_value("bf16", s, "external") for s in shapes]

            with pytest.raises(ValueError, match="world size"):
                other.add_operation([a], [b], instances[0])
