import json
from dataclasses import FrozenInstanceError
import pytest
import trinity_lowering as tl
from trinity_lowering.metadata import requirements
from test_compiler import identity


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(world_size=0),
        lambda d: d.update(target="ampere"),
        lambda d: d["buffers"][0].update(dtype="fp32"),
        lambda d: d["buffers"][0].update(strides=[1, 128]),
        lambda d: d["buffers"][0].update(bytes=1),
        lambda d: d["buffers"][0].update(alignment=3),
        lambda d: d["buffers"][0].update(symmetric=True),
    ],
)
def test_requirements_reject_corruption(mutation):
    source = tl.emit(identity())
    data = json.loads(source._native.requirements_json())
    mutation(data)

    with pytest.raises(ValueError):
        requirements(data)


def test_metadata_immutable():
    req = tl.emit(identity()).requirements

    with pytest.raises(FrozenInstanceError):
        req.buffers[0].bytes = 1


def test_loaded_metadata_cannot_be_rebound():
    from trinity_lowering.runtime import LoadedModule

    req = tl.emit(identity()).requirements
    module = LoadedModule(None, {"schema_version": 1}, req, None, None, "fingerprint")
    for name in ("requirements", "manifest", "device", "world", "fingerprint"):
        with pytest.raises(AttributeError):
            setattr(module, name, None)
