"""CUDA-independent artifact validation. The native loader also checks embedded metadata."""

import json
from dataclasses import dataclass
from types import MappingProxyType


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)

    return value


@dataclass(frozen=True)
class BufferRequirement:
    value: int
    input_names: tuple[str, ...]
    output_name: str | None
    shape: tuple[int, int]
    dtype: str
    strides: tuple[int, int]
    bytes: int
    alignment: int
    external: bool
    symmetric: bool


@dataclass(frozen=True)
class Requirements:
    target: str
    cuda_arch: str
    world_size: int
    buffers: tuple[BufferRequirement, ...]
    workspace_bytes: int
    workspace_alignment: int
    workspace_symmetric: bool
    cooperative_launch: bool
    shared_memory_bytes: int
    block_threads: int
    minimum_workers: int
    nvshmem: bool
    nvls: bool


def _positive(value, name):
    if type(value) is not int or value <= 0 or value > 2**63 - 1:
        raise ValueError(f"invalid {name}")


def requirements(data):
    if data["target"] != "hopper" or data["cuda_arch"] != "sm_90a":
        raise ValueError("unsupported artifact target")

    _positive(data["world_size"], "world_size")

    names = set()
    outputs = 0
    buffers = []
    for index, raw in enumerate(data["buffers"]):
        b = BufferRequirement(
            **{
                **raw,
                "shape": tuple(raw["shape"]),
                "strides": tuple(raw["strides"]),
                "input_names": tuple(raw["input_names"]),
            }
        )

        if b.value != index or b.dtype != "bf16" or len(b.shape) != 2:
            raise ValueError("invalid canonical binding/dtype/shape")
        for extent in b.shape:
            _positive(extent, "shape extent")
        _positive(b.alignment, "alignment")
        if b.alignment & (b.alignment - 1) or b.strides != (b.shape[1], 1):
            raise ValueError("invalid alignment/strides")
        if b.bytes != b.shape[0] * b.shape[1] * 2 or b.bytes > (2**31 - 1) * 2:
            raise ValueError("invalid buffer bytes")
        if bool(b.input_names or b.output_name is not None) != b.external:
            raise ValueError("invalid external binding")

        for name in b.input_names:
            if not isinstance(name, str) or name in names:
                raise ValueError("invalid or duplicate input name")
            names.add(name)
        if b.output_name is not None:
            if not isinstance(b.output_name, str):
                raise ValueError("invalid output name")
            outputs += 1

        if b.symmetric and data["world_size"] == 1:
            raise ValueError("streamed allocation cannot be symmetric")
        buffers.append(b)

    if outputs != 1:
        raise ValueError("exactly one output binding required")

    persistent = data["world_size"] > 1
    if data["nvshmem"] != persistent or data["workspace_symmetric"] != persistent:
        raise ValueError("inconsistent world/workspace mode")
    _positive(data["workspace_alignment"], "workspace alignment")
    _positive(data["minimum_workers"], "minimum workers")
    if persistent:
        _positive(data["workspace_bytes"], "workspace bytes")
        if not data["cooperative_launch"]:
            raise ValueError("persistent launch must be cooperative")
    elif data["workspace_bytes"] != 0 or data["cooperative_launch"]:
        raise ValueError("streamed execution cannot have a control workspace")

    return Requirements(**{**data, "buffers": tuple(buffers)})


def manifest(text):
    data = json.loads(text)
    if data.get("schema_version") != 1 or data.get("host_abi_version") != 1:
        raise ValueError("unsupported artifact/ABI version; recompile the artifact")
    if data.get("library") != "program.so" or data.get("source") != "source.cu":
        raise ValueError("invalid artifact bundle paths")

    req = requirements(data["requirements"])
    if data.get("execution") != ("persistent" if req.nvshmem else "streamed"):
        raise ValueError("inconsistent execution mode")
    if data["toolchain"]["cuda_major"] != 13:
        raise ValueError("runtime currently supports CUDA 13 artifacts")

    return data, req
