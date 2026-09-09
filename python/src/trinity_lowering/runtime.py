"""Tensor bindings. Importing this module does not initialize CUDA."""

import atexit
import hashlib
import json
import threading
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MappingProxyType

from .compiler import CudaArtifact
from .errors import ResourceBusy
from .metadata import freeze, manifest

_native_module = None
_import_lock = threading.Lock()


def native():
    global _native_module
    with _import_lock:
        if _native_module is None:
            try:
                from . import _native
            except ImportError as e:
                raise RuntimeError(
                    "CUDA runtime extension unavailable; build with TRINITY_BUILD_CUDA=1 in the cu130 environment"
                ) from e

            import torch

            if (
                torch.__version__.split("+")[0] != _native.torch_version
                or torch.version.cuda != "13.0"
            ):
                raise RuntimeError("rebuild the native extension for torch 2.9.1+cu130")

            _native_module = _native
            atexit.register(_native.shutdown)

    return _native_module


def device_of(device):
    import torch

    d = torch.device(device)
    if d.type != "cuda":
        raise ValueError("a CUDA device is required")

    if d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())

    if torch.cuda.get_device_capability(d) != (9, 0):
        raise ValueError("this artifact requires a Hopper sm_90a device")

    return d


def stream_of(device, stream=None, wait_for=()):
    import torch

    if stream is None:
        stream = torch.cuda.current_stream(device)

    if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
        raise ValueError("stream must be a torch.cuda.Stream on the execution device")

    events = tuple(wait_for)
    if any(not isinstance(e, torch.cuda.Event) or e.device != device for e in events):
        raise ValueError("wait_for must contain recorded CUDA events on the execution device")

    for event in events:
        stream.wait_event(event)

    return stream


def tensor_check(tensor, b, device, world):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"binding {b.value} requires a Tensor")

    if (
        tensor.device != device
        or tensor.layout != torch.strided
        or tensor.dtype != torch.bfloat16
        or tuple(tensor.shape) != b.shape
        or tuple(tensor.stride()) != b.strides
    ):
        raise ValueError(f"binding {b.value}: dtype/shape/stride/device mismatch")

    if torch.is_grad_enabled() and tensor.requires_grad:
        raise ValueError("use no_grad or inference_mode for gradient Tensors")

    if (
        tensor.data_ptr() % b.alignment
        or tensor.untyped_storage().nbytes() - tensor.storage_offset() * 2 < b.bytes
    ):

        raise ValueError(f"binding {b.value}: alignment/storage mismatch")

    if b.symmetric and (world is None or not world._native.owns(tensor, b.bytes)):
        raise ValueError(f"binding {b.value} requires this world's symmetric allocation")


def resolve_bindings(req, inputs, out, device, world, allow_missing=False):
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping of binding names to Tensors")

    names = {name for b in req.buffers for name in b.input_names}
    if set(inputs) - names or (not allow_missing and set(inputs) != names):
        raise ValueError(f"expected exactly input names {sorted(names)}")

    bound = {}
    for b in req.buffers:
        candidates = [inputs[name] for name in b.input_names if name in inputs]
        if b.output_name is not None and out is not None:
            candidates.append(out)

        for tensor in candidates:
            tensor_check(tensor, b, device, world)
            if b.symmetric and req.nvls:
                world._native.check_multicast(tensor)

        if candidates:
            if any(t.data_ptr() != candidates[0].data_ptr() for t in candidates[1:]):
                raise ValueError("names referring to the same value must share one Tensor region")
            bound[b.value] = candidates[0]

    regions = [(t.data_ptr(), req.buffers[i].bytes) for i, t in bound.items()]
    for i, (address, length) in enumerate(regions):
        if any(address < other + size and other < address + length for other, size in regions[:i]):
            raise ValueError("different canonical values overlap")

    return bound


@dataclass(frozen=True)
class IOBuffers:
    inputs: Mapping
    output: object


def world_guarded(fn):
    @wraps(fn)
    def guarded(self, *args, **kwargs):
        world = self if hasattr(self, "_control_lock") else self.world
        with world._control_lock if world else nullcontext():
            try:
                return fn(self, *args, **kwargs)
            except Exception as error:
                if world and getattr(getattr(world, "_native", None), "poisoned", False):
                    world._fail(str(error))
                raise

    return guarded


def outside_capture(device):
    import torch

    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise ResourceBusy("load, prepare and Trinity allocation must finish before capture")

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise ResourceBusy("load, prepare and Trinity allocation must finish before capture")


class LoadedModule:
    def __init__(self, handle, data, req, device, world, fingerprint):
        self._native, self._requirements, self._device, self._world = handle, req, device, world
        self._manifest, self._fingerprint = freeze(data), fingerprint
        self._closed = False
        self._lock = threading.RLock()

    @property
    def requirements(self):
        return self._requirements

    @property
    def manifest(self):
        return self._manifest

    @property
    def device(self):
        return self._device

    @property
    def world(self):
        return self._world

    @property
    def fingerprint(self):
        return self._fingerprint

    def _open(self):
        if self._closed:
            raise RuntimeError("module is closed")
        if self.world:
            self.world._check()

    def _stage(self, name, fn, signature=None):
        if self.world:
            return self.world._stage(name, fn, signature)
        return fn()

    def _allocate(self, b):
        import torch

        if b.symmetric:
            tensor = self.world._native.allocate(b.bytes, b.alignment, list(b.shape), True)
            if self.requirements.nvls:
                self.world._native.check_multicast(tensor)
            return tensor

        # An overallocated byte storage supports requirements stronger than the allocator's alignment.
        owner = torch.empty(b.bytes + b.alignment - 1, dtype=torch.uint8, device=self.device)
        offset = (-owner.data_ptr()) % b.alignment
        return owner[offset : offset + b.bytes].view(torch.bfloat16).view(b.shape)

    @world_guarded
    def allocate_io(self, inputs=None):
        with self._lock:
            bound = self._stage(
                "allocate_io.validate",
                lambda: self._bindings({} if inputs is None else inputs, None, True),
            )
            missing = [
                b.value for b in self.requirements.buffers if b.external and b.value not in bound
            ]
            self._stage("allocate_io.order", lambda: None, [self.fingerprint, missing])

            for value in missing:
                b = self.requirements.buffers[value]
                bound[value] = self._stage(f"allocate_io.{value}", lambda b=b: self._allocate(b))

            return IOBuffers(
                MappingProxyType(
                    {
                        name: bound[b.value]
                        for b in self.requirements.buffers
                        for name in b.input_names
                    }
                ),
                next(
                    bound[b.value] for b in self.requirements.buffers if b.output_name is not None
                ),
            )

    def _bindings(self, inputs, out, allow_missing=False):
        self._open()
        outside_capture(self.device)
        if self.world:
            self.world._native.check_idle()

        return resolve_bindings(
            self.requirements, inputs, out, self.device, self.world, allow_missing
        )

    @world_guarded
    def prepare(self, inputs, out=None, workers=None):
        import torch

        with self._lock:
            bound = self._stage("prepare.validate", lambda: self._bindings(inputs, out))
            missing = [b.value for b in self.requirements.buffers if b.value not in bound]
            self._stage("prepare.order", lambda: None, [self.fingerprint, missing, workers])

            maximum = self._stage("prepare.module", self._native.prepare)
            if self.world:
                maximum = min(self.world._exchange("prepare.capacity", maximum))

            chosen = maximum if workers is None else workers

            def validate_workers():
                if (
                    type(chosen) is not int
                    or not self.requirements.minimum_workers <= chosen <= maximum
                ):
                    raise ValueError(
                        f"workers must be within [{self.requirements.minimum_workers}, {maximum}]"
                    )

            self._stage("prepare.workers", validate_workers)

            for value in missing:
                b = self.requirements.buffers[value]
                bound[value] = self._stage(f"prepare.buffer.{value}", lambda b=b: self._allocate(b))

            req = self.requirements
            workspace = None

            if req.workspace_symmetric:
                workspace = self._stage(
                    "prepare.workspace",
                    lambda: self.world._native.allocate(
                        req.workspace_bytes, req.workspace_alignment, [req.workspace_bytes], False
                    ),
                )

            st = torch.cuda.current_stream(self.device)
            specs = [
                native().BufferSpec(
                    b.value, b.bytes, b.alignment, *b.shape, *b.strides, b.symmetric
                )
                for b in req.buffers
            ]
            output = next(b.value for b in req.buffers if b.output_name is not None)
            handle = self._stage(
                "prepare.execution",
                lambda: native().Execution.create(
                    self._native,
                    specs,
                    [bound[b.value] for b in req.buffers],
                    workspace,
                    output,
                    chosen,
                    st.cuda_stream,
                ),
            )

            return PreparedExecution(
                self, handle, {n: bound[b.value] for b in req.buffers for n in b.input_names}
            )

    @world_guarded
    def close(self, wait=True):
        with self._lock:
            if not self._closed:
                self._native.close()  # Child owners always make module.close ResourceBusy.
                self._closed = True


def close_owner(handle, wait, timeout):
    """Return safe leases even when completion reports a semantic device error."""
    error = None

    if wait:
        try:
            handle.wait(timeout)
        except Exception as caught:
            error = caught

    try:
        handle.close(False)
    except Exception as cleanup:
        if error is not None:
            error.cleanup_error = cleanup
            raise error from cleanup

        raise

    if error is not None:
        raise error


class PreparedExecution:
    @property
    def world(self):
        return self.module.world

    def __init__(self, module, handle, inputs):
        self._module, self._native = module, handle
        self._inputs = MappingProxyType(inputs)
        self._closed = False

    @property
    def module(self):
        return self._module

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._native.output

    def _call(self, fn):
        if self._closed:
            raise RuntimeError("execution is closed")
        if self.module.world:
            return self.module.world._invoke(fn)
        return fn()

    def run(self, stream=None, wait_for=()):
        st = stream_of(self.module.device, stream, wait_for)
        return self._call(lambda: self._native.run(st.cuda_stream))

    def wait(self, timeout=None):
        if self._closed:
            return

        timeout = (
            self.module.world.timeout
            if timeout is None and self.module.world
            else (-1 if timeout is None else timeout)
        )
        return self._call(lambda: self._native.wait(timeout))

    @world_guarded
    def close(self, wait=True):
        if not self._closed:
            timeout = self.module.world.timeout if self.module.world else -1
            try:
                close_owner(self._native, wait, timeout)
            finally:
                if self._native.closed:
                    self._inputs = MappingProxyType({})
                    self._closed = True


def load(artifact_or_directory, device, world=None):
    def local_load():
        d = device_of(device)
        outside_capture(d)
        if world and d != world.device:
            raise ValueError("module must use its world's device")

        artifact = artifact_or_directory
        with artifact._lock if isinstance(artifact, CudaArtifact) else nullcontext():
            directory = artifact.directory if isinstance(artifact, CudaArtifact) else Path(artifact)
            data, req = manifest((directory / "manifest.json").read_text())

            if req.nvshmem != (world is not None) or (world and world.size != req.world_size):
                raise ValueError("artifact requires a matching world")

            fingerprint = hashlib.sha256(
                (directory / "source.cu").read_bytes()
                + json.dumps(data["requirements"], sort_keys=True).encode()
            ).hexdigest()

            handle = native().Module.load(
                str(directory / "program.so"),
                d.index,
                2 if req.nvshmem else 1,
                world._native if world else None,
            )

            try:
                if json.loads(handle.requirements_json) != data["requirements"]:
                    raise ValueError("artifact manifest differs from the loaded module")
            except BaseException:
                handle.close()
                raise

        return LoadedModule(handle, data, req, d, world, fingerprint)

    if world:
        module = world._stage("load", local_load)
        world._stage("load.plan", lambda: None, module.fingerprint)
        return module

    return local_load()
