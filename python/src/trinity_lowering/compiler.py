import json
import threading
from pathlib import Path

from . import _compiler
from .metadata import freeze, manifest, requirements

PhysicalPlanBuilder = _compiler.PhysicalPlanBuilder
PhysicalPlan = _compiler.PhysicalPlan
Implementation = _compiler.Implementation
CompileConfig = _compiler.CompileConfig
CompileError = _compiler.CompileError

gemm_implementations = _compiler.gemm_implementations
all_gather_implementations = _compiler.all_gather_implementations


class CudaSource:
    def __init__(self, native):
        self._native = native
        self._requirements = requirements(json.loads(native.requirements_json()))

    @property
    def code(self):
        return self._native.code

    @property
    def requirements(self):
        return self._requirements


class CudaArtifact:
    """Wrap a Rust CUDA artifact, retaining its PyO3 owner in ``_handle``."""

    def __init__(self, handle):
        self._handle = handle
        self._lock = threading.RLock()
        data, self._requirements = manifest(handle.manifest_json())
        self._manifest = freeze(data)

    @property
    def requirements(self):
        return self._requirements

    @property
    def manifest(self):
        return self._manifest

    @property
    def directory(self):
        with self._lock:
            return Path(self._handle.directory)

    @property
    def artifact_path(self):
        with self._lock:
            return Path(self._handle.artifact_path)

    @property
    def diagnostics(self):
        with self._lock:
            return freeze(self._handle.diagnostics)

    @property
    def code(self):
        with self._lock:
            return self._handle.code

    def close(self):
        with self._lock:
            self._handle.close()

    def persist(self):
        with self._lock:
            return Path(self._handle.persist())


def emit(plan):
    return CudaSource(_compiler.emit(plan))


def compile(source, config=None):
    return CudaArtifact(_compiler.compile(source._native, config))
