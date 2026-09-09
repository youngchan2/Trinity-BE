"""Explicit collective ownership for one torchrun node. Destructors never finalize."""

import datetime
import os
import socket
import threading
import time
from pathlib import Path

from .errors import DistributedFailure, ResourceBusy
from .runtime import device_of, native, stream_of, world_guarded


class World:
    @property
    def rank(self):
        return self._rank

    @property
    def size(self):
        return self._size

    @property
    def device(self):
        return self._device

    @property
    def timeout(self):
        return self._timeout

    @classmethod
    def from_torchrun(cls, *, nvshmem_root=None, timeout=300):
        import torch
        import torch.distributed as dist

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        rank, size, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        if size < 2:
            raise ValueError("persistent execution requires at least two ranks")

        owns_default = not dist.is_initialized()
        duration = datetime.timedelta(seconds=timeout)
        if owns_default:
            dist.init_process_group("gloo", timeout=duration)
            group = dist.group.WORLD
        else:
            if dist.get_rank() != rank or dist.get_world_size() != size:
                raise ValueError("torchrun and existing process group disagree")
            group = dist.new_group(backend="gloo", timeout=duration)

        self = cls.__new__(cls)
        self._rank, self._size, self._timeout = rank, size, timeout
        self._group, self._owns_default = group, owns_default
        self._sequence, self._closed, self._failure = 0, False, None
        self._control_lock = threading.RLock()
        self._native = None
        self._stop = threading.Event()

        info = self._exchange("topology", [socket.gethostname(), local])
        if len({host for host, _ in info}) != 1 or sorted(i for _, i in info) != list(range(size)):
            raise ValueError("one node and distinct LOCAL_RANK devices are required")
        root = Path(nvshmem_root or os.environ.get("NVSHMEM_HOME", "/opt/nvshmem"))

        def create():
            self._device = device_of(torch.device("cuda", local))
            if not native().nvshmem_enabled:
                raise RuntimeError("native extension needs NVSHMEM_HOME at build time")
            return native().World.create(str(root / "lib/libnvshmem_host.so"), local, rank, size)

        self._native = self._stage("world.create", create)
        uid = self._stage("world.uid", lambda: self._native.unique_id() if rank == 0 else None)
        payload = [uid]
        dist.broadcast_object_list(payload, src=0, group=group)

        # A separate store permits failure notification without per-submission collectives.
        from torch.distributed.distributed_c10d import _get_default_store

        self._store = dist.PrefixStore("trinity-native-world-1/", _get_default_store())
        self._store.set(f"alive/{rank}", "0")
        self._monitor = threading.Thread(target=self._watch, name="trinity-control", daemon=True)
        self._monitor.start()

        self._stage("world.initialize", lambda: self._native.initialize(payload[0]))

        return self

    def _check(self):
        if self._closed:
            raise RuntimeError("world is closed")
        if self._failure:
            raise DistributedFailure(self._failure)
        if self._native:
            self._native.healthy()

    def _exchange(self, stage, value):
        import torch.distributed as dist

        with self._control_lock:
            result = [None] * self.size
            try:
                dist.all_gather_object(result, [self._sequence, stage, value], group=self._group)
            except BaseException as error:
                self._fail(f"control group failed at {stage}: {error}")
                raise DistributedFailure(self._failure) from error

            self._sequence += 1
            if any(item[:2] != result[0][:2] for item in result):
                self._fail(f"rank collective order mismatch at {stage}")
                raise DistributedFailure(self._failure)

            return [item[2] for item in result]

    def _stage(self, stage, fn, signature=None):
        if signature is not None:
            signatures = self._exchange(stage + ".signature", signature)
            if any(s != signatures[0] for s in signatures):
                self._fail(f"rank requirements/order mismatch at {stage}")
                raise DistributedFailure(self._failure)

        value, error = None, None
        try:
            value = fn()
        except Exception as e:
            error = e

        fatal = bool(self._native and self._native.poisoned)
        if error is not None and (
            getattr(error, "submitted", False) or getattr(error, "cleanup_code", 0)
        ):
            fatal = True

        failures = self._exchange(
            stage + ".result", None if error is None else [type(error).__name__, str(error), fatal]
        )
        if any(f is not None for f in failures):
            # Preflight failures and live leases are recoverable. Native submission/transport failures poison there.
            message = f"{stage}: rank failures {failures}"
            if any(f is not None and f[2] for f in failures):
                self._fail(message)
            if all(f is None or f[0] == "ResourceBusy" for f in failures):
                raise ResourceBusy(message) from error
            raise DistributedFailure(message) from error

        return value

    def _fail(self, reason):
        if self._failure is None:
            self._failure = str(reason)
            if self._native:
                self._native.poison(self._failure)
            if hasattr(self, "_store"):
                try:
                    self._store.set(f"error/{self.rank}", self._failure)
                except Exception:
                    pass

    @world_guarded
    def _invoke(self, fn):
        self._check()
        try:
            return fn()
        except (ValueError, TypeError, ResourceBusy):
            raise
        except Exception as e:
            self._fail(str(e))
            raise

    def _watch(self):
        seen = {rank: (None, time.monotonic()) for rank in range(self.size)}
        tick = 0
        while not self._stop.wait(min(1.0, self.timeout / 10)):
            try:
                tick += 1
                self._store.set(f"alive/{self.rank}", str(tick))
                for rank in range(self.size):
                    if self._store.check([f"error/{rank}"]):
                        self._fail(self._store.get(f"error/{rank}").decode())
                        return

                    if self._store.check([f"alive/{rank}"]):
                        value = self._store.get(f"alive/{rank}")
                        if value != seen[rank][0]:
                            seen[rank] = (value, time.monotonic())

                    if time.monotonic() - seen[rank][1] > self.timeout:
                        self._fail(
                            f"rank {rank} control heartbeat timeout; terminate the torchrun job"
                        )
                        return
            except Exception as e:
                self._fail(f"control monitor failed: {e}")
                return

    def record_stream(self, tensor, stream=None):
        st = stream_of(self.device, stream)
        self._invoke(lambda: self._native.record_stream(tensor, st.cuda_stream))

    @world_guarded
    def collect(self):
        if self._closed:
            raise RuntimeError("world is closed")

        local = self._stage("collect.snapshot", self._native.collectable)
        states = self._exchange("collect.leases", local)
        ids = [i for i, _ in states[0]]
        if any([i for i, _ in s] != ids for s in states):
            self._fail("symmetric allocation registries diverged")
            raise DistributedFailure(self._failure)

        ready = [i for index, i in enumerate(ids) if all(s[index][1] for s in states)]
        self._stage("collect.free", lambda: self._native.collect(ready), ready)

        return len(ready)

    @world_guarded
    def close(self):
        import torch.distributed as dist

        if self._closed:
            return
        self.collect()

        def validate():
            if self._native.module_count or self._native.collectable():
                raise ResourceBusy(
                    "world has live modules, Tensor/view leases, or pending stream consumers"
                )

        self._stage("world.close.validate", validate)
        self._stage("world.finalize", self._native.close)

        self._stop.set()
        self._monitor.join(timeout=2)
        dist.destroy_process_group(self._group)
        self._closed = True
