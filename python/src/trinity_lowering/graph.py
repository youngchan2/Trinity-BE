"""A CUDA Graph owner that retains Trinity modules and static Tensor bindings."""

from .runtime import close_owner, native, stream_of, world_guarded
from .errors import ResourceBusy
from contextlib import nullcontext


class Graph:
    @classmethod
    def capture(cls, fn, *, executions, keepalive=(), stream, warmup=3):
        executions = tuple(executions)
        world = next((e.module.world for e in executions if e.module.world), None)
        with world._control_lock if world else nullcontext():
            return cls._capture(
                fn, executions=executions, keepalive=keepalive, stream=stream, warmup=warmup
            )

    @classmethod
    def _capture(cls, fn, *, executions, keepalive=(), stream, warmup=3):
        import torch

        executions = tuple(executions)
        if not executions or type(warmup) is not int or warmup < 1:
            raise ValueError("capture requires executions and at least one warmup")

        device = executions[0].module.device
        st = stream_of(device, stream)
        world = next((e.module.world for e in executions if e.module.world), None)

        def stage(name, fn, signature=None):
            return world._stage(name, fn, signature) if world else fn()

        def preflight():
            if st.cuda_stream == 0:
                raise ValueError("capture requires a non-default CUDA stream")
            with torch.cuda.stream(st):
                if torch.cuda.is_current_stream_capturing():
                    raise ResourceBusy("external or nested CUDA capture is unsupported")

        stage("graph.preflight", preflight, [[e.module.fingerprint for e in executions], warmup])
        handle = stage(
            "graph.create",
            lambda: native().Graph.create([e._native for e in executions], list(keepalive)),
        )
        timeout = world.timeout if world else -1
        try:
            with torch.cuda.stream(st):
                for i in range(warmup):

                    def iteration():
                        handle.begin(st.cuda_stream, False)
                        fn()
                        handle.end_warmup(timeout)

                    stage(f"graph.warmup.{i}", iteration)
                    stage(f"graph.warmup.{i}.calls", lambda: None, list(handle.calls))

                def capture():
                    handle.begin(st.cuda_stream, True)
                    output = fn()
                    if not isinstance(output, torch.Tensor) or output.device != device:
                        raise TypeError(
                            "capture callback must return one CUDA Tensor on its device"
                        )
                    handle.end_capture(output)

                stage("graph.capture", capture)
                stage("graph.capture.calls", lambda: None, list(handle.calls))
        except BaseException as error:
            try:
                handle.abort()
            except Exception as cleanup:
                if world:
                    world._fail(str(cleanup))

                error.cleanup_error = cleanup
                raise error from cleanup
            # Native GC retains partially submitted warmup work until its tail event completes.
            raise

        graph = cls.__new__(cls)
        graph._native, graph._device, graph._world = handle, device, world
        graph._closed = False

        return graph

    @property
    def device(self):
        return self._device

    @property
    def world(self):
        return self._world

    @property
    def output(self):
        if self._closed:
            raise RuntimeError("Graph is closed")

        return self._native.output

    def replay(self, stream=None, wait_for=()):
        st = stream_of(self.device, stream, wait_for)
        if self.world:
            self.world._invoke(lambda: self._native.replay(st.cuda_stream))
        else:
            self._native.replay(st.cuda_stream)

        return self.output

    def wait(self, timeout=None):
        if self._closed:
            return

        timeout = (
            self.world.timeout
            if timeout is None and self.world
            else (-1 if timeout is None else timeout)
        )
        if self.world:
            self.world._invoke(lambda: self._native.wait(timeout))
        else:
            self._native.wait(timeout)

    @world_guarded
    def close(self, wait=True):
        if not self._closed:
            try:
                close_owner(self._native, wait, self.world.timeout if self.world else -1)
            finally:
                self._closed = self._native.closed
