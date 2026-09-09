from types import SimpleNamespace
import threading
import pytest
import trinity_lowering as tl
from trinity_lowering.runtime import resolve_bindings
from trinity_lowering.world import World
from test_compiler import identity


class Region:
    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


def test_names_and_identity_alias(monkeypatch):
    monkeypatch.setattr("trinity_lowering.runtime.tensor_check", lambda *args: None)
    req = tl.emit(identity()).requirements
    x = Region(1024)
    names = req.buffers[0].input_names

    with pytest.raises(ValueError, match="exactly"):
        resolve_bindings(req, {names[0]: x}, None, None, None)
    with pytest.raises(ValueError, match="exactly"):
        resolve_bindings(req, {**dict.fromkeys(names, x), "extra": x}, None, None, None)
    assert resolve_bindings(req, dict.fromkeys(names, x), x, None, None) == {0: x}
    with pytest.raises(ValueError, match="same value"):
        resolve_bindings(req, dict.fromkeys(names, x), Region(2048), None, None)


def test_collect_frees_only_common_unleased_allocations():
    world = World.__new__(World)
    world._closed = False
    world._control_lock = threading.RLock()
    world._check = lambda: None
    world._stage = lambda name, fn, signature=None: fn()
    world._exchange = lambda stage, value: [
        [(0, True), (1, False), (2, True)],
        [(0, False), (1, True), (2, True)],
    ]
    freed = []
    world._native = SimpleNamespace(collectable=lambda: [], collect=lambda ids: freed.extend(ids))

    assert world.collect() == 1
    assert freed == [2]


def test_collect_detects_registry_divergence_before_free():
    world = World.__new__(World)
    world._closed = False
    world._control_lock = threading.RLock()
    world._check = lambda: None
    world._stage = lambda name, fn, signature=None: fn()
    world._exchange = lambda stage, value: [[(0, True)], [(1, True)]]
    world._fail = lambda error: setattr(world, "_failure", error)
    world._native = SimpleNamespace(
        collectable=lambda: [], collect=lambda ids: pytest.fail("unsafe free")
    )

    with pytest.raises(tl.DistributedFailure):
        world.collect()


def test_safe_close_returns_leases_and_preserves_original_error():
    from trinity_lowering.runtime import close_owner

    error = tl.RuntimeFailure("device operation failed", code=17)
    owner = SimpleNamespace(closed=False)

    def wait(timeout):
        raise error

    owner.wait = wait
    owner.close = lambda wait: setattr(owner, "closed", True)

    with pytest.raises(tl.RuntimeFailure) as caught:
        close_owner(owner, True, 1)
    assert caught.value is error and owner.closed


def test_uncertain_close_retains_owner_and_cleanup_error():
    from trinity_lowering.runtime import close_owner

    error = tl.RuntimeFailure("tail timeout", submitted=True)

    def wait(timeout):
        raise error

    def close(wait):
        raise tl.ResourceBusy("tail still pending")

    owner = SimpleNamespace(wait=wait, close=close, closed=False)

    with pytest.raises(tl.RuntimeFailure) as caught:
        close_owner(owner, True, 1)
    assert caught.value is error and not owner.closed
    assert isinstance(error.cleanup_error, tl.ResourceBusy)


def test_world_submission_cannot_interleave_a_preparation_transaction():
    world = World.__new__(World)
    world._control_lock = threading.RLock()
    world._check = lambda: None
    started, submitted = threading.Event(), threading.Event()

    def run():
        started.set()
        world._invoke(submitted.set)

    with world._control_lock:
        thread = threading.Thread(target=run)
        thread.start()

        assert started.wait(1)
        assert not submitted.is_set()

    thread.join(timeout=1)

    assert submitted.is_set() and not thread.is_alive()
