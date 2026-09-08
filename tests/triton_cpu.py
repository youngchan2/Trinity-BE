"""Numerical smoke tests of emitted Python using a small NumPy TL emulator.

This checks emitted arithmetic, masks, storage and wrapper behavior. It does NOT
compile Triton or validate GPU synchronization/layouts. Run after building the
emit_triton example: python3 tests/triton_cpu.py target/debug/examples/emit_triton
"""
import contextlib
import importlib.util
import itertools
import math
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np


class Tile(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tile)


def tile(value):
    return np.asarray(value).view(Tile)


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data.reshape(-1), offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


class Device:
    type = "cuda"


DEVICE = Device()


class HostTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.device = DEVICE
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def stride(self, axis):
        return self.data.strides[axis] // self.data.itemsize

    def view(self, *shape):
        return HostTensor(self.data.reshape(shape))


PROGRAM = ()


class Jit:
    def __init__(self, function):
        self.function = function
        self.configs = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            global PROGRAM
            selected = self.configs[0] if self.configs else {}
            meta = dict(selected, **kwargs)
            concrete_grid = grid(meta) if callable(grid) else grid
            for PROGRAM in itertools.product(*(range(n) for n in concrete_grid)):
                self.function(*(Pointer(a.data) if isinstance(a, HostTensor) else a for a in args), **meta)
        return launch


def access(pointer, mask):
    offsets, mask = np.broadcast_arrays(np.asarray(pointer.offset), mask)
    assert np.all((offsets[mask] >= 0) & (offsets[mask] < pointer.data.size)), "unmasked OOB access"
    return offsets, mask


def load(pointer, mask, other):
    offsets, mask = access(pointer, mask)
    result = np.full(offsets.shape, other, dtype=pointer.data.dtype)
    result[mask] = pointer.data[offsets[mask]]
    assert not np.isnan(result[mask]).any(), "read of uninitialized storage"
    return tile(result)


def store(pointer, value, mask):
    offsets, mask = access(pointer, mask)
    values = np.broadcast_to(value, offsets.shape)
    pointer.data[offsets[mask]] = values[mask]


def install_emulator():
    tl = types.ModuleType("triton.language")
    tl.float16, tl.float32, tl.int32 = np.float16, np.float32, np.int32
    tl.int64 = np.int64
    tl.constexpr = object()
    tl.arange = lambda a, b: tile(np.arange(a, b))
    tl.zeros = lambda shape, dtype: tile(np.zeros(shape, dtype))
    tl.full = lambda shape, value, dtype: tile(np.full(shape, value, dtype))
    tl.load, tl.store = load, store
    tl.program_id = lambda axis: tile(PROGRAM[axis])
    tl.debug_barrier = lambda: None
    tl.permute = lambda value, order: tile(np.transpose(value, order))
    tl.reshape = lambda value, shape: tile(np.reshape(value, shape))
    tl.expand_dims = lambda value, axis: tile(np.expand_dims(value, axis))
    tl.broadcast_to = lambda value, shape: tile(np.broadcast_to(value, shape))
    tl.where = lambda mask, a, b: tile(np.where(mask, a, b))
    tl.dot = lambda a, b, out_dtype=None: tile(np.matmul(a.astype(np.float32), b.astype(np.float32)))
    tl.sigmoid = lambda a: tile(1 / (1 + np.exp(-a)))
    tl.math = types.SimpleNamespace(erf=lambda a: tile(np.vectorize(math.erf)(a)))
    for name in ["exp", "sqrt", "abs", "sum", "max", "min", "maximum", "minimum"]:
        setattr(tl, name, lambda *args, _name=name, **kwargs: tile(getattr(np, _name)(*args, **kwargs)))
    tl.sum = lambda value, axis, dtype=None: tile(np.sum(value, axis=axis, dtype=dtype or (np.float32 if value.dtype == np.float16 else None)))
    triton = types.ModuleType("triton")
    triton.jit = Jit
    triton.language = tl
    triton.Config = lambda values: values
    def autotune(configs, key):
        def decorate(kernel):
            kernel.configs = configs
            return kernel
        return decorate
    triton.autotune = autotune
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = np.float16, np.float32
    torch.device = lambda *args: DEVICE
    torch.cuda = types.SimpleNamespace(device=lambda _: contextlib.nullcontext(), current_device=lambda: 0)
    # Poison allocations so accidental reads from missing initializers fail.
    torch.empty = lambda shape, device, dtype: HostTensor(np.full(shape, np.nan, dtype))
    sys.modules.update({"torch": torch, "triton": triton, "triton.language": tl})


def run(binary, directory, name, ir, shapes, inputs):
    ir_path, shapes_path, output = (directory / f"{name}{suffix}" for suffix in [".ir", ".shapes", ".py"])
    ir_path.write_text(ir)
    shapes_path.write_text("".join(f"{key} {' '.join(map(str, shape))}\n" for key, shape in shapes.items()))
    subprocess.run([binary, str(ir_path), str(shapes_path), str(output)], check=True)
    spec = importlib.util.spec_from_file_location(name, output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arguments = {key: inputs[key] if key in inputs else HostTensor(np.full(shapes[key], np.nan, np.float16)) for key in module.TENSOR_PARAMS}
    # Padded lanes can compute 0/0 before the emitter's final validity selection.
    # Uninitialized memory is checked in load(), independently of these warnings.
    with np.errstate(invalid="ignore", divide="ignore"):
        assert module.forward(**arguments) is None
    return arguments


def close(actual, expected):
    np.testing.assert_allclose(actual.data, np.asarray(expected, dtype=np.float16), rtol=2e-3, atol=2e-3)


def fixture(name, number, replacements):
    lines = (Path(__file__).parent / f"fixtures/analyzer/{name}_cases.txt").read_text().splitlines()
    ir = next(line.split(':', 1)[1] for line in lines if line.startswith(f"{number}:"))
    return re.sub(r"[A-Za-z_]+|\d+", lambda match: str(replacements.get(match[0], match[0])), ir)


def main():
    binary = str(Path(sys.argv[1]).resolve())
    install_emulator()
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory(prefix="trinity-triton-cpu-") as path:
        directory = Path(path)
        a = HostTensor(rng.normal(0, .2, (3, 9)).astype(np.float16))
        b = HostTensor(rng.normal(0, .2, (9, 10)).astype(np.float16))
        result = run(binary, directory, "dot_tail", """
        (ploop 0 10 4 n (sloop 0 9 4 k (store (output C)
          (+ (load (output C) (index fulltile (tile n)))
             (@ (load (input A) (index fulltile (tile k))) (load (input B) (index (tile k) (tile n)))))
          (index fulltile (tile n)))))""", {"A": (3, 9), "B": (9, 10), "C": (3, 10)}, {"A": a, "B": b})
        close(result["C"], a.data.astype(np.float32) @ b.data.astype(np.float32))

        a = HostTensor(np.linspace(-1, 1, 7).astype(np.float16))
        result = run(binary, directory, "exp_tail", """
        (sloop 0 7 4 k (store (output O) (+ (load (output O) (index fulltile))
          (rsum (exp (load (input A) (index (tile k)))) 0)) (index fulltile)))""",
                     {"A": (7,), "O": (1,)}, {"A": a})
        close(result["O"], [np.exp(a.data.astype(np.float32)).sum()])

        result = run(binary, directory, "materialized", """
        (ploop 0 2 1 n (seq
          (sloop 0 7 4 k (store (tensor T) (* (load (input A) (index (tile k))) 2) (index (tile k))))
          (sloop 0 7 4 k (store (output O) (+ (load (output O) (index (elem n)))
             (rsum (load (tensor T) (index (tile k))) 0)) (index (elem n))))))""",
                     {"A": (7,), "T": (7,), "O": (2,)}, {"A": a})
        close(result["O"], [(a.data * 2).astype(np.float32).sum()] * 2)

        result = run(binary, directory, "nested_reset", """
        (ploop 0 8 4 n (sloop 0 6 3 p (seq
          (sloop 0 7 2 k (store (tensor S) (+ (load (tensor S) (index fulltile))
            (rsum (exp (load (input A) (index (tile k)))) 0)) (index fulltile)))
          (store (output O) (+ (load (output O) (index (tile n)))
             (* (load (tensor S) (index fulltile)) (+ p 1))) (index (tile n))))))""",
                     {"A": (7,), "S": (1,), "O": (8,)}, {"A": a})
        close(result["O"], np.full(8, np.exp(a.data.astype(np.float32)).sum() * 5))

        result = run(binary, directory, "kernel_update", """
        (seq (ploop 0 7 4 k (store (tensor T) (load (input A) (index (tile k))) (index (tile k))))
          (sloop 0 1 1 n (seq
            (sloop 0 7 4 k (store (tensor T) (+ (load (tensor T) (index (tile k))) 1) (index (tile k))))
            (sloop 0 7 4 k (store (output O) (+ (load (output O) (index (elem n)))
               (rsum (load (tensor T) (index (tile k))) 0)) (index (elem n)))))))""",
                     {"A": (7,), "T": (7,), "O": (1,)}, {"A": a})
        close(result["O"], [(a.data.astype(np.float32) + 1).astype(np.float16).astype(np.float32).sum()])

        cache = HostTensor(np.arange(14, dtype=np.float16).reshape(2, 7))
        update = HostTensor(np.full((2, 2), 9, dtype=np.float16))
        result = run(binary, directory, "cache", """
        (ploop 0 2 1 n (seq
          (store (input C) (load (input U) (index (elem n) fulltile)) (index (elem n) (const_tile 5 2)))
          (store (output O) (rsum (load (input C) (index (elem n) fulltile)) 1) (index (elem n)))))""",
                     {"C": (2, 7), "U": (2, 2), "O": (2,)}, {"C": cache, "U": update})
        close(result["O"], cache.data.astype(np.float32).sum(axis=1))
        np.testing.assert_array_equal(cache.data[:, 5:], update.data)

        a = HostTensor(np.array([-7, -2, -9, -4, -8], dtype=np.float16))
        result = run(binary, directory, "max_tail", """
        (store (output O) (rmax (load (input A) (index fulltile)) 0) (index fulltile))""",
                     {"A": (5,), "O": (1,)}, {"A": a})
        close(result["O"], [-2])

        # A zero-filled load is safe for dot, but exp changes padded zeros to
        # ones. Neutralize only the contracted axis, including after a dot.
        a = rng.normal(0, .1, (3, 9)).astype(np.float16)
        b = rng.normal(0, .1, (9, 5)).astype(np.float16)
        result = run(binary, directory, "exp_dot_tail", """
        (ploop 0 5 4 n (sloop 0 9 4 k
          (store (output C) (+ (load (output C) (index fulltile (tile n)))
            (@ (exp (load (input A) (index fulltile (tile k))))
               (load (input B) (index (tile k) (tile n)))))
            (index fulltile (tile n)))))""",
                     {"A": (3, 9), "B": (9, 5), "C": (3, 5)},
                     {"A": HostTensor(a), "B": HostTensor(b)})
        expected = np.sum(np.exp(a.astype(np.float32)).astype(np.float16)[:, :, None] * b[None, :, :], axis=1, dtype=np.float32)
        close(result["C"], expected)

        result = run(binary, directory, "dot_exp_sum_tail", """
        (store (output O) (rsum (exp (@
          (load (input A) (index fulltile fulltile))
          (load (input B) (index fulltile fulltile)))) 1) (index fulltile))""",
                     {"A": (3, 9), "B": (9, 5), "O": (3,)},
                     {"A": HostTensor(a), "B": HostTensor(b)})
        product = np.sum(a[:, :, None] * b[None, :, :], axis=1, dtype=np.float32)
        close(result["O"], np.exp(product).sum(axis=1))

        # Exercise actual scheduled corpus graphs at smaller extents, including
        # the failed FFN graph and grouped/batched vanilla attention.
        def random(shape):
            return HostTensor(rng.normal(0, .1, shape).astype(np.float16))

        def dot(a, b):
            # The original backend's small-M GEMV path multiplies fp16 lanes
            # before the fp32 sum; these reduced corpus fixtures have M=3.
            return np.sum(np.expand_dims(a.astype(np.float16), -1) * np.expand_dims(b.astype(np.float16), -3), axis=-2, dtype=np.float32)

        ffn_shapes = {name: (3, 32) for name in ["O2", "X", "attn_O1", "attn_O2", "attn_O_norm", "FF2"]}
        ffn_shapes.update(WO=(32, 32), attn_O3=(3,), WFF1a=(32, 48), WFF1b=(32, 48), FF1a=(3, 48), FF1b=(3, 48), WFF2=(48, 32))
        inputs = {name: random(ffn_shapes[name]) for name in ["O2", "X", "WO", "WFF1a", "WFF1b", "WFF2"]}
        stage1 = dot(inputs["O2"].data, inputs["WO"].data).astype(np.float16)
        stage2 = (stage1.astype(np.float32) + inputs["X"].data.astype(np.float32)).astype(np.float16).astype(np.float32)
        norm = stage2 / np.sqrt((stage2 * stage2).sum(axis=1, keepdims=True) / 32)
        fa, fb = (dot(norm, inputs[name].data) for name in ["WFF1a", "WFF1b"])
        expected = dot(fa * (fb / (1 + np.exp(-fb))), inputs["WFF2"].data)
        for number in [4, 177]:
            ir = fixture("ffn", number, {"4096": 32, "14336": 48, "tile_n": 16, "tile_k": 8, "tile_p": 16})
            result = run(binary, directory, f"ffn_{number}", ir, ffn_shapes, inputs)
            if number == 177:
                fa, fb = fa.astype(np.float16).astype(np.float32), fb.astype(np.float16).astype(np.float32)
                expected = dot(fa * (fb / (1 + np.exp(-fb))), inputs["WFF2"].data)
            close(result["FF2"], expected)

        shapes = {name: (3, 32) for name in ["X", "Q1", "K1", "V1", "O2"]}
        shapes.update({name: (32, 32) for name in ["WQ", "WK", "WV"]})
        shapes.update({name: (2, 3, 16) for name in ["Q", "K", "V", "O"]})
        shapes.update(K_cache=(2, 9, 16), V_cache=(2, 9, 16), C_exp=(2, 3, 9), C_sum=(2, 3))
        inputs = {name: random(shapes[name]) for name in ["X", "WQ", "WK", "WV", "K_cache", "V_cache"]}
        q, k, v = (dot(inputs["X"].data, inputs[name].data).reshape(3, 2, 16).transpose(1, 0, 2) for name in ["WQ", "WK", "WV"])
        kc, vc = inputs["K_cache"].data.copy(), inputs["V_cache"].data.copy()
        kc[:, 6:, :], vc[:, 6:, :] = k.astype(np.float16), v.astype(np.float16)
        scores = np.exp(dot(q, kc.transpose(0, 2, 1)))
        expected = (dot(scores, vc) / scores.sum(axis=2, keepdims=True)).transpose(1, 0, 2).reshape(3, 32)
        ir = fixture("vanilla", 591, {"4096": 32, "128": 16, "1024": 9, "1008": 6, "16": 3, "tile_k": 8, "tile_p": 4})
        result = run(binary, directory, "vanilla_591", ir, shapes, inputs)
        close(result["O2"], expected)
        np.testing.assert_array_equal(inputs["K_cache"].data, kc)
        np.testing.assert_array_equal(inputs["V_cache"].data, vc)

        result = run(binary, directory, "constant", "(store (output O) 009 (index fulltile))", {"O": (3,)}, {})
        close(result["O"], [9, 9, 9])
    print("13 emitted-Python numerical smoke tests passed (NumPy emulation; no GPU compilation).")


if __name__ == "__main__":
    main()
