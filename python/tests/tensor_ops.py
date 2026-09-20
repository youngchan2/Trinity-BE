"""Physical-plan fixtures for primitive tensor and distributed operation tests."""

import struct
import sys
from pathlib import Path
import trinity_lowering as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from plan_syntax import all_gather, index, load, store


class Builder:
    def __init__(self, world_size=1):
        self.builder = tl.PhysicalPlanBuilder(world_size)
        self.values = []
        self.statements = []

    def value(self, dtype, shape, storage):
        value = self.builder.add_value(dtype, list(shape), storage)
        self.values.append((dtype, list(shape)))
        return value

    def input(self, name, dtype, shape):
        value = self.value(dtype, shape, "external")
        self.builder.bind_input(name, value)
        return value

    def operation(self, name, inflows, dtype, shape, storage="global", **attributes):
        output = self.value(dtype, shape, storage)
        if name == "reduce_sum":
            access = index("(tile row 1)")
            value = load(inflows[0], self.values[inflows[0]][1], index("(tile row 1)", "fulltile"))
            rhs = f"(rsum {value} 1)"
        else:
            parts = (
                ["(clipped_tile col 128)"]
                if len(shape) == 1
                else ["(tile row 1)", "(clipped_tile col 128)"]
            )
            access = index(*parts)
            if name == "broadcast":
                rhs = (
                    f"(bcast {load(inflows[0], self.values[inflows[0]][1], index('(tile row 1)'))} 1)"
                )
            else:
                args = [load(v, self.values[v][1], access) for v in inflows]
                if name == "scalar_div":
                    bits = struct.unpack("I", struct.pack("f", attributes["scalar"]))[0]
                    args.append(f"(float_bits {bits})")
                operator = {
                    "add": "+",
                    "mul": "*",
                    "div": "/",
                    "scalar_div": "/",
                    "square": "sqr",
                }.get(name, name)
                rhs = f"({operator} {' '.join(args)})"
        body = store(output, shape, rhs, access)
        node = self.builder.add_operation(inflows, [output], expression=body)
        if name == "reduce_sum":
            node = self.builder.add_loop("parallel", "row", 0, shape[0], 1, [node])
        else:
            node = self.builder.add_loop(
                "parallel", "col", 0, ((shape[-1] + 127) // 128) * 128, 128, [node]
            )
            if len(shape) == 2:
                node = self.builder.add_loop("parallel", "row", 0, shape[0], 1, [node])
        self.statements.append(node)
        return output

    def pointwise(self, name, inflows, dtype, storage="global", **attributes):
        return self.operation(
            name, inflows, dtype, self.values[inflows[0]][1], storage, **attributes
        )

    def reduce(self, value, storage="global"):
        return self.operation(
            "reduce_sum",
            [value],
            "fp32",
            self.values[value][1][:1],
            storage,
            axis=1,
        )

    def broadcast(self, value, columns, storage="global"):
        dtype, shape = self.values[value]
        return self.operation(
            "broadcast",
            [value],
            dtype,
            [shape[0], columns],
            storage,
            axis=1,
        )

    def finish(self, output):
        return self.builder.build(self.statements, "Y", output)


def pointwise(name, shape, dtypes, world_size=1, **attributes):
    b = Builder(world_size)
    inputs = [b.input(f"X{i}", dtype, shape) for i, dtype in enumerate(dtypes[:-1])]
    return b.finish(b.pointwise(name, inputs, dtypes[-1], "external", **attributes))


def silu(shape=(16, 16384), world_size=1):
    b = Builder(world_size)
    x = b.input("X", "bf16", shape)
    sigmoid = b.pointwise("sigmoid", [x], "fp32")
    return b.finish(b.pointwise("mul", [sigmoid, x], "bf16", "external"))


def normalization(shape=(16, 4096), world_size=1):
    b = Builder(world_size)
    x = b.input("X", "bf16", shape)
    square = b.pointwise("square", [x], "fp32")
    total = b.reduce(square)
    mean = b.pointwise("scalar_div", [total], "fp32", scalar=float(shape[1]))
    root = b.pointwise("sqrt", [mean], "fp32")
    denominator = b.broadcast(root, shape[1])
    return b.finish(b.pointwise("div", [x, denominator], "bf16", "external"))


def gated_silu(shape=(16, 16384), world_size=1):
    b = Builder(world_size)
    a = b.input("A", "bf16", shape)
    x = b.input("B", "bf16", shape)
    sigmoid = b.pointwise("sigmoid", [x], "fp32")
    activation = b.pointwise("mul", [sigmoid, x], "bf16")
    return b.finish(b.pointwise("mul", [a, activation], "bf16", "external"))


def gather_sum_squares(world_size):
    b = Builder(world_size)
    x = b.input("X", "bf16", [128, 128])
    gathered = b.value("bf16", [128, 128 * world_size], "global")
    access = index("(tile row 64)", "(tile col 64)")
    body = all_gather(x, [128, 128], access, gathered, [128, 128 * world_size], access, 1)
    op = b.builder.add_operation([x], [gathered], expression=body)
    col = b.builder.add_loop("parallel", "col", 0, 128, 64, [op])
    row = b.builder.add_loop("parallel", "row", 0, 128, 64, [col])
    b.statements.append(row)
    square = b.pointwise("square", [gathered], "fp32")
    return b.finish(b.reduce(square, "external"))
