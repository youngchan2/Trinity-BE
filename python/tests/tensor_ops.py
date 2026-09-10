"""Physical-plan fixtures for primitive tensor and distributed operation tests."""

import trinity_lowering as tl


class Builder:
    def __init__(self, world_size=1):
        self.builder = tl.PhysicalPlanBuilder(world_size)
        self.values = []

    def value(self, dtype, shape, storage):
        value = self.builder.add_value(dtype, list(shape), storage)
        self.values.append((dtype, list(shape)))
        return value

    def input(self, name, dtype, shape):
        value = self.value(dtype, shape, "external")
        self.builder.bind_input(name, value)
        return value

    def operation(self, definition, inputs, dtype, shape, storage="global", **attributes):
        output = self.value(dtype, shape, storage)
        operands = [self.values[v] for v in [*inputs, output]]
        candidates = definition.enumerate(
            [v[0] for v in operands], [v[1] for v in operands], **attributes
        )
        if not candidates:
            raise ValueError(f"no implementation for {definition.id}")
        operation = self.builder.add_operation(inputs, [output], candidates[0])
        self.builder.add_statement([operation])
        return output

    def pointwise(self, name, inputs, dtype, storage="global", **attributes):
        definition = next(d for d in tl.pointwise_implementations() if d.id == f"cuda.{name}")
        return self.operation(
            definition, inputs, dtype, self.values[inputs[0]][1], storage, **attributes
        )

    def reduce(self, value, storage="global"):
        return self.operation(
            tl.reduce_sum_implementations()[0],
            [value],
            "fp32",
            self.values[value][1][:1],
            storage,
            axis=1,
        )

    def broadcast(self, value, columns, storage="global"):
        dtype, shape = self.values[value]
        return self.operation(
            tl.broadcast_implementations()[0],
            [value],
            dtype,
            [shape[0], columns],
            storage,
            axis=1,
        )

    def finish(self, output):
        return self.builder.finalize("Y", output)


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
    definition = next(d for d in tl.all_gather_implementations() if d.id.endswith("peer_pull"))
    implementation = definition.enumerate(
        "bf16", [[128, 128], [128, 128 * world_size]], 1, world_size
    )[0]
    op = b.builder.add_operation([x], [gathered], implementation)
    b.builder.add_statement([op])
    square = b.pointwise("square", [gathered], "fp32")
    return b.finish(b.reduce(square, "external"))
