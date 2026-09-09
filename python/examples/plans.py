"""Concrete plans built using Rust implementation enumeration."""

import trinity_lowering as tl


def gemm(m=128, n=128, k=64, world_size=1):
    """BF16 GEMM followed by a BF16 [M, N] bias addition."""
    builder = tl.PhysicalPlanBuilder(world_size)
    output = _gemm_bias(builder, m, n, k, "external")
    return builder.finalize("Y", output)


def gemm_relu(m=128, n=128, k=64, world_size=1):
    """BF16 GEMM, bias addition and ReLU, each as a singleton Action."""
    builder = tl.PhysicalPlanBuilder(world_size)
    linear = _gemm_bias(builder, m, n, k, "global")
    output = builder.add_value("bf16", [m, n], "external")
    definition = next(d for d in tl.pointwise_implementations() if d.id == "cuda.relu")
    instance = definition.enumerate(["bf16"] * 2, [[m, n]] * 2)[0]
    op = builder.add_operation([linear], [output], instance)
    builder.add_action([op])
    return builder.finalize("Y", output)


def _gemm_bias(builder, m, n, k, output_storage):
    shapes = [[m, k], [k, n], [m, n]]

    x = builder.add_value("bf16", shapes[0], "external")
    weight = builder.add_value("bf16", shapes[1], "external")
    product = builder.add_value("bf16", shapes[2], "global")
    bias = builder.add_value("bf16", shapes[2], "external")
    output = builder.add_value("bf16", shapes[2], output_storage)

    builder.bind_input("X", x)
    builder.bind_input("W", weight)
    builder.bind_input("bias", bias)

    instance = tl.gemm_implementations()[0].enumerate(["bf16"] * 3, shapes)[0]
    op = builder.add_operation([x, weight], [product], instance)
    builder.add_action([op])

    definition = next(d for d in tl.pointwise_implementations() if d.id == "cuda.add")
    instance = definition.enumerate(["bf16"] * 3, [[m, n]] * 3)[0]
    op = builder.add_operation([product, bias], [output], instance)
    builder.add_action([op])

    return output


def gather(backend="peer_push", axis=0, world_size=2):
    shapes = [[128, 128], [128, 128]]
    shapes[1][axis] *= world_size

    builder = tl.PhysicalPlanBuilder(world_size)

    x, y = [builder.add_value("bf16", s, "external") for s in shapes]

    builder.bind_input("X", x)

    definition = next(d for d in tl.all_gather_implementations() if backend in d.id)
    implementation = definition.enumerate("bf16", shapes, axis, world_size)[0]
    op = builder.add_operation([x], [y], implementation)
    builder.add_action([op])

    return builder.finalize("Y", y)
