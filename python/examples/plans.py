"""Concrete plans built using Rust implementation enumeration."""

import trinity_lowering as tl


def gemm(m=128, n=128, k=64, world_size=1):
    shapes = [[m, k], [k, n], [m, n]]
    builder = tl.PhysicalPlanBuilder(world_size)

    values = [builder.add_value("bf16", shape, "external") for shape in shapes]

    builder.bind_input("X", values[0])
    builder.bind_input("W", values[1])

    instance = tl.gemm_implementations()[0].enumerate(["bf16"] * 3, shapes)[0]
    op = builder.add_operation(values[:2], values[2:], instance)
    builder.add_action([op])

    return builder.finalize("Y", values[2])


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
