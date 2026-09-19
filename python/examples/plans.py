"""Explicit loop programs; Emit selects their implementations."""

import trinity_lowering as tl
from plan_syntax import all_gather, index, load, store


def gemm(m=128, n=128, k=64, world_size=1):
    """BF16 GEMM followed by a BF16 [M, N] bias addition."""
    builder = tl.PhysicalPlanBuilder(world_size)
    output, statements = _gemm_bias(builder, m, n, k, "external")
    return builder.build(statements, "Y", output)


def gemm_relu(m=128, n=128, k=64, world_size=1):
    """BF16 GEMM, bias addition and ReLU, each as a singleton Statement."""
    builder = tl.PhysicalPlanBuilder(world_size)
    linear, statements = _gemm_bias(builder, m, n, k, "global")
    output = builder.add_value("bf16", [m, n], "external")
    access = index("(tile row 1)", "(clipped_tile col 128)")
    body = store(output, [m, n], f"(relu {load(linear, [m, n], access)})", access)
    op = builder.add_operation([linear], [output], expression=body)
    col = builder.add_loop("parallel", "col", 0, ((n + 127) // 128) * 128, 128, [op])
    row = builder.add_loop("parallel", "row", 0, m, 1, [col])
    return builder.build([*statements, row], "Y", output)


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

    ai = index("(tile m 128)", "(tile k 64)")
    bi = index("(tile k 64)", "(tile n 128)")
    ci = index("(tile m 128)", "(tile n 128)")
    rhs = f"(+ {load(product, shapes[2], ci)} (@ {load(x, shapes[0], ai)} {load(weight, shapes[1], bi)}))"
    op = builder.add_operation(
        [x, weight], [product], expression=store(product, shapes[2], rhs, ci)
    )
    reduction = builder.add_loop("sequential", "k", 0, k, 64, [op])
    columns = builder.add_loop("parallel", "n", 0, n, 128, [reduction])
    rows = builder.add_loop("parallel", "m", 0, m, 128, [columns])

    access = index("(tile row 1)", "(clipped_tile col 128)")
    rhs = f"(+ {load(product, shapes[2], access)} {load(bias, shapes[2], access)})"
    op = builder.add_operation(
        [product, bias], [output], expression=store(output, shapes[2], rhs, access)
    )
    col = builder.add_loop("parallel", "col", 0, ((n + 127) // 128) * 128, 128, [op])
    row = builder.add_loop("parallel", "row", 0, m, 1, [col])
    return output, [rows, row]


def gather(axis=0, world_size=2):
    """Rank-major all-gather with a backend-independent source tile loop."""
    shapes = [[128, 128], [128, 128]]
    shapes[1][axis] *= world_size
    builder = tl.PhysicalPlanBuilder(world_size)
    x, y = [builder.add_value("bf16", shape, "external") for shape in shapes]
    builder.bind_input("X", x)
    access = index("(tile row 64)", "(tile col 64)")
    body = all_gather(x, shapes[0], access, y, shapes[1], access, axis)
    op = builder.add_operation([x], [y], expression=body)
    col = builder.add_loop("parallel", "col", 0, shapes[0][1], 64, [op])
    row = builder.add_loop("parallel", "row", 0, shapes[0][0], 64, [col])
    return builder.build([row], "Y", y)
