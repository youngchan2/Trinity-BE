"""Plan construction does not infer loops or compute expressions."""

import json
import sys
from pathlib import Path

import pytest
import trinity_lowering as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from plans import gather, gemm_relu
from tensor_ops import gated_silu, normalization
from plan_syntax import all_gather, index


def test_identity_emits_without_a_launch():
    b = tl.PhysicalPlanBuilder()
    x = b.add_value("fp32", [4], "external")
    b.bind_input("X", x)
    plan = b.build([], "Y", x)
    source = tl.emit(plan)
    assert "<<<" not in source.code
    assert source.requirements.buffers[0].alignment == 4


def copy_builder():
    b = tl.PhysicalPlanBuilder()
    x = b.add_value("fp32", [64], "external", name="X")
    y = b.add_value("fp32", [64], "external", name="Y")
    b.bind_input("X", x)
    return b, x, y


def expression(access):
    return f"(store (view (output Y) (layout (axis a 64))) (sqrt (load (view (input X) (layout (axis a 64))) (keyed_index (slot a {access})))) (keyed_index (slot a {access})))"


def test_compute_requires_an_explicit_body_and_no_loop_is_inferred():
    b, x, y = copy_builder()
    with pytest.raises(TypeError, match="expression"):
        b.add_operation([x], [y])
    node = b.add_operation(
        inflows=[x],
        outflows=[y],
        expression=expression("fulltile"),
    )
    metadata = json.loads(b.build([node], "Y", y).metadata_json())
    assert metadata["statements"] == [{"operations": [0]}]
    assert metadata["operations"] == [
        {
            "id": 0,
            "inflows": [metadata["inputs"][0]["value"]],
            "outflows": [metadata["output"]["value"]],
        }
    ]


def test_nonzero_serial_domain_is_preserved_and_matches_ir():
    b, x, y = copy_builder()
    body = expression("(tile i 16)")
    op = b.add_operation([x], [y], expression=body)
    loop = b.add_loop("sequential", "i", 16, 64, 16, [op])
    plan = json.loads(b.build([loop], "Y", y).metadata_json())
    domain = plan["statements"][0]["loop"]["domain"]
    assert domain == {
        "variable": "lv0",
        "start": {"Constant": 16},
        "stop": {"Constant": 64},
        "step": {"Constant": 16},
    }
    assert plan["statements"][0]["loop"]["body"] == [{"operations": [0]}]
    (ir,) = tl.lower_ir(f"(sloop 16 64 16 i {body})", {}, {"X": "fp32", "Y": "fp32"})
    assert json.loads(ir.metadata_json()) == plan


def test_explicit_nested_bounds_and_duplicate_membership():
    b, x, y = copy_builder()
    op = b.add_operation([x], [y], expression=expression("fulltile"))
    inner = b.add_loop("sequential", "j", "i", "(+ i 1)", 1, [op])
    outer = b.add_loop("parallel", "i", 0, 2, 1, [inner])
    plan = json.loads(b.build([outer], "Y", y).metadata_json())
    inner = plan["statements"][0]["loop"]["body"][0]["loop"]
    assert inner["domain"]["start"] == {"Variable": "lv0"}
    assert inner["domain"]["stop"] == {"Add": [{"Variable": "lv0"}, {"Constant": 1}]}

    b, x, y = copy_builder()
    op = b.add_operation([x], [y], expression=expression("fulltile"))
    loop = b.add_loop("parallel", "i", 0, 1, 1, [op])
    with pytest.raises(ValueError, match="multiple statements"):
        b.build([op, loop], "Y", y)


def test_examples_and_tensor_fixtures_supply_explicit_programs():
    for plan in [gemm_relu(), normalization((3, 257)), gated_silu((3, 257))]:
        data = json.loads(plan.metadata_json())
        assert all("loop" in statement for statement in data["statements"])
    for axis in [0, 1]:
        plan = gather(axis=axis)
        data = json.loads(plan.metadata_json())
        assert len(data["operations"]) == 1
        assert "loop" in data["statements"][0]


def test_plan_examples_do_not_enumerate_implementations(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Plan construction must not select an implementation")

    for name in ["gemm", "pointwise", "reduce_sum", "broadcast", "all_gather"]:
        monkeypatch.setattr(tl, f"{name}_implementations", unexpected)
    for plan in [gemm_relu(), normalization((3, 257)), gather(world_size=3)]:
        assert json.loads(plan.metadata_json())["operations"]


@pytest.mark.parametrize("wrong_source", [False, True])
def test_communication_expression_must_match_declared_flows(wrong_source):
    b = tl.PhysicalPlanBuilder(2)
    x = b.add_value("bf16", [128], "external")
    y = b.add_value("bf16", [256], "external")
    b.bind_input("X", x)
    access = index("fulltile")
    body = all_gather(y if wrong_source else x, [256] if wrong_source else [128], access,
                      y if wrong_source else x, [256] if wrong_source else [128], access, 0)
    op = b.add_operation([x], [y], expression=body)
    with pytest.raises(ValueError, match="declared inflows" if wrong_source else "declared outflow"):
        b.build([op], "Y", y)
