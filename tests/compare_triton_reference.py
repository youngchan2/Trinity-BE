"""Compare Rust output against frozen output from Trinity/backend/codegen.

Checks ABI, autotune configuration, wrapper launches, loop nests, arithmetic,
pointer offsets and masks. Temporary names/parentheses/comments are immaterial.
The only semantic normalizations are redundant x = x + 0 keepalives and an
explicit fp16 cast at stores (the reference also stores through fp16 pointers).
"""
import ast
import copy
import difflib
from pathlib import Path
import subprocess
import sys
import tempfile


def dump(node):
    return ast.dump(node, include_attributes=False)


class Expand(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Name(self, node):
        return copy.deepcopy(self.values.get(node.id, node))


def normalize_body(statements, incoming=None):
    values = dict(incoming or {})
    events = []
    zeros = []
    for statement in statements:
        if isinstance(statement, ast.Assign):
            name = statement.targets[0].id
            value = Expand(values).visit(copy.deepcopy(statement.value))
            if name.startswith(("offset_", "mask_", "temp_")) or "_indices" in name:
                values[name] = value
                continue
            # Known artificial loop keepalives from the original emitter.
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add) and isinstance(value.left, ast.Name) and value.left.id == name and isinstance(value.right, ast.Constant) and value.right.value == 0:
                continue
            if isinstance(value, ast.Call) and ast.unparse(value.func) == "tl.zeros":
                zeros.append((name, dump(value)))
            else:
                events.append(("assign", name, dump(value)))
        elif isinstance(statement, ast.For):
            events.append(("for", dump(statement.target), dump(statement.iter), normalize_body(statement.body, values)))
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            value = Expand(values).visit(copy.deepcopy(statement.value))
            if ast.unparse(value.func) == "tl.store":
                stored = value.args[1]
                if isinstance(stored, ast.Call) and isinstance(stored.func, ast.Attribute) and stored.func.attr == "to" and len(stored.args) == 1 and ast.unparse(stored.args[0]) == "tl.float16":
                    value.args[1] = stored.func.value
            events.append(("call", dump(value)))
        elif isinstance(statement, ast.Pass):
            continue
        else:
            raise AssertionError(f"unexpected kernel statement: {ast.unparse(statement)}")
    return sorted(zeros), events


def check(reference, actual):
    ref, got = ast.parse(reference), ast.parse(actual)
    def metadata(module):
        return {n.targets[0].id: ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)}
    assert metadata(ref) == metadata(got), "benchmark metadata differs"
    refs = {n.name: n for n in ref.body if isinstance(n, ast.FunctionDef)}
    actuals = {n.name: n for n in got.body if isinstance(n, ast.FunctionDef)}
    assert refs.keys() == actuals.keys(), "kernel/wrapper names differ"
    for name, original in refs.items():
        generated = actuals[name]
        assert dump(original.args) == dump(generated.args), f"{name}: ABI differs"
        assert [dump(x) for x in original.decorator_list] == [dump(x) for x in generated.decorator_list], f"{name}: autotune differs"
        if name == "forward":
            a = [dump(n) for n in original.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
            b = [dump(n) for n in generated.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
        else:
            a, b = normalize_body(original.body), normalize_body(generated.body)
        if a != b:
            import pprint
            difference = '\n'.join(difflib.unified_diff(pprint.pformat(a, width=180).splitlines(), pprint.pformat(b, width=180).splitlines(), fromfile="Trinity/backend", tofile="Rust"))
            raise AssertionError(f"{name}: source structure differs\n{difference}")


def main():
    if sys.argv[1] == "--sources":
        references = Path(__file__).parent / "fixtures/triton_reference"
        for name, source in zip(["ffn_177", "vanilla_591", "vanilla_falcon_591"], sys.argv[2:], strict=True):
            check((references / f"{name}.py").read_text(), Path(source).read_text())
        return
    binary = str(Path(sys.argv[1]).resolve())
    fixtures = Path(__file__).parent / "fixtures"
    with tempfile.TemporaryDirectory(prefix="trinity-reference-") as temporary:
        directory = Path(temporary)
        for task, number in [("ffn", 177), ("vanilla", 591)]:
            ir = next(line.split(':', 1)[1] for line in (fixtures / f"analyzer/{task}_cases.txt").read_text().splitlines() if line.startswith(f"{number}:"))
            (directory / "input.ir").write_text(ir)
            output = directory / "kernel.py"
            subprocess.run([binary, str(directory / "input.ir"), str(fixtures / f"triton_reference/{task}.shapes"), str(output), "tile_n=128", "tile_k=64", "tile_p=64"], check=True)
            check((fixtures / f"triton_reference/{task}_{number}.py").read_text(), output.read_text())
            print(f"{task} {number}: reference ABI, autotune, wrapper and kernel structure match")
        subprocess.run([binary, str(fixtures / "triton_reference/vanilla_falcon_591.ir"), str(fixtures / "triton_reference/vanilla_falcon.shapes"), str(output), "tile_n=128", "tile_k=64", "tile_p=64"], check=True)
        check((fixtures / "triton_reference/vanilla_falcon_591.py").read_text(), output.read_text())
        print("vanilla Falcon 591: reference ABI, autotune, wrapper and kernel structure match")


if __name__ == "__main__":
    main()
