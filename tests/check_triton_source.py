"""Check generated Python syntax and conservative loop definition dominance.

New names defined only inside a loop cannot escape it. This catches the legacy
FFN undefined-name pattern without claiming to run the Triton compiler.
Usage: python3 tests/check_triton_source.py GENERATED_DIRECTORY [...]
"""
import ast
from pathlib import Path
import sys


def reads(node, defined):
    missing = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - defined
    assert not missing, f"line {node.lineno}: names lack dominating definitions: {sorted(missing)}"


def block(statements, incoming):
    defined = set(incoming)
    for statement in statements:
        if isinstance(statement, ast.Assign):
            reads(statement.value, defined)
            for target in statement.targets:
                defined.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(statement, ast.For):
            reads(statement.iter, defined)
            block(statement.body, defined | {statement.target.id})
        elif isinstance(statement, ast.If):
            reads(statement.test, defined)
            a = block(statement.body, defined)
            b = block(statement.orelse, defined)
            defined.update(a & b)
        elif isinstance(statement, ast.Expr):
            reads(statement.value, defined)
        else:
            assert isinstance(statement, ast.Pass), type(statement).__name__
    return defined


def main():
    count = 0
    for directory in sys.argv[1:]:
        paths = sorted(Path(directory).glob("*.py"))
        assert paths, f"no generated Python files in {directory}"
        for path in paths:
            try:
                module = ast.parse(path.read_text())
                for node in module.body:
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("kernel_"):
                        block(node.body, {arg.arg for arg in node.args.args} | {"tl", "range", "float"})
            except Exception as error:
                raise AssertionError(f"{path}: {error}") from error
            count += 1
    print(f"{count} modules passed Python syntax and kernel definition-dominance checks.")


if __name__ == "__main__":
    main()
