from pathlib import Path
import subprocess


def test_native_core_without_cuda(tmp_path):
    root = Path(__file__).resolve().parents[1]
    common = [
        "c++",
        "-std=c++17",
        "-pthread",
        "-I",
        str(root.parent / "src/native"),
        "-I",
        str(root.parent / "src/emit/cuda"),
    ]
    libs = []
    for name, defines in [
        ("good", []),
        ("version", ["-DBAD_VERSION"]),
        ("offset", ["-DBAD_OFFSET"]),
    ]:
        lib = tmp_path / f"{name}.so"
        subprocess.run(
            common
            + defines
            + ["-fPIC", "-shared", str(root / "tests/fixtures/library.cpp"), "-o", str(lib)],
            check=True,
            timeout=30,
        )
        libs.append(str(lib))

    executable = tmp_path / "core_test"
    subprocess.run(
        common + [str(root / "tests/fixtures/core_test.cpp"), "-ldl", "-o", str(executable)],
        check=True,
        timeout=30,
    )

    subprocess.run([str(executable), *libs], check=True, timeout=10)
