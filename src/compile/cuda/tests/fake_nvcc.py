#!/usr/bin/env python3
"""Host-only compiler fixture: retain linker policy, replace CUDA translation."""
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[2]
if sys.argv[1:] == ["--version"]:
    version = root / "version"
    print(version.read_text() if version.exists() else "Cuda compilation tools, release 13.0, V13.0.88")
    sys.exit(0)

mode = (root / "mode").read_text()
print("fixture compiler stdout", flush=True)
print("fixture compiler warning", file=sys.stderr, flush=True)
if mode == "failure":
    sys.exit(42)
if mode == "missing":
    sys.exit(0)

args = sys.argv[1:]
output = args[args.index("-o") + 1]
if mode == "invalid":
    pathlib.Path(output).write_text("not an ELF library")
    sys.exit(0)

command = ["c++", "-std=c++17", "-shared", "-fPIC", str(root / "fixture.cpp"), "-o", output]
for i, arg in enumerate(args):
    if arg == "-Xlinker":
        command += [arg, args[i + 1]]

sys.exit(subprocess.run(command).returncode)
