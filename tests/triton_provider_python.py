"""Provider API and launch-ABI checks, optionally compiling CUDA without a GPU.

Run after building the extension. --compiler-path can point to the Rust cdylib
when the package is not installed. No timing or GPU accuracy claim is made.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaptureLaunch:
    def __init__(self, original, configs, records):
        self.fn = original.fn
        self.best_config = configs[0]
        self.records = records

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            constants = dict(self.best_config.kwargs)
            constants.update(kwargs)
            values = dict(zip(self.fn.arg_names, args)) | constants
            assert set(values) == set(self.fn.arg_names), (self.fn.arg_names, values.keys())
            launch_grid = grid(constants) if callable(grid) else grid
            assert all(n > 0 for n in launch_grid), launch_grid
            self.records.append((self.fn, values, self.best_config, launch_grid))
        return call


def check_program(compiler, name, text, shapes, output, offline, target):
    opts = json.dumps({'shapes': shapes, 'autotune': {'max_configs': 1}})
    program = compiler.lower_triton(text, opts)
    ast.parse(program.source)
    # Re-emission only needs the common plan, not the source text.
    regenerated = compiler.emit_triton(program.physical_plan, opts)
    assert regenerated == program.source
    source = compiler.emit_python(program.physical_plan)
    ast.parse(source)
    path = output / f'{name}_program.py'
    path.write_text(source)
    module = load(f'provider_{name}', path)
    manifest = module._MANIFEST
    assert manifest['mode'] == 'triton_program'
    metadata = json.loads(program.physical_plan.metadata_json())
    values = {v['value']: v for v in metadata['values']}
    dtypes = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}
    inputs = {binding['name']: torch.empty(values[binding['value']]['shape'],
              dtype=dtypes[values[binding['value']]['dtype']]) for binding in metadata['inputs']}
    records = []
    for i in range(manifest['kernels']):
        setattr(module, f'kernel_{i}', CaptureLaunch(getattr(module, f'kernel_{i}'),
                getattr(module, f'KERNEL_{i}_CONFIGS'), records))
    executable = module.prepare(inputs, providers=['triton'])
    assert executable.reports[0]['comparison'] == 'not_performed'
    result = executable(inputs)
    assert len(records) == manifest['kernels']
    results = result if isinstance(result, tuple) else (() if result is None else (result,))
    for binding, tensor in zip(metadata['outputs'], results, strict=True):
        assert list(tensor.shape) == values[binding['value']]['shape']
        if binding['value'] in metadata['mutable_inputs']:
            assert any(tensor is t for t in inputs.values())
    if offline:
        pointer_types = {torch.float16: '*fp16', torch.bfloat16: '*bf16', torch.float32: '*fp32'}
        for fn, args, config, grid in records:
            signature, constants = {}, {}
            for i, arg in enumerate(fn.arg_names):
                value = args[arg]
                if i in fn.constexprs:
                    signature[arg] = 'constexpr'
                    constants[arg] = value
                else:
                    signature[arg] = pointer_types[value.dtype]
            compiled = triton.compile(ASTSource(fn, signature, constexprs=constants),
                target=GPUTarget('cuda', target, 32),
                options={'num_warps': config.num_warps, 'num_stages': config.num_stages})
            assert compiled.asm['ptx'] and compiled.asm['cubin']
            print(f'{name}/{fn.__name__}: CUDA compiled, grid={grid}, shared={compiled.metadata.shared}')
    print(f'{name}: common plan, source, input/output ABI and {len(records)} ordered launches passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler-path', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--target', type=int, default=90)
    args = parser.parse_args()
    if args.compiler_path:
        compiler = load('_compiler', args.compiler_path.resolve())
    else:
        from trinity_lowering import _compiler as compiler
    root = Path(__file__).resolve().parents[1]
    fixtures = root / 'tests/fixtures/batched_mla'
    output = root / 'target/tests/triton_provider'
    output.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for line in (fixtures / 'shapes.txt').read_text().splitlines():
        name, *dims = line.split()
        shapes[name] = [int(n) for n in dims]
    for stage in (14, 16, 20):
        text = (fixtures / f'batched_mla_postprocessed_stage{stage}.txt').read_text()
        check_program(compiler, f'mla_stage{stage}', text, shapes, output, args.offline, args.target)
    mutation = '''(ploop 0 8 4 i (seq
      (store (view (output Y) (layout (axis m 8))) 2 (keyed_index (slot m (tile i 4))))
      (store (view (output Cache) (layout (axis m 8)))
        (+ (load (view (input Cache) (layout (axis m 8))) (keyed_index (slot m (tile i 4)))) 1)
        (keyed_index (slot m (tile i 4))))))'''
    check_program(compiler, 'cache_outputs', mutation, {}, output, args.offline, args.target)


if __name__ == '__main__':
    main()
