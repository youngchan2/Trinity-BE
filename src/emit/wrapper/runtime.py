"""Streamed execution and explicit, per-instance candidate preparation."""
import hashlib
import json
import linecache
import statistics
import threading
import types


def _load_candidate(candidate, sources):
    source = sources[candidate['key']]
    digest = hashlib.sha256(source.encode()).hexdigest()
    filename = f'<trinity_{digest}.py>'
    # Triton JIT inspects its function's source. Keep source in linecache so
    # generated modules work without temporary files or package installation.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(f'trinity_{digest}')
    module.__file__ = filename
    exec(compile(source, filename, 'exec'), module.__dict__)
    return module.run


def _dtype(name):
    return {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[name]


def _bind(manifest, inputs):
    expected_names = {b['name'] for b in manifest['inputs']}
    if set(inputs) != expected_names:
        raise ValueError(f'expected inputs {sorted(expected_names)}')
    if not inputs:
        raise ValueError('Python execution currently needs at least one input for the device')
    values = {}
    device = next(iter(inputs.values())).device
    for binding in manifest['inputs']:
        value = inputs[binding['name']]
        i = binding['value']
        if i in values and values[i] is not value:
            raise ValueError('input aliases must refer to the same tensor object')
        values[i] = value
    for spec in manifest['values']:
        i = spec['id']
        if i not in values:
            values[i] = torch.empty(spec['shape'], dtype=_dtype(spec['dtype']), device=device)
        value = values[i]
        if not value.is_cuda or value.device != device:
            raise ValueError('all inputs must be on the same CUDA device')
        if list(value.shape) != spec['shape'] or value.dtype != _dtype(spec['dtype']) or not value.is_contiguous():
            raise ValueError(f'value {i}: expected contiguous {spec["dtype"]} with shape {spec["shape"]}')
    if tuple(torch.cuda.get_device_capability(device)) != tuple(manifest['capability']):
        raise ValueError('device capability differs from the compiled plan')
    return values, device


def _bench(call, device, repeats):
    with torch.cuda.device(device):
        for _ in range(5):
            call()
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                call()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) / repeats)
        return statistics.median(samples)


class Executable:
    """Fixed-shape selection on one device. Calls allocate fresh intermediates.

    Reports contain all candidate outcomes and the winner per operation. No
    benchmark, package probing or silent reselection occurs during __call__.
    """
    def __init__(self, manifest, device, steps, reports):
        self.manifest, self.device = manifest, device
        self._steps, self.reports = steps, reports

    def __call__(self, inputs):
        with torch.no_grad():
            values, device = _bind(self.manifest, inputs)
            if device != self.device:
                raise ValueError('prepare again for a different device')
            with torch.cuda.device(device):
                for operation, run in self._steps:
                    run(values)
            return values[self.manifest['output']]


_PREPARE_LOCK = threading.Lock()


def _prepare(manifest, sources, inputs, *, providers=None, rtol=1e-2, atol=1e-2, repeats=10):
    if repeats < 1 or not isinstance(repeats, int):
        raise ValueError('repeats must be a positive integer')
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError('tolerances must be finite and nonnegative')
    if providers is not None:
        if isinstance(providers, str):
            raise ValueError('providers must be a sequence, e.g. ["triton", "quack"]')
        if not providers or set(providers) - {'triton', 'quack'}:
            raise ValueError('providers must contain triton and/or quack')
    reports, steps = [], []
    with _PREPARE_LOCK, torch.no_grad():
        values, device = _bind(manifest, inputs)
        with torch.cuda.device(device):
            for operation in manifest['operations']:
                if operation.get('expression') is None and manifest.get('mode') == 'region_candidates':
                    candidates = [c for c in operation['candidates']
                                  if c['provider'] == 'triton' and (providers is None or 'triton' in providers)]
                    if not candidates:
                        raise RuntimeError(f'region {operation["id"]} requires Triton fallback')
                    candidate = candidates[0]
                    run = _load_candidate(candidate, sources)
                    run(values)
                    steps.append((operation, run))
                    reports.append({'region': operation['id'], 'selected': candidate['key'],
                                    'comparison': 'not_performed',
                                    'candidates': operation['rejections']})
                    continue
                output = operation['output']
                expected = torch.empty_like(values[output])
                expected.view(operation['output_view_shape']).copy_(torch.as_tensor(_reference(operation['expression'], values), device=device, dtype=torch.float32))
                candidates = [c for c in operation['candidates'] if providers is None or c['provider'] in providers]
                compare = lambda actual, reference: torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol, equal_nan=False)
                # Only this operation's memory boundary belongs in its trial.
                trial_values = {i: values[i] for i in set(operation['inputs']) | {output}}
                try:
                    run, selected, outcomes = _select(
                        candidates, lambda c: _load_candidate(c, sources), trial_values, output, expected,
                        compare, lambda call: _bench(call, device, repeats),
                    )
                except RuntimeError as error:
                    error.operation = operation['id']
                    error.reports = operation['rejections'] + getattr(error, 'reports', [])
                    raise
                reports.append({'operation': operation['id'], 'selected': selected, 'candidates': operation['rejections'] + outcomes})
                steps.append((operation, run))
                # The next operation receives exactly the preceding store's dtype.
                # Use the reference during preparation so errors cannot accumulate.
                values[output].copy_(expected)
    return Executable(manifest, device, steps, reports)
