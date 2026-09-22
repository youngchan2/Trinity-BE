"""Validate every candidate before allowing its timing to influence selection."""
import math
import torch


def _select(candidates, load, values, output, expected, compare, benchmark):
    reports = []
    valid = []
    for candidate in candidates:
        report = dict(candidate, status='compile')
        try:
            run = load(candidate)
            # Allocate isolated trial state. Candidate writes cannot leak into
            # another candidate, the reference, or the caller's sample tensors.
            trial = {i: v.clone() for i, v in values.items()}
            trial[output].fill_(float('nan'))
            run(trial)  # JIT and lazy library initialization happen before timing.
            report['status'] = 'correctness'
            compare(trial[output], expected)
            for i in values:
                if i != output:
                    torch.testing.assert_close(trial[i], values[i], rtol=0, atol=0, equal_nan=True)
            report['status'] = 'benchmark'
            elapsed = float(benchmark(lambda: run(trial)))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('benchmark must return finite, positive milliseconds')
            report.update(status='passed', execution_time_ms=elapsed)
            valid.append((elapsed, candidate['key'], run))
        except Exception as error:
            # A poisoned CUDA context cannot safely try another implementation.
            if any(s in str(error).lower() for s in ('illegal memory access', 'device-side assert')):
                raise RuntimeError('fatal CUDA failure while evaluating a candidate') from error
            report['reason'] = f'{type(error).__name__}: {error}'
        reports.append(report)
    if not valid:
        error = RuntimeError('no candidate passed compilation, correctness and timing')
        error.reports = reports
        raise error
    _, key, run = min(valid, key=lambda item: item[0])
    return run, key, reports
