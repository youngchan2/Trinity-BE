class Executable:
    """Ordered Triton fallback launches for a scheduled program.

    The provider preserves regions and handles its own constexpr tuning. No
    comparison with other providers is claimed for this program path.
    """
    def __init__(self):
        self.manifest = _MANIFEST
        self.reports = [{'provider': 'triton', 'mode': 'scheduled_fallback',
                         'comparison': 'not_performed',
                         'kernels': _MANIFEST['kernels']}]

    def _arguments(self, inputs):
        if set(inputs) != {b['name'] for b in self.manifest['inputs']}:
            raise ValueError('input names differ from the program bindings')
        values, arguments = {}, {}
        for binding in self.manifest['inputs']:
            value = inputs[binding['name']]
            if binding['value'] in values and values[binding['value']] is not value:
                raise ValueError('input aliases must refer to the same tensor object')
            values[binding['value']] = value
            arguments[binding['argument']] = value
        return arguments

    def __call__(self, inputs):
        with torch.no_grad():
            return forward(**self._arguments(inputs))


def prepare(inputs, *, providers=None, rtol=1e-2, atol=1e-2, repeats=10):
    if providers is not None and (isinstance(providers, str) or 'triton' not in providers):
        raise ValueError('this scheduled program requires the Triton fallback provider')
    executable = Executable()
    executable._arguments(inputs)
    return executable
