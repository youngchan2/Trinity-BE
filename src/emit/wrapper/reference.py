"""Evaluate the supplied equation; runtime applies each memory boundary's dtype.

Region equations come from a proven common summary. Explicit casts remain in
that equation; register producer expansion does not invent storage rounding.
"""
import torch


def _reference(expression, values):
    op = expression['op']
    if op == 'stores':
        local = dict(values)
        result = None
        for step in expression['steps']:
            result = _reference(step['expression'], local)
            dtype = torch.float32 if step['register'] else {
                'fp16': torch.float16, 'bf16': torch.bfloat16,
                'fp32': torch.float32}[step['dtype']]
            base = torch.empty(step['shape'], dtype=dtype, device=result.device)
            view = step['output']
            base.as_strided(view['shape'], view['strides'], view['offset']).copy_(result)
            local[view['value']] = base
        return result.reshape(expression['shape'])
    if op == 'view':
        view = expression['view']
        base = values[view['value']]
        return base.as_strided(view['shape'], view['strides'],
                               base.storage_offset() + view['offset']).float()
    if op == 'load':
        value = values[expression['value']]
        if expression.get('view_shape') is not None:
            value = value.view(expression['view_shape'])
        return value.float()
    if op == 'constant':
        return expression['value']
    args = [_reference(a, values) for a in expression['args']]
    if op == 'matmul':
        # No global TF32 flags are changed. Reference construction is not timed.
        return torch.matmul(args[0].double(), args[1].double()).float()
    if op == 'concat':
        return torch.cat(args, dim=expression['axis'])
    if op == 'add':
        return args[0] + args[1]
    if op == 'sub':
        return args[0] - args[1]
    if op == 'mul':
        return args[0] * args[1]
    if op == 'div':
        return args[0] / args[1]
    if op == 'sqr':
        return args[0] * args[0]
    if op == 'sqrt':
        return torch.sqrt(torch.as_tensor(args[0], dtype=torch.float32))
    if op == 'sigmoid':
        return torch.sigmoid(torch.as_tensor(args[0], dtype=torch.float32))
    if op == 'relu':
        return torch.clamp_min(torch.as_tensor(args[0], dtype=torch.float32), 0)
    if op == 'sum':
        return torch.sum(args[0], dim=expression['axis'])
    if op == 'unsqueeze':
        return torch.unsqueeze(args[0], expression['axis'])
    if op == 'squeeze':
        return torch.squeeze(args[0], expression['axis'])
    if op == 'max_reduce':
        return torch.amax(args[0], dim=expression['axis'])
    if op == 'min_reduce':
        return torch.amin(args[0], dim=expression['axis'])
    if op in ('exp', 'erf', 'abs'):
        return getattr(torch, op)(args[0])
    if op in ('maximum', 'minimum'):
        x = torch.as_tensor(args[0])
        return getattr(torch, op)(x, torch.as_tensor(args[1], device=x.device))
    if op == 'cast':
        name = expression['dtype'].lstrip('?')
        dtype = {'fp16': torch.float16, 'float16': torch.float16,
                 'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
                 'fp32': torch.float32, 'float32': torch.float32}[name]
        return args[0].to(dtype).float()
    raise ValueError(f'unknown reference operator: {op}')
