"""Reference evaluation preserves expression order and each store's dtype boundary."""
import torch


def _reference(expression, values):
    op = expression['op']
    if op == 'load':
        return values[expression['value']].float()
    if op == 'constant':
        return expression['value']
    args = [_reference(a, values) for a in expression['args']]
    if op == 'matmul':
        # No global TF32 flags are changed. Reference construction is not timed.
        return torch.matmul(args[0].double(), args[1].double()).float()
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
    raise ValueError(f'unknown reference operator: {op}')
