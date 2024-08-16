from inspect import isfunction
import torch

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def extract_t_1(a, t, x_shape):
    b, *_ = t.shape
    t_clamped = torch.clamp(t, min=0)
    gathered = a.gather(-1, t_clamped)
    zeros = torch.zeros_like(gathered)
    mask = t == -1
    out = torch.where(mask, zeros, gathered)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
