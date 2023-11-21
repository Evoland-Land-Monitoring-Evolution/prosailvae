#!/usr/bin/env python3
"""
Created on Sun Nov 20 17:05:00 2022

@author: yoel
"""
import torch


def pos_expi(x):
    k = torch.arange(1, 40, dtype=x.dtype)
    r = torch.cumprod(x.unsqueeze(-1) * k / torch.square(k + 1), dim=-1)
    ga = torch.tensor([0.5772156649015328], dtype=x.dtype)
    y = ga + torch.log(x) + x * (1 + (r).sum(-1))
    return y


def ein(x):
    k = torch.arange(1, 40, dtype=x.dtype)
    r = torch.cumprod(-x.unsqueeze(-1) * k / torch.square(k + 1), dim=-1)
    return x * (1 + (r).sum(-1))


def e1(x):
    ga = torch.tensor([0.5772156649015328], dtype=x.dtype)
    y = -ga - torch.log(x) + ein(x)
    return y


def neg_expi(x):
    return -e1(-x)


def expi(x):
    ei = torch.zeros_like(x)
    x_l0 = x < 0
    x_g0 = x > 0
    ei[x_l0] = neg_expi(x[x_l0])
    ei[x_g0] = pos_expi(x[x_g0])
    return ei
