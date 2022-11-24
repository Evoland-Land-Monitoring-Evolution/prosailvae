#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:51:41 2022

@author: yoel
"""
import torch
import matplotlib.pyplot as plt

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + torch.erf((x-mu)/(sigma*2**0.5))) 

a = torch.tensor(0)
b = torch.tensor(1)

mu = torch.tensor(0)
sigma=torch.tensor(0.005)
x = torch.arange(0,1.001,0.001)
e = torch.erfinv(x)
plt.figure()
plt.plot(x,e)
x = torch.arange(0.999,1.00001,0.00001)
dedx = 0.5 * torch.sqrt(torch.tensor(torch.pi)) * torch.exp(torch.erfinv(x)**2)
plt.figure()
plt.plot(x,dedx)

phi_b = normal_cdf(b, mu, sigma)
phi_a = normal_cdf(a, mu, sigma)
Z = phi_b-phi_a
phi_m1 = torch.sqrt(torch.tensor(2)) * torch.erfinv(2* (x * Z + phi_a)-1)