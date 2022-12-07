#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:33:35 2022

@author: yoel
"""
max_iter=10000
iter_i = 0
sim = torch.tensor([1.6846e+00, 2.0355e+01, 1.8098e+00, 6.6344e-04, 4.0115e-03, 9.8562e-03,
        7.4946e+00, 7.7200e+01, 1.0272e-05, 3.8622e-02, 3.0127e-01]).unsqueeze(0).unsqueeze(2)
angles = torch.tensor([ 44.7698,   2.6547, 133.4176]).unsqueeze(0)
while iter_i<max_iter:
    print(iter_i)
    ds = torch.randn((1,11,1))*sim*0.0001
    da = torch.randn((1,3))*angles*0.001
    # decoding
    rec = self.decode(sim+ds, angles+da)
    if torch.isnan(rec).any():
        iter_i=max_iter
        print(ds)
        print(da)
        torch.save(sim+ds,"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/prosailvae/nan_sim.pt")
        torch.save(angles+da,"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/prosailvae/nan_angle.pt")
        break
    iter_i+=1
    
sim = torch.load("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/prosailvae/nan_sim.pt")
angles = torch.load("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/prosailvae/nan_angle.pt")