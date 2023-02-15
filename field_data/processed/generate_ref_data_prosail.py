#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:59:35 2022

@author: yoel
"""
import numpy as np
import os
import prosail
fname = r"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailpython/tests/data/REFL_CAN.txt"
save_dir = r"/home/yoel/Documents/Dev/PROSAIL-VAE/"
w, resv, rdot, rsot, rddt, rsdt = np.loadtxt(fname, unpack=True)

list_rsot = []
list_rddt = []
list_rsdt = []
list_rdot = []
n_samples = 100


n_range = [0.8, 2.5]
cab_range = [0, 80]
car_range = [0, 20]
cbrown_range = [0, 1]
cw_range = [0, 200/1000]
cm_range = [0, 200/1000]
lai_range = [0, 10]
lidfa_range = [0, 90]
tts_range = [0, 90]
tto_range = [0, 90]
psi_range = [0, 360]
n_list = np.zeros((n_samples,1))
cab_list = np.zeros((n_samples,1))
car_list = np.zeros((n_samples,1))
cbrown_list = np.zeros((n_samples,1))
cw_list = np.zeros((n_samples,1))
cm_list = np.zeros((n_samples,1))
lai_list = np.zeros((n_samples,1))
lidfa_list = np.zeros((n_samples,1))
hspot_list = np.zeros((n_samples,1))
psoil_list = np.ones((n_samples,1))
rsoil_list = np.ones((n_samples,1))
tts_list = np.zeros((n_samples,1))
tto_list = np.zeros((n_samples,1))
psi_list = np.zeros((n_samples,1))


for i in range(n_samples):
    u = np.random.uniform(0,1,size=11)
    n = u[0] * (n_range[1] - n_range[0]) + n_range[0]
    cab = u[1] * (cab_range[1] - cab_range[0]) + cab_range[0]
    car = u[2] * (car_range[1] - car_range[0]) + car_range[0]
    cbrown = u[3] * (cbrown_range[1] - cbrown_range[0]) + cbrown_range[0]
    cw = u[4] * (cw_range[1] - cw_range[0]) + cw_range[0]
    cm = u[5] * (cm_range[1] - cm_range[0]) + cm_range[0]
    lai = u[6] * (lai_range[1] - lai_range[0]) + lai_range[0]
    lidfa = u[7] * (lidfa_range[1] - lidfa_range[0]) + lidfa_range[0]
    hspot = 0.01
    tts = u[8] * (tts_range[1] - tts_range[0]) + tts_range[0]
    tto = u[9] * (tto_range[1] - tto_range[0]) + tto_range[0]
    psi = u[10] * (psi_range[1] - psi_range[0]) + psi_range[0]
    rsoil = rsoil_list[i]
    psoil = psoil_list[i]
    rsot, rddt, rsdt, rdot = prosail.run_prosail(
                n=n,
                cab=cab,
                car=car,
                cbrown=cbrown,
                cw=cw,
                cm=cm,
                lai=lai,
                lidfa=lidfa,
                hspot=hspot,
                tts=tts,
                tto=tto,
                psi=psi,
                typelidf=2,
                lidfb=-0.15,
                rsoil=rsoil,
                psoil=psoil,
                factor="ALL",
            )
    list_rsot.append(rsot.reshape(1,-1))
    list_rddt.append(rddt.reshape(1,-1))
    list_rsdt.append(rsdt.reshape(1,-1))
    list_rdot.append(rdot.reshape(1,-1))
    n_list[i]=n
    cab_list[i]=cab
    car_list[i]=car
    cbrown_list[i]=cbrown
    cw_list[i]=cw
    cm_list[i]=cm
    lai_list[i]=lai
    lidfa_list[i]=lidfa
    hspot_list[i]=hspot
    tts_list[i]=tts
    tto_list[i]=tto
    psi_list[i]=psi

param_array = np.concatenate([n_list,
                              cab_list,
                              car_list,
                              cbrown_list,
                              cw_list,
                              cm_list,
                              lai_list,
                              lidfa_list,
                              hspot_list,
                              rsoil_list, 
                              psoil_list,
                              tts_list,
                              tto_list,
                              psi_list], axis=1)
np.save(save_dir + 'prosail_params.npy', param_array)
np.save(save_dir + 'prosail_rsot.npy', np.concatenate(list_rsot, axis=0))
np.save(save_dir + 'prosail_rddt.npy', np.concatenate(list_rddt, axis=0))
np.save(save_dir + 'prosail_rsdt.npy', np.concatenate(list_rsdt, axis=0))
np.save(save_dir + 'prosail_rdot.npy', np.concatenate(list_rdot, axis=0))          
                                          
np.save(save_dir + 'prosail_w.npy', w.reshape(-1,1))

list_refl = []
list_trans = []
for i in range(n_samples):
    u = np.random.uniform(0,1,size=11)
    n = u[0] * (n_range[1] - n_range[0]) + n_range[0]
    cab = u[1] * (cab_range[1] - cab_range[0]) + cab_range[0]
    car = u[2] * (car_range[1] - car_range[0]) + car_range[0]
    cbrown = u[3] * (cbrown_range[1] - cbrown_range[0]) + cbrown_range[0]
    cw = u[4] * (cw_range[1] - cw_range[0]) + cw_range[0]
    cm = u[5] * (cm_range[1] - cm_range[0]) + cm_range[0]

    w, refl, trans = prosail.run_prospect(
                n,
                cab,
                car,
                cbrown,
                cw,
                cm,
                prospect_version="5",
            )
    list_refl.append(refl.reshape(1,-1))
    list_trans.append(trans.reshape(1,-1))

    n_list[i]=n
    cab_list[i]=cab
    car_list[i]=car
    cbrown_list[i]=cbrown
    cw_list[i]=cw
    cm_list[i]=cm

param_array = np.concatenate([n_list,
                                cab_list,
                                car_list,
                                cbrown_list,
                                cw_list,
                                cm_list,
                                ], axis=1)
np.save(save_dir + 'prospect5_params.npy', param_array)
np.save(save_dir + 'prospect5_refl.npy', np.concatenate(list_refl, axis=0))
np.save(save_dir + 'prospect5_trans.npy', np.concatenate(list_trans, axis=0))        


# rsot, rddt, rsdt, rdot = prosail.run_prosail(
#             1.5,
#             40.0,
#             8.0,
#             0.0,
#             0.01,
#             0.009,
#             3.0,
#             -0.35,
#             0.01,
#             30.0,
#             10.0,
#             0.0,
#             typelidf=2,
#             lidfb=-0.15,
#             rsoil=1.0,
#             psoil=1.0,
#             factor="ALL",
#         )
# rr = np.concatenate([w.reshape(-1,1), resv.reshape(-1,1), 
#                      rdot.reshape(-1,1), rsot.reshape(-1,1), 
#                      rddt.reshape(-1,1), rsdt.reshape(-1,1)],axis=1)
# # np.savetxt("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailpython/tests/data/REFL_CAN_sim1.txt", 
# #            rr, fmt=['%.0f','%.6f','%.6f','%.6f','%.6f','%.6f'],delimiter=' ')
# rsot, rddt, rsdt, rdot = prosail.run_prosail(
#         2,
#         40.0,
#         8.0,
#         0.0,
#         0.01,
#         0.009,
#         3.0,
#         -0.35,
#         0.01,
#         30.0,
#         10.0,
#         0.0,
#         typelidf=2,
#         lidfb=-0.15,
#         rsoil=1.0,
#         psoil=1.0,
#         factor="ALL",
#     )
# rr = np.concatenate([w.reshape(-1,1), resv.reshape(-1,1), 
#                      rdot.reshape(-1,1), rsot.reshape(-1,1), 
#                      rddt.reshape(-1,1), rsdt.reshape(-1,1)],axis=1)
# # np.savetxt("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailpython/tests/data/REFL_CAN_sim2.txt",
# #            rr, fmt=['%.0f','%.6f','%.6f','%.6f','%.6f','%.6f'],delimiter=' ')
