#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import argparse
import os
from analytical_examples import get_model
from help_functions import tune_hyperparameter, compute_reduction, find_CDF, compute_MC, compute_MFMC

#%% Parameters

torch.manual_seed(2024)
torch.set_default_tensor_type(torch.DoubleTensor)

#%% Info

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=1000, help="Number of data")
args = parser.parse_args()
N = args.N

save = True
hyperparameter_tuning = True

iterations = 100
M = 1000
w = 0.01

#%% Model and data

name_HF = "Q_HF"
name_LF = "Q_LF"

d, f_HF = get_model(name_HF)
d, f_LF = get_model(name_LF)

data = 2*torch.rand((N, d)) - 1
f_data_HF = f_HF(data)
f_data_LF = f_LF(data)

if save:
    
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/" + name_HF):
        os.mkdir("./results/" + name_HF)
    if not os.path.exists("./results/" + name_LF):
        os.mkdir("./results/" + name_LF)

#%% Dimensionality reduction

r = 1
activation = torch.nn.Tanh()
epochs = 10000

if hyperparameter_tuning:

    k_splits = 5
    max_evals = 100
    layers_max = 4
    neurons_max = 16
    hyperparameters_HF = tune_hyperparameter(d, data, f_data_HF, r, activation, layers_max, neurons_max, epochs, k_splits, max_evals, name_HF)
    hyperparameters_LF = tune_hyperparameter(d, data, f_data_LF, r, activation, layers_max, neurons_max, epochs, k_splits, max_evals, name_LF)
    
    layers_AE_HF = hyperparameters_HF['layers_AE']
    neurons_AE_HF = hyperparameters_HF['neurons_AE']
    layers_surrogate_HF = hyperparameters_HF['layers_surrogate']
    neurons_surrogate_HF = hyperparameters_HF['neurons_surrogate']
    
    layers_AE_LF = hyperparameters_LF['layers_AE']
    neurons_AE_LF = hyperparameters_LF['neurons_AE']
    layers_surrogate_LF = hyperparameters_LF['layers_surrogate']
    neurons_surrogate_LF = hyperparameters_LF['neurons_surrogate']
    
else:
    
    layers_AE_HF, layers_AE_LF = [2]*2
    neurons_AE_HF, neurons_AE_LF = [8]*2
    layers_surrogate_HF, layers_surrogate_LF = [2]*2
    neurons_surrogate_HF, neurons_surrogate_LF = [8]*2
    
MSE_reduced_HF = np.empty(iterations)
MSE_reduced_surrogate_HF = np.empty(iterations)
MAE_reduced_HF = np.empty(iterations)
MAE_reduced_surrogate_HF = np.empty(iterations)

MSE_reduced_LF = np.empty(iterations)
MSE_reduced_surrogate_LF = np.empty(iterations)
MAE_reduced_LF = np.empty(iterations)
MAE_reduced_surrogate_LF = np.empty(iterations)

rho = np.empty(iterations)
rho_AE = np.empty(iterations)
rho_AE_ideal = np.empty(iterations)
MC = np.empty(iterations)
MFMC = np.empty(iterations)
MFMC_AE = np.empty(iterations)
    
for it in range(iterations):
    
    print("Iteration: " + str(it+1) + '/' + str(iterations))

    autoencoder_HF, surrogate_HF, _ = compute_reduction(d, data, f_data_HF, r, activation, layers_AE_HF, neurons_AE_HF, layers_surrogate_HF, neurons_surrogate_HF, epochs, show=False)
    autoencoder_LF, surrogate_LF, _ = compute_reduction(d, data, f_data_LF, r, activation, layers_AE_LF, neurons_AE_LF, layers_surrogate_LF, neurons_surrogate_LF, epochs, show=False)
    
    f_reduced_HF = lambda x: f_HF(autoencoder_HF(x)).detach()
    f_reduced_surrogate_HF = lambda x: torch.squeeze(surrogate_HF(autoencoder_HF.encoder(x))).detach()
    
    f_reduced_LF = lambda x: f_LF(autoencoder_LF(x)).detach()
    f_reduced_surrogate_LF = lambda x: torch.squeeze(surrogate_LF(autoencoder_LF.encoder(x))).detach()

    data_new = 2*torch.rand((M, d)) - 1
    
    MSE_reduced_HF[it] = torch.mean((f_HF(data_new) - f_reduced_HF(data_new))**2)
    MSE_reduced_surrogate_HF[it] = torch.mean((f_HF(data_new) - f_reduced_surrogate_HF(data_new))**2)
    MAE_reduced_HF[it] = torch.mean(torch.abs(f_HF(data_new) - f_reduced_HF(data_new)))
    MAE_reduced_surrogate_HF[it] = torch.mean(torch.abs(f_HF(data_new) - f_reduced_surrogate_HF(data_new)))
    
    MSE_reduced_LF[it] = torch.mean((f_LF(data_new) - f_reduced_LF(data_new))**2)
    MSE_reduced_surrogate_LF[it] = torch.mean((f_LF(data_new) - f_reduced_surrogate_LF(data_new))**2)
    MAE_reduced_LF[it] = torch.mean(torch.abs(f_LF(data_new) - f_reduced_LF(data_new)))
    MAE_reduced_surrogate_LF[it] = torch.mean(torch.abs(f_LF(data_new) - f_reduced_surrogate_LF(data_new)))
    
    F_ideal_HF, Finv_ideal_HF = find_CDF(torch.unsqueeze(f_HF(data_new), 1).detach())
    F_ideal_LF, Finv_ideal_LF = find_CDF(torch.unsqueeze(f_LF(data_new), 1).detach())
    uniform_samples = torch.rand((M,1))
    cov_matrix_ideal = np.cov(torch.squeeze(Finv_ideal_HF(uniform_samples)), torch.squeeze(Finv_ideal_LF(uniform_samples)))
    rho_AE_ideal[it] = cov_matrix_ideal[0,1]/np.sqrt(cov_matrix_ideal[0,0]*cov_matrix_ideal[1,1])
    
    F_HF, Finv_HF = find_CDF(autoencoder_HF.encoder(data_new).detach())
    F_LF, Finv_LF = find_CDF(autoencoder_LF.encoder(data_new).detach())
    f_LF_AE = lambda x: f_LF(autoencoder_LF.decoder(Finv_LF(F_HF(autoencoder_HF.encoder(x))))).detach()

    MC[it] = compute_MC(f_HF, d, M)
    MFMC[it], rho[it] = compute_MFMC(f_HF, f_LF, d, M, w)
    MFMC_AE[it], rho_AE[it] = compute_MFMC(f_HF, f_LF_AE, d, M, w)

#%% Saving

if save:
        
    np.savetxt("results/MC.txt", MC)
    np.savetxt("results/MFMC.txt", MFMC)
    np.savetxt("results/MFMC_AE.txt", MFMC_AE)
    np.savetxt("results/rho.txt", rho)
    np.savetxt("results/rho_AE.txt", rho_AE)
    np.savetxt("results/rho_AE_ideal.txt", rho_AE_ideal)
    
    np.savetxt("results/" + name_HF + "/hyperparameters_HF.txt", np.array([layers_AE_HF, neurons_AE_HF, layers_surrogate_HF, neurons_surrogate_HF]).astype(int), fmt='%i')
    np.savetxt("results/" + name_HF + "/MSE_reduced_HF.txt", MSE_reduced_HF)
    np.savetxt("results/" + name_HF + "/MSE_reduced_surrogate_HF.txt", MSE_reduced_surrogate_HF)
    np.savetxt("results/" + name_HF + "/MAE_reduced_HF.txt", MAE_reduced_HF)
    np.savetxt("results/" + name_HF + "/MAE_reduced_surrogate_HF.txt", MAE_reduced_surrogate_HF)
    
    np.savetxt("results/" + name_LF + "/hyperparameters_LF.txt", np.array([layers_AE_LF, neurons_AE_LF, layers_surrogate_LF, neurons_surrogate_LF]).astype(int), fmt='%i')
    np.savetxt("results/" + name_LF + "/MSE_reduced_LF.txt", MSE_reduced_LF)
    np.savetxt("results/" + name_LF + "/MSE_reduced_surrogate_LF.txt", MSE_reduced_surrogate_LF)
    np.savetxt("results/" + name_LF + "/MAE_reduced_LF.txt", MAE_reduced_LF)
    np.savetxt("results/" + name_LF + "/MAE_reduced_surrogate_LF.txt", MAE_reduced_surrogate_LF)
    