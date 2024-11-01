#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import os
from help_functions import tune_hyperparameter, compute_reduction, find_CDF, compute_MC, compute_MFMC
import importlib.util
import sys
import json



#   parser = argparse.ArgumentParser()
#   parser.add_argument('--N', type=int, default=1000, help="Number of data")
#   args = parser.parse_args()
#   N = args.N # Number of training samples

#   use_random_seed = 2024
#   iterations = 2
#   M = 1000 # Number of testing samples
#   cost_ratio = 0.01 # Ratio of LF:HF cost
#   r = 1 # Reduced dimension
#   epochs = 10000
#   activation = torch.nn.Tanh()

#   save = True
#   hyperparameter_tuning = False

#   # Path to a .py file which includes a get_model() function for the HF and LF models
#   analytical_example_path = "../Hartmann/"

#   # Path to location for outputs/results
#   base_path = "./"

#   # Names of QoIs
#   name_HF = "Hartmann_flow_velocity"
#   name_LF = "Hartmann_magnetic_field"

def run_using_model(config):

    # -------------------------------
    # User inputs
    # -------------------------------

#   parser = argparse.ArgumentParser()
#   parser.add_argument('input_file', help="Configuration file in JSON format")
#   args = parser.parse_args()
#   with open(args.input_file) as f:
#       config = json.load(f)

    N = config["number_of_training_samples"]
    M = config["number_of_testing_samples"]
    use_random_seed = config["random_seed"]
    iterations = config["number_of_iterations"]
    cost_ratio = config["model"]["cost_ratio"]
    r = config["reduced_dimension"]
    epochs = config["epochs"]
    save = config["save"]
    hyperparameter_tuning = config["hyperparameter_tuning"]
    analytical_example_path = config["model"]["model_path"]
    name_HF = config["model"]["HF_QoI_name"]
    name_LF = config["model"]["LF_QoI_name"]
    base_path = "./"
    activation = torch.nn.Tanh()

    # -------------------------------
    # Create model and data
    # -------------------------------

    # Import functions for the specific analytical function
    spec = importlib.util.spec_from_file_location("analytical_example", analytical_example_path+"/model.py")
    analytical_example = importlib.util.module_from_spec(spec)
    sys.modules["analytical_example"] = analytical_example
    spec.loader.exec_module(analytical_example)

    if use_random_seed:
        torch.manual_seed(use_random_seed)
    torch.set_default_tensor_type(torch.DoubleTensor)

    # Get dimensionality and model
    d, f_HF = analytical_example.get_model(name_HF)
    d, f_LF = analytical_example.get_model(name_LF)

    # Set up training data
    data = 2*torch.rand((N, d)) - 1
    f_data_HF = f_HF(data)
    f_data_LF = f_LF(data)

    # -------------------------------
    # Tune hyperparameters
    # -------------------------------

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
       
        # Default hyperparameters
        layers_AE_HF, layers_AE_LF = [2]*2
        neurons_AE_HF, neurons_AE_LF = [8]*2
        layers_surrogate_HF, layers_surrogate_LF = [2]*2
        neurons_surrogate_HF, neurons_surrogate_LF = [8]*2
        
    # -------------------------------
    # Arrays to save results
    # -------------------------------

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

    # -------------------------------
    # Dimensionality reduction
    # -------------------------------
        
    for i in range(iterations):
        
        print("Iteration: " + str(i+1) + '/' + str(iterations))
        
        # Compute autoencoders and surrogate models for dimensionality reduction
        # --- for HF model
        autoencoder_HF, surrogate_HF, _ = compute_reduction(d, data, f_data_HF, r, activation, layers_AE_HF, neurons_AE_HF, layers_surrogate_HF, neurons_surrogate_HF, epochs, show=False)
        # --- for LF model
        autoencoder_LF, surrogate_LF, _ = compute_reduction(d, data, f_data_LF, r, activation, layers_AE_LF, neurons_AE_LF, layers_surrogate_LF, neurons_surrogate_LF, epochs, show=False)
       
        # Create functions that operate on full-dimensional and reduced-dimensional data
        f_reduced_HF = lambda x: f_HF(autoencoder_HF(x)).detach()
        f_reduced_surrogate_HF = lambda x: torch.squeeze(surrogate_HF(autoencoder_HF.encoder(x))).detach()
        
        f_reduced_LF = lambda x: f_LF(autoencoder_LF(x)).detach()
        f_reduced_surrogate_LF = lambda x: torch.squeeze(surrogate_LF(autoencoder_LF.encoder(x))).detach()

        # Create new testing data
        data_new = 2*torch.rand((M, d)) - 1
       
        # Compute error metrics
        # --- Mean squared error between HF model evaluated on test data and decode(encode(test data))
        MSE_reduced_HF[i] = torch.mean((f_HF(data_new) - f_reduced_HF(data_new))**2)
        # --- Mean squared error between HF model evaluated on test data and surrogate model evaluated on latent space of test data
        MSE_reduced_surrogate_HF[i] = torch.mean((f_HF(data_new) - f_reduced_surrogate_HF(data_new))**2)
        # --- Mean absolute error between HF model evaluated on test data and decode(encode(test data))
        MAE_reduced_HF[i] = torch.mean(torch.abs(f_HF(data_new) - f_reduced_HF(data_new)))
        # --- Mean absolute error between HF model evaluated on test data and surrogate model evaluated on latent space of test data
        MAE_reduced_surrogate_HF[i] = torch.mean(torch.abs(f_HF(data_new) - f_reduced_surrogate_HF(data_new)))
       
        # --- Above errors computed on LF model
        MSE_reduced_LF[i] = torch.mean((f_LF(data_new) - f_reduced_LF(data_new))**2)
        MSE_reduced_surrogate_LF[i] = torch.mean((f_LF(data_new) - f_reduced_surrogate_LF(data_new))**2)
        MAE_reduced_LF[i] = torch.mean(torch.abs(f_LF(data_new) - f_reduced_LF(data_new)))
        MAE_reduced_surrogate_LF[i] = torch.mean(torch.abs(f_LF(data_new) - f_reduced_surrogate_LF(data_new)))
       
        # Compute inverse CDF normalizing flow and correlation between HF and LF models for an ideal case
        F_ideal_HF, Finv_ideal_HF = find_CDF(torch.unsqueeze(f_HF(data_new), 1).detach())
        F_ideal_LF, Finv_ideal_LF = find_CDF(torch.unsqueeze(f_LF(data_new), 1).detach())
        uniform_samples = torch.rand((M,1))
        cov_matrix_ideal = np.cov(torch.squeeze(Finv_ideal_HF(uniform_samples)), torch.squeeze(Finv_ideal_LF(uniform_samples)))
        rho_AE_ideal[i] = cov_matrix_ideal[0,1]/np.sqrt(cov_matrix_ideal[0,0]*cov_matrix_ideal[1,1])
        
        # Compute inverse CDF normalizing flow for the current data
        F_HF, Finv_HF = find_CDF(autoencoder_HF.encoder(data_new).detach())
        F_LF, Finv_LF = find_CDF(autoencoder_LF.encoder(data_new).detach())
        
        # Create a function for the new/resampled LF model
        f_LF_AE = lambda x: f_LF(autoencoder_LF.decoder(Finv_LF(F_HF(autoencoder_HF.encoder(x))))).detach()

        # Compute Monte Carlo estimate
        MC[i] = compute_MC(f_HF, d, M)
        # Compute multifidelity Mote Carlo estimate
        MFMC[i], rho[i] = compute_MFMC(f_HF, f_LF, d, M, cost_ratio)
        # Compute multifidelity Mote Carlo estimate with dimensionality reduction
        MFMC_AE[i], rho_AE[i] = compute_MFMC(f_HF, f_LF_AE, d, M, cost_ratio)

    # -------------------------------
    # Save results
    # -------------------------------

    if save:
        
        if not os.path.exists(base_path + "/results"):
            os.mkdir(base_path+"/results")
        if not os.path.exists(base_path + "/results/" + name_HF):
            os.mkdir(base_path + "/results/" + name_HF)
        if not os.path.exists(base_path + "/results/" + name_LF):
            os.mkdir(base_path + "/results/" + name_LF)
            
        np.savetxt(base_path + "/results/MC.txt", MC)
        np.savetxt(base_path + "/results/MFMC.txt", MFMC)
        np.savetxt(base_path + "/results/MFMC_AE.txt", MFMC_AE)
        np.savetxt(base_path + "/results/rho.txt", rho)
        np.savetxt(base_path + "/results/rho_AE.txt", rho_AE)
        np.savetxt(base_path + "/results/rho_AE_ideal.txt", rho_AE_ideal)
        
        np.savetxt(base_path + "/results/" + name_HF + "/hyperparameters_HF.txt", np.array([layers_AE_HF, neurons_AE_HF, layers_surrogate_HF, neurons_surrogate_HF]).astype(int), fmt='%i')
        np.savetxt(base_path + "/results/" + name_HF + "/MSE_reduced_HF.txt", MSE_reduced_HF)
        np.savetxt(base_path + "/results/" + name_HF + "/MSE_reduced_surrogate_HF.txt", MSE_reduced_surrogate_HF)
        np.savetxt(base_path + "/results/" + name_HF + "/MAE_reduced_HF.txt", MAE_reduced_HF)
        np.savetxt(base_path + "/results/" + name_HF + "/MAE_reduced_surrogate_HF.txt", MAE_reduced_surrogate_HF)
        
        np.savetxt(base_path + "/results/" + name_LF + "/hyperparameters_LF.txt", np.array([layers_AE_LF, neurons_AE_LF, layers_surrogate_LF, neurons_surrogate_LF]).astype(int), fmt='%i')
        np.savetxt(base_path + "/results/" + name_LF + "/MSE_reduced_LF.txt", MSE_reduced_LF)
        np.savetxt(base_path + "/results/" + name_LF + "/MSE_reduced_surrogate_LF.txt", MSE_reduced_surrogate_LF)
        np.savetxt(base_path + "/results/" + name_LF + "/MAE_reduced_LF.txt", MAE_reduced_LF)
        np.savetxt(base_path + "/results/" + name_LF + "/MAE_reduced_surrogate_LF.txt", MAE_reduced_surrogate_LF)
        
