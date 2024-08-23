import numpy as np
import torch
import matplotlib.pyplot as plt
from help_functions import find_surrogate_reduced_model, find_surrogate_reduced_correlated_models_invCDF
from help_functions import find_normalizing_flow, find_normalizing_flow_1D, find_normalizing_flow_spline, find_normalizing_flow_invCDF
from itertools import permutations
import os
import argparse
from optuna_tuning import tune_hyperparameter
from data_normalization import write_normalized_data, write_unnormalized_data

# QoI details
QoI_LF_name = 'mean_flow:lca1:BC_lca1'
QoI_HF_name = 'max_osi_sten_lad'

# Number of pilot samples
num_pilot_samples_to_use = 250

# Hyperparameters

#hyperparameter_tuning = True # True/False/filename with tuned hyperparameters
hyperparameter_tuning = 'hyperparameters/hyperparameters_together.txt' # True/False/filename with tuned hyperparameters

# Configuration options

save = True
plot = True

together = True
sequential = True
absolute_value = False

#repeated_trials = 200 # 0/1: Run the method once; >1: Run multiple times
repeated_trials = 0 # 0/1: Run the method once; >1: Run multiple times

base_path = os.path.abspath("./")

# Settings

#torch.manual_seed(2024)
#np.random.seed(2024)
torch.set_default_tensor_type(torch.DoubleTensor)

# Parser

parser = argparse.ArgumentParser()

parser.add_argument('--DR', type=int, default=1, help="Reduced dimension")
parser.add_argument('--FT', type=str, default="invCDF", help="Normalizing flow type")

args = parser.parse_args()

dim_reduced = args.DR
flow_type = args.FT

if (together and flow_type != "invCDF"):
    raise RuntimeError("together==True and flow_type != 'invCDF'")

config_string = "_DR" + str(dim_reduced) + "_FT" + str(flow_type) \
                + "_together" + str(int(together)) + "_abs" + str(int(absolute_value))

# Default hyperparameters

k_splits = 5
max_evals = 100

activation = torch.nn.Tanh()
layers_surrogate = 2
neurons_surrogate = 4
layers_AE = 2
neurons_AE = 4
lr = 1e-3
gamma = 1
alpha = 1.0

layers_NF_AE = 2
neurons_NF_AE = 8
lr_NF_AE = 1e-3
gamma_NF_AE = 1

layers_max = 4
neurons_max = 16
lr_min = 1e-4
lr_max = 1e-2
gamma_min = 0.999
gamma_max = 1
epochs = 10000

if hyperparameter_tuning:
    hyperparameters_name = 'hyperparameters' + config_string + ".log"

if (repeated_trials > 1):
    surrogate_f_error = []
    surrogate_g_error = []
    initial_correlation = []
    final_correlation = []
    plot = False
    trials_filename = "trials" + config_string + ".log"
else:
    repeated_trials = 1


for trial_idx in range(repeated_trials):
    
    if (repeated_trials > 1):
        print("\n Running trial " + str(trial_idx))
        trial_idx_str = "_"+str(trial_idx).zfill(4)
    else:
        trial_idx_str = ""
    
    # Write data files
    write_normalized_data(base_path, QoI_LF_name, QoI_HF_name, num_pilot_samples_to_use, trial_idx_str)

    # Read data
    data = torch.from_numpy(np.genfromtxt("data/parameters_normalized"+trial_idx_str+".csv", delimiter=','))
    f_output = torch.from_numpy(np.genfromtxt("data/QoI_HF_normalized"+trial_idx_str+".csv", delimiter=','))
    g_output = torch.from_numpy(np.genfromtxt("data/QoI_LF_normalized"+trial_idx_str+".csv", delimiter=','))
    data_propagation = torch.from_numpy(np.genfromtxt("data/parameters_propagation_normalized"+trial_idx_str+".csv", delimiter=','))
    
    cov_matrix = np.cov(f_output.numpy(), g_output.numpy())
    rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])

    if (repeated_trials > 1):
        initial_correlation.append(rho)

    print("Correlation coefficient: " + str(rho))

    # Plot initial correlation
    if plot:
        plt.figure()
        plt.scatter(f_output, g_output)
        plt.show()

    if hyperparameter_tuning:
    # Hyperparameter tuning

        if (hyperparameter_tuning == 1): # Tune hyperparameters
            hyperparameters = tune_hyperparameter(data, data, f_output, g_output, dim_reduced, activation, flow_type, layers_max, neurons_max, lr_min, lr_max, gamma_min, gamma_max, epochs, k_splits, max_evals, hyperparameters_name, together, sequential, absolute_value)
        
        elif isinstance(hyperparameter_tuning, str): # Read tuned hyperparameters
            hyperparameters = dict((line.split(':')[0], float(line.split(':')[1])) for line in open(hyperparameter_tuning)) 
            # Convert below parameters to int
            for key in ['layers_surrogate', 'neurons_surrogate', 'layers_AE', 'neurons_AE']:
                hyperparameters[key] = int(hyperparameters.get(key))
        
        else:
            raise RuntimeError("ERROR: Invalid input for hyperparameter_tuning. Should be True/False or filename with saved hyperparameters.")
        
        neurons_surrogate = hyperparameters['neurons_surrogate']
        layers_surrogate = hyperparameters['layers_surrogate']
        neurons_AE = hyperparameters['neurons_AE']
        layers_AE = hyperparameters['layers_AE']
        lr = hyperparameters['lr']
        gamma = hyperparameters['gamma']
        if together:
            alpha = hyperparameters['alpha']
        
        if flow_type != 'invCDF':
            if flow_type == 'RealNVP':
                neurons_NF_AE = hyperparameters['neurons_NF_AE']
                layers_NF_AE = hyperparameters['layers_NF_AE']
            lr_NF_AE = hyperparameters['lr_NF_AE']
            gamma_NF_AE = hyperparameters['gamma_NF_AE']
        
    # Surrogate and reduced model

    print("Finding surrogate and reduced models ...")

    if together:
        f_surrogate, model_f, g_surrogate, model_g, _ = find_surrogate_reduced_correlated_models_invCDF(data, data, f_output, g_output, 'together', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, alpha, epochs, sequential, absolute_value)
    else:
        f_surrogate, model_f, _ = find_surrogate_reduced_model(data, f_output, 'f', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, absolute_value)
        g_surrogate, model_g, _ = find_surrogate_reduced_model(data, g_output, 'g', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, absolute_value)

    error_f = (100*torch.norm(f_output - f_surrogate(data))/torch.norm(f_output)).item()
    error_g = (100*torch.norm(g_output - g_surrogate(data))/torch.norm(g_output)).item()

    print("Error surrogate f: " + str(error_f) + " %")
    print("Error surrogate g: " + str(error_g) + " %")

    if (repeated_trials > 1):
        surrogate_f_error.append(error_f)
        surrogate_g_error.append(error_g)

    # Normalizing flow for AE

    print("Finding normalizing flows for AE ...")

    data_reduced_f = model_f.encode(data).detach()
    data_reduced_g = model_g.encode(data).detach()
        
    if flow_type == 'spline':
        
        T_f, T_inverse_f, _ = find_normalizing_flow_spline(data_reduced_f, epochs, 'f', lr_NF_AE, gamma_NF_AE)
        T_g, T_inverse_g, _ = find_normalizing_flow_spline(data_reduced_g, epochs, 'g', lr_NF_AE, gamma_NF_AE)
        
    elif flow_type == 'RealNVP':

        if dim_reduced == 1:
            T_f, T_inverse_f, _ = find_normalizing_flow_1D(data_reduced_f, layers_NF_AE, neurons_NF_AE, epochs, 'f', lr_NF_AE, gamma_NF_AE)
            T_g, T_inverse_g, _ = find_normalizing_flow_1D(data_reduced_g, layers_NF_AE, neurons_NF_AE, epochs, 'g', lr_NF_AE, gamma_NF_AE)
        else:
            T_f, T_inverse_f, _ = find_normalizing_flow(data_reduced_f, layers_NF_AE, neurons_NF_AE, epochs, 'f', lr_NF_AE, gamma_NF_AE)
            T_g, T_inverse_g, _ = find_normalizing_flow(data_reduced_g, layers_NF_AE, neurons_NF_AE, epochs, 'g', lr_NF_AE, gamma_NF_AE)
            
    elif flow_type == 'invCDF' and dim_reduced == 1:
        
        T_f, T_inverse_f = find_normalizing_flow_invCDF(data_reduced_f)
        T_g, T_inverse_g = find_normalizing_flow_invCDF(data_reduced_g)
        
    if plot and dim_reduced == 1:
        
        z_standardGaussian = torch.normal(0,1, size = (1000,1))

        plt.figure()
        plt.hist(data_reduced_f[:,0], color = 'r', label = 'data',alpha = 0.4, density=True)
        plt.hist(T_f(z_standardGaussian).detach()[:,0], color = 'b', label = 'NF',alpha = 0.4, density=True)
        plt.title('Distribution latent space f')
        plt.legend()

        plt.figure()
        plt.hist(T_inverse_f(data_reduced_f).detach()[:,0], alpha = 0.4, density=True)
        plt.title('Check standard Gaussian f')

        plt.figure()
        plt.hist(data_reduced_g[:,0], color = 'r', label = 'data',alpha = 0.4, density=True)
        plt.hist(T_g(z_standardGaussian).detach()[:,0], color = 'b', label = 'NF',alpha = 0.4, density=True)
        plt.title('Distribution latent space g')
        plt.legend()

        plt.figure()
        plt.hist(T_inverse_g(data_reduced_g).detach()[:,0], alpha = 0.4, density=True)
        plt.title('Check standard Gaussian g')
        
    # Find best ordering

    print("Finding best ordering for AE ...")

    if dim_reduced > 1:
        
        p_max = list(range(dim_reduced))
        rho_max = 0
        z = T_inverse_f(model_f.encode(data))
        for p in list(permutations(range(dim_reduced))):
            p = list(p)
            g_output_new = g_surrogate(model_g.decode(T_g(z[:,p]))).detach().numpy()
            cov_matrix = np.cov(f_output, g_output_new)
            rho_fg = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])
            if np.abs(rho_fg) > np.abs(rho_max):
                rho_max = rho_fg
                p_max = p
                
    else:
        
        p_max = 0
        
    # New LF data

    if p_max == 0:
        data_g_reduced = model_g.decode(T_g(T_inverse_f(model_f.encode(data)))).detach().numpy()
    else:
        data_g_reduced = model_g.decode(T_g(T_inverse_f(model_f.encode(data))[:,p_max])).detach().numpy()

    if save:
        if p_max == 0:
            data_g_reduced_propagation = model_g.decode(T_g(T_inverse_f(model_f.encode(data_propagation)))).detach().numpy()
        else:
            data_g_reduced_propagation = model_g.decode(T_g(T_inverse_f(model_f.encode(data_propagation))[:,p_max])).detach().numpy()
        
        if not os.path.exists("./results"):
            os.mkdir("./results")
        
        # Write normalized data
        np.savetxt("results/new_parameters_LF_normalized_AE"+config_string+trial_idx_str+".csv", data_g_reduced, delimiter=",")
        np.savetxt("results/new_parameters_propagation_LF_normalized_AE"+config_string+trial_idx_str+".csv", data_g_reduced_propagation, delimiter=",")
        
        # Write un-normalized data
        write_unnormalized_data(base_path, config_string, trial_idx_str)

    g_output_new = g_surrogate(torch.from_numpy(data_g_reduced)).detach()

    cov_matrix = np.cov(f_output, g_output_new)
    rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])

    print("Correlation coefficient: " + str(rho))

    if (repeated_trials > 1):
        final_correlation.append(rho)

if plot:
    plt.figure()
    plt.scatter(f_output, g_output_new)
    plt.title(r'$\rho = ' + str(rho) + r'$')
    plt.show()

if (repeated_trials > 1):
    np.savetxt(trials_filename, np.column_stack((np.array(surrogate_f_error),np.array(surrogate_g_error),np.array(initial_correlation),np.array(final_correlation))), header = 'surrogate_f_error, surrogate_g_error, initial_correlation, final_correlation')
