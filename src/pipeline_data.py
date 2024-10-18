import numpy as np
import torch
from utils_model_data import find_surrogate_reduced_model, find_surrogate_reduced_correlated_models_invCDF, tune_hyperparameters
from utils_model_data import load_surrogate_reduced_model, load_surrogate_reduced_correlated_models_invCDF
from utils_model_data import find_normalizing_flow, find_normalizing_flow_1D, find_normalizing_flow_spline, find_normalizing_flow_invCDF
from utils_model_data import write_normalized_data, write_unnormalized_data
from utils_model_data import read_simulation_data
from itertools import permutations
import os
from utils_general import read_json_entry
import scipy.stats as stats

# -------------------------------

def run_using_data(config):
        
    # -------------------------------                                 
    # User inputs                                                     
    # -------------------------------                                 

    use_random_seed = read_json_entry(config, "random_seed", False)
    save = read_json_entry(config, "save", True)
    # Repeated trials
    # 0,1: Run the method once / >1: Run multiple trials
    repeated_trials = read_json_entry(config, "number_of_iterations", 1)
    # True OR False OR filename with tuned hyperparameters
    hyperparameter_tuning = read_json_entry(config, "hyperparameter_tuning", False)
    # Number of epochs for training
    epochs = read_json_entry(config, "epochs")
    
    # Path to simulation data
    data_path = read_json_entry(config["model"], "data_path")
    # Reduced dimension
    dim_reduced = read_json_entry(config["model"], "reduced_dimension", 1) 
    # QoI details
    QoI_HF_name = read_json_entry(config["model"], "HF_QoI_name")
    QoI_LF_name = read_json_entry(config["model"], "LF_QoI_name")
    # Number of pilot samples to use for constructing shared space and surrogate model
    # Positive integer: Use specified number of samples OR -1: Use all samples
    num_pilot_samples_to_use = read_json_entry(config["model"], "num_pilot_samples", -1)
    # Type of normalizing flow. Default is inverse CDF
    flow_type = read_json_entry(config["model"], "normalizing_flow_type", "invCDF")
    cost_ratio = read_json_entry(config["model"], "cost_ratio")
    # Train models or load saved models? Either False or a string with path to the saved models "NN_models*.pt"
    load_NN_models = read_json_entry(config["model"], "load_NN_models", False)
    # Train HF and LF models together?
    together = read_json_entry(config["model"], "train_together", True)
    # Train sequentially, i.e. first train LF and HF models separately and initialize with those weights when training them together. 
    # Default value is true - works much better than directly training HF and LF models together
    sequential = read_json_entry(config["model"], "train_sequentially", True)

    # Path to location for outputs/results
    base_path = "./"
    activation = torch.nn.Tanh()

    # -------------------------------
    # Check user inputs
    # -------------------------------

    if (together and flow_type != "invCDF"):
        raise RuntimeError("ERROR: together == True and flow_type != 'invCDF'. HF and LF model reduction can be computed together only if the normalizing flow is inverse CDF.")

    if (flow_type == "invCDF" and dim_reduced != 1):
        raise RuntimeError("ERROR: flow_type == 'invCDF' and dim_reduced != 1. Inverse CDF normalizing flow can only be used with reduced dimension of 1.")

    # -------------------------------
    # General book-keeping
    # -------------------------------

    if use_random_seed:
        torch.manual_seed(use_random_seed)
        np.random.seed(use_random_seed)
    torch.set_default_tensor_type(torch.DoubleTensor)

    config_string = "_DR" + str(dim_reduced) + "_FT" + str(flow_type) \
                    + "_together" + str(int(together))

    if hyperparameter_tuning:
        hyperparameters_name = 'hyperparameters' + config_string + ".log"

    if (repeated_trials > 1):
        surrogate_f_error = []
        surrogate_g_error = []
        initial_correlation = []
        final_correlation = []
        #plot = False
        trials_filename = "trials" + config_string + ".log"
    else:
        repeated_trials = 1

    if isinstance(load_NN_models, str):
        NN_model_path = load_NN_models

    if repeated_trials > 1:
        print('WARNING: For "model_type" : "data", "number_of_iterations" > 1 is only implemented in the construction of the neural active manifold. This will construct the reduced dimension and surrogate model the specified number of times using the pilot samples. However, it is not implemented when performing uncertainty quantification in --resampled_data mode. If multiple iterations of the multi-fidelity uncertainty quantification are required, users can call this program multiple times from an external program/loop. Please read the documentation.')

    # -------------------------------
    # Run dimensionality reduction
    # -------------------------------

    for trial_idx in range(repeated_trials):
        
        if (repeated_trials > 1):
            print("\n Running trial " + str(trial_idx))
            trial_idx_str = "_"+str(trial_idx).zfill(4)
        else:
            trial_idx_str = ""
        
        # Write data files
        write_normalized_data(base_path, data_path, QoI_LF_name, QoI_HF_name, num_pilot_samples_to_use, trial_idx_str)

        # Read data
        f_data = torch.from_numpy(np.genfromtxt(base_path + "/results/parameters_normalized"+trial_idx_str+".csv", delimiter=','))
        g_data = torch.from_numpy(np.genfromtxt(base_path + "/results/parameters_normalized"+trial_idx_str+".csv", delimiter=','))
        f_output = torch.from_numpy(np.genfromtxt(base_path + "/results/QoI_HF_normalized"+trial_idx_str+".csv", delimiter=','))
        g_output = torch.from_numpy(np.genfromtxt(base_path + "/results/QoI_LF_normalized"+trial_idx_str+".csv", delimiter=','))
        data_propagation = torch.from_numpy(np.genfromtxt(base_path + "/results/parameters_propagation_normalized"+trial_idx_str+".csv", delimiter=','))

        # Compute correlation between LF and HF models
        cov_matrix = np.cov(f_output.numpy(), g_output.numpy())
        rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])

        if (repeated_trials > 1):
            initial_correlation.append(rho)

        print("Correlation coefficient: " + str(rho))

        # -------------------------------
        # Hyperparameter tuning
        # -------------------------------
        
        if hyperparameter_tuning:

            k_splits = 5
            max_evals = 100
            layers_max = 4
            neurons_max = 16
            lr_min = 1e-4
            lr_max = 1e-2
            gamma_min = 0.999
            gamma_max = 1

            if (hyperparameter_tuning == 1): 
                # Tune hyperparameters
                hyperparameters = tune_hyperparameters(base_path, f_data, g_data, f_output, g_output, dim_reduced, activation, flow_type, layers_max, neurons_max, lr_min, lr_max, gamma_min, gamma_max, epochs, k_splits, max_evals, hyperparameters_name, together, sequential)
            
            elif isinstance(hyperparameter_tuning, str): 
                # Read tuned hyperparameters from a file
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
            
        else:

            # Default hyperparameters
            neurons_surrogate = 4
            layers_surrogate = 2
            neurons_AE = 4
            layers_AE = 2
            lr = 1e-3
            gamma = 1
            
            if together:
                alpha = 1.0
            
            if flow_type != 'invCDF':
                if flow_type == 'RealNVP':
                    neurons_NF_AE = 8
                    layers_NF_AE = 2
                lr_NF_AE = 1e-3
                gamma_NF_AE = 1
        
        # -------------------------------
        # Surrogate and reduced model
        # -------------------------------

        if not load_NN_models:
            
            print("Finding surrogate and reduced models.")

            if together:
                f_surrogate, model_f, g_surrogate, model_g, loss , model_state_dict = find_surrogate_reduced_correlated_models_invCDF(
                        f_data, g_data, f_output, g_output, 'together', dim_reduced, activation, layers_surrogate, neurons_surrogate, 
                        layers_AE, neurons_AE, lr, gamma, alpha, epochs, sequential)
            
            else:
                f_surrogate, model_f, loss_f, model_state_dict_f = find_surrogate_reduced_model(f_data, f_output, 'f', dim_reduced, 
                        activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs)
                g_surrogate, model_g, loss_g, model_state_dict_g = find_surrogate_reduced_model(g_data, g_output, 'g', dim_reduced, 
                        activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs)

                model_state_dict = {"HF" : model_state_dict_f, "LF" : model_state_dict_g}

        else:

            print("Loading surrogate and reduced models from " + NN_model_path)
            
            _, dim_f = f_data.shape
            _, dim_g = g_data.shape
            if together:
                f_surrogate, model_f, g_surrogate, model_g = load_surrogate_reduced_correlated_models_invCDF(NN_model_path, 
                        dim_f, dim_g, dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE)
            
            else:
                f_surrogate, model_f = load_surrogate_reduced_model(NN_model_path, "HF", dim_f, dim_reduced, activation, 
                        layers_surrogate, neurons_surrogate, layers_AE, neurons_AE)
                g_surrogate, model_g = load_surrogate_reduced_model(NN_model_path, "LF", dim_g, dim_reduced, activation, 
                        layers_surrogate, neurons_surrogate, layers_AE, neurons_AE)
        
        # Error between simulation outputs and surrogate model outputs
        error_f = (100*torch.norm(f_output - f_surrogate(f_data))/torch.norm(f_output)).item()
        error_g = (100*torch.norm(g_output - g_surrogate(g_data))/torch.norm(g_output)).item()

        print("Error surrogate f: " + str(error_f) + " %")
        print("Error surrogate g: " + str(error_g) + " %")

        if (repeated_trials > 1):
            surrogate_f_error.append(error_f)
            surrogate_g_error.append(error_g)

        # -------------------------------
        # Normalizing flow for AE
        # -------------------------------

        print("Finding normalizing flows for AE.")

        data_reduced_f = model_f.encode(f_data).detach()
        data_reduced_g = model_g.encode(g_data).detach()
            
        if flow_type == 'spline':
            
            T_f, T_inverse_f, _ = find_normalizing_flow_spline(data_reduced_f, epochs, 'f', lr_NF_AE, gamma_NF_AE)
            T_g, T_inverse_g, _ = find_normalizing_flow_spline(data_reduced_g, epochs, 'g', lr_NF_AE, gamma_NF_AE)
            
        elif flow_type == 'RealNVP':

            if dim_reduced == 1:
                T_f, T_inverse_f, _ = find_normalizing_flow_1D(data_reduced_f, layers_NF_AE, neurons_NF_AE, epochs, 
                        'f', lr_NF_AE, gamma_NF_AE)
                T_g, T_inverse_g, _ = find_normalizing_flow_1D(data_reduced_g, layers_NF_AE, neurons_NF_AE, epochs, 
                        'g', lr_NF_AE, gamma_NF_AE)
            else:
                T_f, T_inverse_f, _ = find_normalizing_flow(data_reduced_f, layers_NF_AE, neurons_NF_AE, epochs, 'f', 
                        lr_NF_AE, gamma_NF_AE)
                T_g, T_inverse_g, _ = find_normalizing_flow(data_reduced_g, layers_NF_AE, neurons_NF_AE, epochs, 'g', 
                        lr_NF_AE, gamma_NF_AE)
                
        elif flow_type == 'invCDF' and dim_reduced == 1:
            
            T_f, T_inverse_f = find_normalizing_flow_invCDF(data_reduced_f)
            T_g, T_inverse_g = find_normalizing_flow_invCDF(data_reduced_g)
            
        # -------------------------------------------
        # Find best ordering if reduced dimension > 1
        # -------------------------------------------

        print("Finding best ordering for AE ...")

        if dim_reduced > 1:
            
            p_max = list(range(dim_reduced))
            rho_max = 0
            z = T_inverse_f(model_f.encode(f_data))
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
            
        # -------------------------------
        # Compute new/resampled LF inputs
        # -------------------------------

        if p_max == 0:
            data_g_reduced = model_g.decode(T_g(T_inverse_f(model_f.encode(f_data)))).detach().numpy()
        else:
            data_g_reduced = model_g.decode(T_g(T_inverse_f(model_f.encode(f_data))[:,p_max])).detach().numpy()

        # -------------------------------
        # Save data
        # -------------------------------
        if save:
            if p_max == 0:
                data_g_reduced_propagation = model_g.decode(T_g(T_inverse_f(model_f.encode(data_propagation)))).detach().numpy()
            else:
                data_g_reduced_propagation = model_g.decode(T_g(T_inverse_f(model_f.encode(data_propagation))[:,p_max])).detach().numpy()
            
            if not os.path.exists(base_path + "/results"):
                os.mkdir(base_path + "/results")
            
            # Write normalized resampled inputs
            np.savetxt(base_path + "/results/new_parameters_LF_normalized_AE"+config_string+trial_idx_str+".csv", 
                    data_g_reduced, delimiter=",")
            np.savetxt(base_path + "/results/new_parameters_propagation_LF_normalized_AE"+config_string+trial_idx_str+".csv", 
                    data_g_reduced_propagation, delimiter=",")
            
            # Un-normalize resampled inputs
            write_unnormalized_data(base_path, config_string, trial_idx_str)

            # Save autoencoders and surrogate models
            if not load_NN_models:
                if together:
                    torch.save(model_state_dict, base_path + "/results/NN_models"+trial_idx_str+".pt")
                else:
                    torch.save(model_state_dict_HF, base_path + "/results/NN_models_HF"+trial_idx_str+".pt")
                    torch.save(model_state_dict_LF, base_path + "/results/NN_models_LF"+trial_idx_str+".pt")

            # Save loss history
            if not os.path.exists(base_path + "/results/losses"):
                os.mkdir(base_path + "/results/losses")
            
            if together:
                # Losses for the second stage where HF and LF are trained together
                np.savetxt(base_path + "/results/losses/test_loss"+trial_idx_str+".dat", np.asarray(loss[0]))
                np.savetxt(base_path + "/results/losses/train_loss"+trial_idx_str+".dat", np.asarray(loss[1]))
                if sequential:
                    np.savetxt(base_path + "/results/losses/train_loss_HF"+trial_idx_str+".dat", np.asarray(loss[2][0]))
                    np.savetxt(base_path + "/results/losses/test_loss_HF"+trial_idx_str+".dat", np.asarray(loss[2][1]))
                    np.savetxt(base_path + "/results/losses/train_loss_LF"+trial_idx_str+".dat", np.asarray(loss[3][0]))
                    np.savetxt(base_path + "/results/losses/test_loss_LF"+trial_idx_str+".dat", np.asarray(loss[3][1]))
            else:
                np.savetxt(base_path + "/results/losses/test_loss_HF"+trial_idx_str+".dat", np.asarray(loss_f[0]))
                np.savetxt(base_path + "/results/losses/train_loss_HF"+trial_idx_str+".dat", np.asarray(loss_f[1]))
                np.savetxt(base_path + "/results/losses/test_loss_LF"+trial_idx_str+".dat", np.asarray(loss_g[0]))
                np.savetxt(base_path + "/results/losses/train_loss_LF"+trial_idx_str+".dat", np.asarray(loss_g[1]))

        # --------------------------------------
        # Compute new correlation b/w HF and LF 
        # Note: Using the surrogate LF model
        # --------------------------------------
        g_output_new = g_surrogate(torch.from_numpy(data_g_reduced)).detach()

        cov_matrix = np.cov(f_output, g_output_new)
        rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])

        print("Correlation coefficient: " + str(rho))

        # Save correlation for each trial (if running repeated trials) 
        if (repeated_trials > 1):
            final_correlation.append(rho)

    # --------------------------------------------
    # Save the results if running repeated trials
    # --------------------------------------------
    if (repeated_trials > 1):
        np.savetxt(trials_filename, np.column_stack((np.array(surrogate_f_error), np.array(surrogate_g_error), 
            np.array(initial_correlation),np.array(final_correlation))), 
            header = 'surrogate_f_error, surrogate_g_error, initial_correlation, final_correlation')

    # --------------------------------------------
    # Print message for user
    # --------------------------------------------
    print("\n--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    print("The neural active manifold and reduced-order surrogate models have been constructed and saved in " + base_path + "/results.")
    print("The next step is to run the low-fidelity model using the resampled inputs and save the results in the `simulations` directory.")
    print("Please find the new resampled inputs (parameters) for the pilot sample in " + base_path + "/results/new_parameters_LF_AE"+config_string+trial_idx_str + ".csv")
    print("Please find the new resampled inputs (parameters) for the propagation samples in " + base_path + "/results/new_parameters_propagation_LF_AE"+config_string+trial_idx_str + ".csv")
    print("The simulation data should be stored in the same format as the outputs from the original pilot/propagation samples.")
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------\n")

# -------------------------------

def process_resampled_sim_data(config):

    # Path to location for outputs/results
    base_path = "./"
#   base_path = os.path.abspath(base_path)

    # Path to simulation data
    data_path = read_json_entry(config["model"], "data_path")
    
    # QoI details
    QoI_HF_name = read_json_entry(config["model"], "HF_QoI_name")
    QoI_LF_name = read_json_entry(config["model"], "LF_QoI_name")
    cost_ratio = read_json_entry(config["model"], "cost_ratio")
    
    save = read_json_entry(config, "save", True)
    # Repeated trials
    # 0,1: Run the method once / >1: Run multiple trials
    repeated_trials = read_json_entry(config, "number_of_iterations", 1)

    if repeated_trials > 1:
        print('WARNING: number_of_iterations > 1 is not implemented when running in --resampled_data mode with "model_type" : "data". Please see the documentation.')

#   # File paths
#   base_path = "./"
#   base_path = os.path.abspath(base_path)

    parameters_file = base_path + data_path + "/simulations/all_param_values.json"
    all_0d_data_file = base_path + data_path + "/simulations/all_0d_data.json"
    all_3d_data_file = base_path + data_path + "/simulations/all_3d_data.json"
    #new_0d_data_file = base_path + "/simulations/all_0d_data_AE.json"
    #all_0d_data_file_prop = base_path + "/simulations/all_0d_data_propagation.json"
    #new_0d_data_file_prop = base_path + "/simulations/all_0d_data_AE_propagation.json"

#   # QoI details
#   QoI_LF_name = 'mean_flow:lca1:BC_lca1'
#   QoI_HF_name = 'max_osi_sten_lad'
#   cost_ratio = 10/(7*96*60*60 + 1*60*60) # LF:10 sec ; HF:7 hr on 96 procs + 1 hr on 1 proc 

#   # Options
#   plot = False
#   save = False

#   # Check for repeated trials
#   repeated_trials = 1
#   trial_files = glob.glob("trials_*.log")
#   if len(trial_files) == 1:
#       trials_header = ' '.join(list(np.genfromtxt(trial_files[0], comments=None, max_rows=1, dtype='U')))
#       trials_data = np.genfromtxt(trial_files[0], comments=None, skip_header=1)
#       repeated_trials = trials_data.shape[0]
#       final_correlation = np.zeros(repeated_trials)
#       MC_mean = np.zeros(repeated_trials)
#       plot = False
#   elif len(trial_files) > 1:
#       raise RuntimeError("ERROR: len(glob.glob('trials_*.log')) > 1")

#   if repeated_trials != read_json_entry(config, "number_of_iterations", 1):
#       raise RuntimeError("ERROR: The number of repeated trials specified in the JSON file is not the same 
#               as those found in trials_*.log.")

#   for trial_idx in range(repeated_trials):

#       if (repeated_trials > 1):
#           print("\n Trial: "  +str(trial_idx))
#           trial_idx_str = "_"+str(trial_idx).zfill(4)
#       else:
#           trial_idx_str = ""

    # Read data
    #new_0d_data_file = base_path + data_path + "/simulations/all_0d_data_AE"+trial_idx_str+".json"
    new_0d_data_file = base_path + data_path + "/simulations/all_0d_data_AE.json"
    all_0d_data_file_prop = base_path + data_path + "/simulations/all_0d_data_propagation.json"
    new_0d_data_file_prop = base_path + data_path + "/simulations/all_0d_data_AE_propagation.json"
#   samples, parameters, QoI_LF, QoI_HF, _, new_QoI_LF_AE, _, _ = read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_file, all_0d_data_file, all_3d_data_file, None, new_0d_data_file)
    
    samples, parameters, QoI_LF, QoI_HF, _, QoI_LF_AE, QoI_LF_prop, QoI_LF_prop_AE = \
    read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_file, all_0d_data_file, all_3d_data_file, 
            None, new_0d_data_file, all_0d_data_file_prop, new_0d_data_file_prop)

    num_samples = len(samples)

    # If not using all samples
    #use_sample_idx_file = base_path + "/results/sample_idxs"+trial_idx_str+".csv"
    use_sample_idx_file = base_path + "/results/sample_idxs.csv"
    if os.path.isfile(use_sample_idx_file):
        sample_idxs = np.genfromtxt(use_sample_idx_file, dtype=int)
        QoI_LF = QoI_LF[sample_idxs]
        QoI_HF = QoI_HF[sample_idxs]
        parameters = parameters[sample_idxs,:]
        num_samples = len(sample_idxs)

    # Compute initial correlation
    cov_matrix = np.cov(QoI_HF, QoI_LF)
    rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])

    print("Original correlation coefficient: " + str(rho))

    # Compute final correlation
    cov_matrix_AE = np.cov(QoI_HF, QoI_LF_AE)
    rho_AE = cov_matrix_AE[0,1]/np.sqrt(cov_matrix_AE[0,0]*cov_matrix_AE[1,1])

    print("New correlation coefficient (AE): " + str(rho_AE))

#   if (repeated_trials > 1):
#       MC_mean[trial_idx] = np.mean(QoI_HF)
    MC_mean = np.mean(QoI_HF)

#   if plot:
#       # Initial correlation
#       plt.figure()
#       plt.scatter(QoI_HF, QoI_LF)
#       plt.title(r'$\rho = ' + str(rho) + r'$')
#       plt.show()
#       # Final correlation
#       plt.figure()
#       plt.scatter(QoI_HF, new_QoI_LF_AE)
#       plt.title(r'$\rho_{\mathrm{AE}} = ' + str(rho_AE) + r'$')
#       plt.show()

#   if (repeated_trials == 1):

    # Number of samples

    N_HF = num_samples
    N_LF = 10000

    N_HF_AE = N_HF
    N_LF_AE = N_LF
    N_budget = N_HF + cost_ratio*N_LF

    QoI_LF = np.append(QoI_LF, QoI_LF_prop[0:N_LF-N_HF])
    new_QoI_LF_AE = np.append(QoI_LF_AE, QoI_LF_prop_AE[0:N_LF-N_HF])

    # Verify QoI sizes

    if (N_HF != len(QoI_HF)):
        raise RuntimeError("ERROR: N_HF != len(QoI_HF)")

    if (N_LF != len(QoI_LF)):
        raise RuntimeError("ERROR: N_LF != len(QoI_LF)")

    if (N_LF != len(new_QoI_LF_AE)):
        raise RuntimeError("ERROR: N_LF != len(new_QoI_LF_AE)")

    # Find optimal allocation for given budget and correlation
    def find_optimal_allocation(budget, rho):
        coeff = np.sqrt((rho**2)/(cost_ratio*(1 - rho**2)))
        N_HF = round(budget/(1 + cost_ratio*coeff))
        N_LF = round((coeff*budget)/(1 + cost_ratio*coeff))
        return N_HF, N_LF, N_HF + cost_ratio*N_LF

    N_HF_optimal, N_LF_optimal, total_cost = find_optimal_allocation(N_HF, rho)
    print("Optimal number HF samples: " + str(N_HF_optimal))
    print("Optimal number LF samples: " + str(N_LF_optimal))

    N_HF_optimal_AE, N_LF_optimal_AE, total_cost = find_optimal_allocation(N_HF, rho_AE)
    print("Optimal number HF samples: " + str(N_HF_optimal_AE))
    print("Optimal number LF samples: " + str(N_LF_optimal_AE))

    # Monte Carlo

    print("\n--- Monte Carlo estimator ---")
    MC_mean = np.mean(QoI_HF[:N_HF])
    #MC_std = np.std(QoI_HF[:N_HF])/np.sqrt(N_budget)
    MC_std = np.std(QoI_HF[:N_HF])/np.sqrt(N_HF)

    print("Monte Carlo estimator mean: " + str(MC_mean))
    print("Monte Carlo estimator std: " + str(MC_std))

    # Multifidelity Monte Carlo

    print("\n--- Multifidelity Monte Carlo estimator ---")
    cov_matrix = np.cov(QoI_HF[:N_HF], QoI_LF[:N_HF])
    alpha = cov_matrix[0,1]/cov_matrix[1,1] # \rho*sqrt(Var[HF]/Var[LF]) = Cov[HF, LF]/Var[LF]
    rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])
    print("Correlation: " + str(rho))

    MFMC_mean = np.mean(QoI_HF[:N_HF]) - alpha*(np.mean(QoI_LF[:N_HF]) - np.mean(QoI_LF[:N_LF]))
    #MFMC_std = (np.std(QoI_HF[:N_HF])/np.sqrt(N_HF))*np.sqrt(1-((N_LF-N_HF)/N_LF)*rho**2)
    MFMC_std = MC_std*np.sqrt(1-((N_LF-N_HF)/N_LF)*rho**2)
    #MFMC_std_optimal = (np.std(QoI_HF[:N_HF])/np.sqrt(N_HF))*(np.sqrt(1 - rho**2) + np.sqrt(cost_ratio*(rho**2)))
    MFMC_std_optimal = MC_std*(np.sqrt(1 - rho**2) + np.sqrt(cost_ratio*(rho**2)))
    #MFMC_std_optimal2 = (np.std(QoI_HF[:N_HF])/np.sqrt(N_HF))*np.sqrt(1-((62511-234)/62511)*rho**2)

    print("Multifidelity Monte Carlo estimator mean: " + str(MFMC_mean))
    print("Multifidelity Monte Carlo estimator std: " + str(MFMC_std))
    print("Multifidelity Monte Carlo estimator std (optimal): " + str(MFMC_std_optimal))
    #print("Multifidelity Monte Carlo estimator std (optimal): " + str(MFMC_std_optimal2))

    # Multifidelity Monte Carlo AE

    print("\n--- Multifidelity Monte Carlo estimator with autoencoders ---")
    cov_matrix_AE = np.cov(QoI_HF[:N_HF_AE], new_QoI_LF_AE[:N_HF_AE])
    alpha_AE = cov_matrix_AE[0,1]/cov_matrix_AE[1,1]
    rho_AE = cov_matrix_AE[0,1]/np.sqrt(cov_matrix_AE[0,0]*cov_matrix_AE[1,1])
    print("Correlation: " + str(rho_AE))
    MFMC_AE_mean = np.mean(QoI_HF[:N_HF_AE]) - alpha_AE*(np.mean(new_QoI_LF_AE[:N_HF_AE]) - np.mean(new_QoI_LF_AE[:N_LF_AE]))
    #MFMC_AE_std = (np.std(QoI_HF[:N_HF_AE])/np.sqrt(N_HF_AE))*np.sqrt(1-((N_LF_AE-N_HF_AE)/N_LF_AE)*rho_AE**2)
    MFMC_AE_std = MC_std*np.sqrt(1-((N_LF_AE-N_HF_AE)/N_LF_AE)*rho_AE**2)
    #MFMC_AE_std_optimal = (np.std(QoI_HF[:N_HF_AE])/np.sqrt(N_HF_AE))*(np.sqrt(1 - rho_AE**2) + np.sqrt(cost_ratio*(rho_AE**2)))
    MFMC_AE_std_optimal = MC_std*(np.sqrt(1 - rho_AE**2) + np.sqrt(cost_ratio*(rho_AE**2)))
    #MFMC_AE_std_optimal2 = (np.std(QoI_HF[:N_HF_AE])/np.sqrt(233))*np.sqrt(1-((674797-233)/674797)*rho_AE**2)

    print("Multifidelity Monte Carlo estimator AE mean: " + str(MFMC_AE_mean))
    print("Multifidelity Monte Carlo estimator AE std: " + str(MFMC_AE_std))
    print("Multifidelity Monte Carlo estimator AE std (optimal): " + str(MFMC_AE_std_optimal))
    #print("Multifidelity Monte Carlo estimator AE std (optimal): " + str(MFMC_AE_std_optimal2))

#   if plot:

#       plt.figure()
#       mu = MC_mean
#       sigma = MC_std
#       x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#       plt.plot(x, stats.norm.pdf(x, mu, sigma), label="MC")
#       mu = MFMC_mean
#       sigma = MFMC_std
#       x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#       plt.plot(x, stats.norm.pdf(x, mu, sigma), label="MFMC")
#       mu = MFMC_mean
#       sigma = MFMC_std_optimal
#       x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#       plt.plot(x, stats.norm.pdf(x, mu, sigma), ls='--', label="MFMC optimal")
#       mu = MFMC_AE_mean
#       sigma = MFMC_AE_std
#       x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#       plt.plot(x, stats.norm.pdf(x, mu, sigma), label="MFMC-AE")
#       mu = MFMC_AE_mean
#       sigma = MFMC_AE_std_optimal
#       x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#       plt.plot(x, stats.norm.pdf(x, mu, sigma), ls='--', label="MFMC-AE optimal")
#       plt.xlabel(QoI_HF_name)
#       plt.legend()
#       plt.savefig("results/estimators.pdf")

#       fig, ax = plt.subplots()
#       x_pos = [1,2,3]
#       x_labels = ["MC", "MFMC", "MFMC-AE"]
#       means = np.array([MC_mean, MFMC_mean, MFMC_AE_mean])
#       std = np.array([MC_std, MFMC_std, MFMC_AE_std])
#       confidence = 0.99
#       k = np.sqrt(1.0/(1.0-confidence)) # Chebyshev's inequality: P(|X-\mu| >= k\sigma) = 1/k^2
#       k_99 = k
#       eb1 = ax.errorbar(x_pos, means, k*std, fmt='ks', markersize=7.0, capsize=5.0, ecolor='k', elinewidth=1.0)
#       eb1[-1][0].set_linestyle(':')
#       confidence = 0.95
#       k = np.sqrt(1.0/(1.0-confidence)) # Chebyshev's inequality: P(|X-\mu| >= k\sigma) = 1/k^2
#       k_95 = k
#       ax.errorbar(x_pos, means, k*std, fmt='ks', markersize=7.0, capsize=5.0, ecolor='k', elinewidth=2.0)
#       ax.set_xlim(x_pos[0]-1, x_pos[-1]+1)
#       ax.set_xticks(x_pos)
#       ax.set_xticklabels(x_labels)
#       ax.set_ylabel(QoI_HF_name)
#       plt.savefig("results/estimators_errorbar.pdf")

#       # Number of additional simulations to reduce variance
#       #std_reduction_range = np.linspace(0.01, 0.8, 10)
#       std_reduction_range = np.linspace(MFMC_AE_std_optimal/MC_std, 0.9, 10)
#       num_hf_mc = np.zeros_like(std_reduction_range)
#       num_hf_mfmc = np.zeros_like(std_reduction_range)
#       num_hf_mfmc_ae = np.zeros_like(std_reduction_range)
#       std_mfmc_fixed_HF_budget = np.zeros_like(std_reduction_range)
#       std_mfmc_fixed_HF_budget_ae = np.zeros_like(std_reduction_range)
#       std_mfmc_fixed_budget = np.zeros_like(std_reduction_range)
#       std_mfmc_fixed_budget_ae = np.zeros_like(std_reduction_range)
#       std_hf = np.std(QoI_HF[:N_HF])

#       # Find number of LF simulations for given standard deviation with following data: 
#       # number of HF samples (num_HF), correlation (rho), required Std. (std_reqd), Std. of HF samples (std_HF)
#       def find_num_LF_MFMC(num_HF, rho, std_reqd, std_HF):
#           std_MC = std_HF/np.sqrt(num_HF) # Monte Carlo SD
#           num_LF_MFMC = num_HF*(1.0-(1/rho**2)*(1.0-(std_reqd/std_MC)**2))**(-1)
#           return num_LF_MFMC

#       for i, std_reduc in enumerate(std_reduction_range):
#           # Required standard deviation
#           std_new = std_reduc*MC_std

#           # Required number of HF sims for MC
#           num_hf_mc[i] = (std_hf/std_new)**2 
#        
#           num_additional_HF = -1 # Add HF sims if target std. cannot be reached with only new LF sims
#           num_lf_mfmc = -1 # Negative value indicates target std. not reached
#           while (num_lf_mfmc < 0):
#               num_additional_HF = num_additional_HF + 1
#               num_lf_mfmc = find_num_LF_MFMC(N_HF+num_additional_HF, rho, std_new, std_hf)
#           #print(std_reduc, num_additional_HF, num_lf_mfmc, num_lf_mfmc*cost_ratio)
#           # Equivalent cost in HF sims
#           num_hf_mfmc[i] = N_HF + num_additional_HF + num_lf_mfmc*cost_ratio 
#           
#           num_additional_HF = -1 # Add HF sims if target std. cannot be reached with only new LF sims
#           num_lf_mfmc_ae = -1 # Negative value indicates target std. not reached
#           while (num_lf_mfmc_ae < 0):
#               num_additional_HF = num_additional_HF + 1
#               num_lf_mfmc_ae = find_num_LF_MFMC(N_HF+num_additional_HF, rho_AE, std_new, std_hf)
#           #print(std_reduc, num_additional_HF, num_lf_mfmc_ae, num_lf_mfmc_ae*cost_ratio)
#           # Equivalent cost in HF sims
#           num_hf_mfmc_ae[i] = N_HF + num_additional_HF + num_lf_mfmc_ae*cost_ratio 
#           
#           # Convert MC cost into additional LF simulations beyond N_HF pilot
#           # Fixed HF budget (plus additional LF simulations)
#           num_propagation_LF = (num_hf_mc[i] - N_HF)/cost_ratio
#           # Compute std. of estimator with N_HF + num_propagation_LF sims
#           std_mfmc_fixed_HF_budget[i] = MC_std*np.sqrt(1-((num_propagation_LF - N_HF)/num_propagation_LF)*rho**2)
#           std_mfmc_fixed_HF_budget_ae[i] = MC_std*np.sqrt(1-((num_propagation_LF - N_HF)/num_propagation_LF)*rho_AE**2)
#           
#           # Use MC cost in optimal allocation for MFMC and MFMC-AE
#           num_HF, num_LF, budget_used = find_optimal_allocation(num_hf_mc[i], rho)
#           std_mfmc_fixed_budget[i] = (std_hf/np.sqrt(num_HF))*np.sqrt(1-((num_LF-num_HF)/num_LF)*rho**2)
#           num_HF, num_LF, budget_used = find_optimal_allocation(num_hf_mc[i], rho_AE)
#           std_mfmc_fixed_budget_ae[i] = (std_hf/np.sqrt(num_HF))*np.sqrt(1-((num_LF-num_HF)/num_LF)*rho_AE**2)

#       plt.figure()
#       plt.plot(num_hf_mc, (std_reduction_range*MC_std)**2, '-o', label="MC")
#       #plt.plot(num_hf_mc, (std_mfmc_fixed_HF_budget)**2, '-o', label="MFMC")
#       plt.plot(num_hf_mc, (std_mfmc_fixed_budget)**2, '-o', label="MFMC")
#       #plt.plot(num_hf_mc, (std_mfmc_fixed_HF_budget_ae)**2, '-o', label="MFMC-AE")
#       plt.plot(num_hf_mc, (std_mfmc_fixed_budget_ae)**2, '-o', label="MFMC-AE")
#       plt.yscale('log')
#       plt.xscale('log')
#       plt.ylabel("Estimator variance")
#       plt.xlabel("HF-equivalent simulations")
#       plt.legend()
#       plt.savefig("results/std_fixed_budget.pdf")

#       plt.figure()
#       plt.plot(num_hf_mc - N_HF, (std_reduction_range)**2, '-o', label="MC")
#       plt.plot(num_hf_mfmc - N_HF, (std_reduction_range)**2, '-o', label="MFMC")
#       plt.plot(num_hf_mfmc_ae - N_HF, (std_reduction_range)**2, '-o', label="MFMC-AE")
#       plt.yscale('log')
#       plt.xscale('log')
#       plt.ylabel("Variance reduction factor")
#       plt.xlabel("Additional HF-equivalent simulations")
#       plt.legend()
#       plt.savefig("results/budget_fixed_std.pdf")

    #   # Optimal allocation
    #   std_reduction_range = np.linspace(0.75*MFMC_AE_std_optimal/MC_std, 0.9, 10)
    #   num_hf_mc_optimal = np.zeros_like(std_reduction_range)
    #   num_hf_mfmc_optimal = np.zeros_like(std_reduction_range)
    #   num_hf_mfmc_ae_optimal = np.zeros_like(std_reduction_range)
    #   std_hf = np.std(QoI_HF[:N_HF])

    #   for i, std_reduc in enumerate(std_reduction_range):
    #       std_new = std_reduc*MC_std
    #       num_hf_mc[i] = (std_hf/std_new)**2 

    #       num_hf, num_lf, budget_used = find_optimal_allocation(num_hf_mc[i], rho_AE)
    #       print(num_hf, num_lf, budget_used, num_hf_mc[i])
    #       print(MC_std*(np.sqrt(1 - rho_AE**2) + np.sqrt(cost_ratio*(rho_AE**2))))
    #       print(MC_std*np.sqrt(1-((num_lf-num_hf)/num_lf)*rho_AE**2))

    if save:
        with open(base_path+'results/MC.dat', 'w') as f:
            f.write("Mean = " + str(MC_mean) + "\n")
            f.write("Std = " + str(MC_std) + "\n")
        with open(base_path+'results/MFMC.dat', 'w') as f:
            f.write("Mean = " + str(MFMC_mean) + "\n")
            f.write("Std = " + str(MFMC_std) + "\n")
            f.write("Correlation = " + str(rho) + "\n")
            f.write("Optimal number HF samples: " + str(N_HF_optimal) + "\n")
            f.write("Optimal number LF samples: " + str(N_LF_optimal) + "\n")
            f.write("Optimal std = " + str(MFMC_std_optimal))
        with open(base_path+'results/MFMC_AE.dat', 'w') as f:
            f.write("Mean = " + str(MFMC_AE_mean) + "\n")
            f.write("Std = " + str(MFMC_AE_std) + "\n")
            f.write("Correlation = " + str(rho_AE) + "\n")
            f.write("Optimal number HF samples: " + str(N_HF_optimal_AE) + "\n")
            f.write("Optimal number LF samples: " + str(N_LF_optimal_AE) + "\n")
            f.write("Optimal std = " + str(MFMC_AE_std_optimal) + "\n")
#       np.savez("results/propagation.npz", means=means, std=std, k_99=k_99, k_95=k_95)
#       np.savez("results/cost_analysis.npz", std_reduction_range=std_reduction_range, MC_std=MC_std, 
#               std_mfmc_fixed_budget=std_mfmc_fixed_budget, std_mfmc_fixed_budget_ae=std_mfmc_fixed_budget_ae,
#               N_HF=N_HF, num_hf_mc=num_hf_mc, num_hf_mfmc=num_hf_mfmc, num_hf_mfmc_ae=num_hf_mfmc_ae)

#       else:
        #final_correlation[trial_idx] = rho_AE
        final_correlation = rho_AE
