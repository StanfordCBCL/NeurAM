import numpy as np
import pandas as pd
import json
import os
from help_functions import read_simulation_data

def write_normalized_data(base_path, QoI_LF_name, QoI_HF_name, num_pilot_samples_to_use, trial_name_str = ""):

    parameters_file = base_path + "/simulations/all_param_values.json"
    all_0d_data_file = base_path + "/simulations/all_0d_data.json"
    all_3d_data_file = base_path + "/simulations/all_3d_data.json"
    parameters_file_propagation = base_path + "/simulations/all_param_values_propagation.json"

    # Read data

    samples, parameters, QoI_LF, QoI_HF, parameters_propagation, _, _, _ = read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_file, all_0d_data_file, all_3d_data_file, parameters_file_propagation)
    num_params = parameters.shape[1]

    # Number of pilot samples to use
    if num_pilot_samples_to_use != -1: # If -1, use all samples
        num_samples = parameters.shape[0]
        use_sample_idxs = np.random.default_rng().integers(0, num_samples, num_pilot_samples_to_use)
        QoI_LF = QoI_LF[use_sample_idxs]
        QoI_HF = QoI_HF[use_sample_idxs]
        parameters = parameters[use_sample_idxs,:]

    # Values parameters

    R_min = 0.5
    R_max = 2.0

    R_cor_min = 0.70
    R_cor_max = 1.25

    min_parameters = np.array([R_min]*(num_params-1))
    min_parameters = np.append(min_parameters,R_cor_min)
    max_parameters = np.array([R_max]*(num_params-1))
    max_parameters = np.append(max_parameters,R_cor_max)

    # Normalization

    min_QoI_HF = np.min(QoI_HF) - 0.01
    max_QoI_HF = np.max(QoI_HF) + 0.01
    min_QoI_LF = np.min(QoI_LF) - 0.01
    max_QoI_LF = np.max(QoI_LF) + 0.01

    parameters_normalized = np.empty(parameters.shape)
    parameters_propagation_normalized = np.empty(parameters_propagation.shape)
    for j in range(parameters.shape[1]):
        parameters_normalized[:,j] = (2*parameters[:,j] - min_parameters[j] - max_parameters[j])/(max_parameters[j] - min_parameters[j])
        parameters_propagation_normalized[:,j] = (2*parameters_propagation[:,j] - min_parameters[j] - max_parameters[j])/(max_parameters[j] - min_parameters[j])
        
    QoI_HF_normalized = (2*QoI_HF - min_QoI_HF - max_QoI_HF)/(max_QoI_HF - min_QoI_HF)
    QoI_LF_normalized = (2*QoI_LF - min_QoI_LF - max_QoI_LF)/(max_QoI_LF - min_QoI_LF)

    # Save

    if not os.path.exists("./data"):
        os.mkdir("./data")

    np.savetxt("data/parameters_normalized"+trial_name_str+".csv", parameters_normalized, delimiter=",")
    np.savetxt("data/parameters_propagation_normalized"+trial_name_str+".csv", parameters_propagation_normalized, delimiter=",")

    np.savetxt("data/QoI_HF_normalized"+trial_name_str+".csv", QoI_HF_normalized, delimiter=",")
    np.savetxt("data/QoI_LF_normalized"+trial_name_str+".csv", QoI_LF_normalized, delimiter=",")

    if num_pilot_samples_to_use != -1:
        np.savetxt("data/use_sample_idxs"+trial_name_str+".csv", use_sample_idxs, fmt='%i', delimiter=",")


def write_unnormalized_data(base_path, config_string, trial_name_str = ""):

    new_parameters_normalized_file = base_path + "/results/new_parameters_LF_normalized_AE"+config_string+trial_name_str+".csv"
    if ("_normalized" in new_parameters_normalized_file):
        new_parameters_file = base_path + "/results/" + new_parameters_normalized_file.split("/")[-1].replace("_normalized", "")
    else:
        raise RuntimeError("No keyword '_normalized' in new_parameters_normalized_file. Cannot create name for output file.")

    new_parameters_propagation_normalized_file = base_path + "/results/new_parameters_propagation_LF_normalized_AE"+config_string+trial_name_str+".csv"
    if ("_normalized" in new_parameters_propagation_normalized_file):
        new_parameters_propagation_file = base_path + "/results/" + new_parameters_propagation_normalized_file.split("/")[-1].replace("_normalized", "")
    else:
        raise RuntimeError("No keyword '_normalized' in new_parameters_normalized_file. Cannot create name for output file.")

    # Read normalized parameters

    new_parameters_LF_normalized_AE = np.genfromtxt(new_parameters_normalized_file, delimiter=',')
    new_parameters_prop_LF_normalized_AE = np.genfromtxt(new_parameters_propagation_normalized_file, delimiter=',')

    # Values parameters

    R_min = 0.5
    R_max = 2.0

    R_cor_min = 0.70
    R_cor_max = 1.25

    num_params = new_parameters_LF_normalized_AE.shape[1]
    min_parameters = np.array([R_min]*(num_params-1))
    min_parameters = np.append(min_parameters,R_cor_min)
    max_parameters = np.array([R_max]*(num_params-1))
    max_parameters = np.append(max_parameters,R_cor_max)

    # Un-normalize parameters values

    new_parameters_LF_AE = np.empty(new_parameters_LF_normalized_AE.shape)
    new_parameters_prop_LF_AE = np.empty(new_parameters_prop_LF_normalized_AE.shape)

    for j in range(new_parameters_LF_AE.shape[1]):
        new_parameters_LF_AE[:,j] = (min_parameters[j] + max_parameters[j] + (max_parameters[j] - min_parameters[j])*new_parameters_LF_normalized_AE[:,j])/2
        new_parameters_prop_LF_AE[:,j] = (min_parameters[j] + max_parameters[j] + (max_parameters[j] - min_parameters[j])*new_parameters_prop_LF_normalized_AE[:,j])/2

    # Save
    np.savetxt(new_parameters_file, new_parameters_LF_AE, delimiter=",")
    np.savetxt(new_parameters_propagation_file, new_parameters_prop_LF_AE, delimiter=",")
