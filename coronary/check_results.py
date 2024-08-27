import numpy as np
import matplotlib.pyplot as plt
import os
import json
from help_functions import read_simulation_data
import glob

# QoI details
QoI_LF_name = 'mean_flow:lca1:BC_lca1'
QoI_HF_name = 'max_osi_sten_lad'

# File paths
base_path = "./"
base_path = os.path.abspath(base_path)

parameters_file = base_path + "/simulations/all_param_values.json"
all_0d_data_file = base_path + "/simulations/all_0d_data.json"
all_3d_data_file = base_path + "/simulations/all_3d_data.json"

# Options
plot = True

# Check for repeated trials
repeated_trials = 1
trial_files = glob.glob("trials_*.log")
if len(trial_files) == 1:
    trials_header = ' '.join(list(np.genfromtxt(trial_files[0], comments=None, max_rows=1, dtype='U')))
    trials_data = np.genfromtxt(trial_files[0], comments=None, skip_header=1)
    repeated_trials = trials_data.shape[0]
    final_correlation = np.zeros(repeated_trials)
    MC_mean = np.zeros(repeated_trials)
    plot = False
elif len(trial_files) > 1:
    raise RuntimeError("ERROR: len(glob.glob('trials_*.log')) > 1")

for trial_idx in range(repeated_trials):

    if (repeated_trials > 1):
        print("\n Trial: "  +str(trial_idx))
        trial_idx_str = "_"+str(trial_idx).zfill(4)
    else:
        trial_idx_str = ""
    
    # Read data
    new_0d_data_file = base_path + "/simulations/all_0d_data_AE"+trial_idx_str+".json"
    samples, parameters, QoI_LF, QoI_HF, _, new_QoI_LF_AE, _, _ = read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_file, all_0d_data_file, all_3d_data_file, None, new_0d_data_file)
    num_samples = len(samples)

    # If not using all samples
    use_sample_idx_file = base_path + "/data/use_sample_idxs"+trial_idx_str+".csv"
    if os.path.isfile(use_sample_idx_file):
        use_sample_idxs = np.genfromtxt(use_sample_idx_file, dtype=int)
        QoI_LF = QoI_LF[use_sample_idxs]
        QoI_HF = QoI_HF[use_sample_idxs]
        parameters = parameters[use_sample_idxs,:]
        num_samples = len(use_sample_idxs)

    # Compute initial correlation
    cov_matrix = np.cov(QoI_HF, QoI_LF)
    rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])

    print("Original correlation coefficient: " + str(rho))

    # Compute final correlation
    cov_matrix_AE = np.cov(QoI_HF, new_QoI_LF_AE)
    rho_AE = cov_matrix_AE[0,1]/np.sqrt(cov_matrix_AE[0,0]*cov_matrix_AE[1,1])

    print("New correlation coefficient (AE): " + str(rho_AE))

    if (repeated_trials > 1):
        MC_mean[trial_idx] = np.mean(QoI_HF)

    if plot:
        # Initial correlation
        plt.figure()
        plt.scatter(QoI_HF, QoI_LF)
        plt.title(r'$\rho = ' + str(rho) + r'$')
        plt.show()
        # Final correlation
        plt.figure()
        plt.scatter(QoI_HF, new_QoI_LF_AE)
        plt.title(r'$\rho_{\mathrm{AE}} = ' + str(rho_AE) + r'$')
        plt.show()

    if (repeated_trials == 1):
        # Optimal allocation original data
        budget = num_samples
        cost_ratio = 10/(7*96*60*60 + 1*60*60) # LF:10 sec ; HF:7 hr on 96 procs + 1 hr on 1 proc

        coeff = np.sqrt((rho**2)/(cost_ratio*(1 - rho**2)))
        N_HF = round(budget/(1 + cost_ratio*coeff))
        N_LF = round((coeff*budget)/(1 + cost_ratio*coeff))

        print("Optimal number HF samples: " + str(N_HF))
        print("Optimal number LF samples: " + str(N_LF))


        # Optimal allocation new data AE
        coeff_AE = np.sqrt((rho_AE**2)/(cost_ratio*(1 - rho_AE**2)))
        N_HF_AE = round(budget/(1 + cost_ratio*coeff_AE))
        N_LF_AE = round((coeff_AE*budget)/(1 + cost_ratio*coeff_AE))

        print("Optimal number HF samples: " + str(N_HF_AE))
        print("Optimal number LF samples: " + str(N_LF_AE))
    
    else:
        final_correlation[trial_idx] = rho_AE

if repeated_trials > 1:
    np.savetxt(trial_files[0], np.column_stack((trials_data,final_correlation,MC_mean)), header = trials_header+', final_correlation (pilot), MC mean')


