import numpy as np
import os
import json
from help_functions import read_simulation_data
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob

# File paths
base_path = "./"
base_path = os.path.abspath(base_path)

parameters_file = base_path + "/simulations/all_param_values.json"
all_0d_data_file = base_path + "/simulations/all_0d_data.json"
all_3d_data_file = base_path + "/simulations/all_3d_data.json"
#new_0d_data_file = base_path + "/simulations/all_0d_data_AE.json"
#all_0d_data_file_prop = base_path + "/simulations/all_0d_data_propagation.json"
#new_0d_data_file_prop = base_path + "/simulations/all_0d_data_AE_propagation.json"

# QoI details
QoI_LF_name = 'mean_flow:lca1:BC_lca1'
QoI_HF_name = 'max_osi_sten_lad'
cost_ratio = 10/(7*96*60*60 + 1*60*60) # LF:10 sec ; HF:7 hr on 96 procs + 1 hr on 1 proc 

# Options
plot = False
save = False

# Read data

#samples, parameters, QoI_LF, QoI_HF, _, QoI_LF_AE, QoI_LF_prop, QoI_LF_prop_AE = read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_file, all_0d_data_file, all_3d_data_file, None, new_0d_data_file, all_0d_data_file_prop, new_0d_data_file_prop)

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
    all_0d_data_file_prop = base_path + "/simulations/all_0d_data_propagation.json"
    new_0d_data_file_prop = base_path + "/simulations/all_0d_data_AE_propagation.json"
#   samples, parameters, QoI_LF, QoI_HF, _, new_QoI_LF_AE, _, _ = read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_file, all_0d_data_file, all_3d_data_file, None, new_0d_data_file)
    
    samples, parameters, QoI_LF, QoI_HF, _, QoI_LF_AE, QoI_LF_prop, QoI_LF_prop_AE = read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_file, all_0d_data_file, all_3d_data_file, None, new_0d_data_file, all_0d_data_file_prop, new_0d_data_file_prop)

    num_samples = len(samples)

    # If not using all samples
    use_sample_idx_file = base_path + "/data/sample_idxs"+trial_idx_str+".csv"
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

        if plot:

            plt.figure()
            mu = MC_mean
            sigma = MC_std
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), label="MC")
            mu = MFMC_mean
            sigma = MFMC_std
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), label="MFMC")
            mu = MFMC_mean
            sigma = MFMC_std_optimal
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), ls='--', label="MFMC optimal")
            mu = MFMC_AE_mean
            sigma = MFMC_AE_std
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), label="MFMC-AE")
            mu = MFMC_AE_mean
            sigma = MFMC_AE_std_optimal
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), ls='--', label="MFMC-AE optimal")
            plt.xlabel(QoI_HF_name)
            plt.legend()
            plt.savefig("results/estimators.pdf")

            fig, ax = plt.subplots()
            x_pos = [1,2,3]
            x_labels = ["MC", "MFMC", "MFMC-AE"]
            means = np.array([MC_mean, MFMC_mean, MFMC_AE_mean])
            std = np.array([MC_std, MFMC_std, MFMC_AE_std])
            confidence = 0.99
            k = np.sqrt(1.0/(1.0-confidence)) # Chebyshev's inequality: P(|X-\mu| >= k\sigma) = 1/k^2
            k_99 = k
            eb1 = ax.errorbar(x_pos, means, k*std, fmt='ks', markersize=7.0, capsize=5.0, ecolor='k', elinewidth=1.0)
            eb1[-1][0].set_linestyle(':')
            confidence = 0.95
            k = np.sqrt(1.0/(1.0-confidence)) # Chebyshev's inequality: P(|X-\mu| >= k\sigma) = 1/k^2
            k_95 = k
            ax.errorbar(x_pos, means, k*std, fmt='ks', markersize=7.0, capsize=5.0, ecolor='k', elinewidth=2.0)
            ax.set_xlim(x_pos[0]-1, x_pos[-1]+1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.set_ylabel(QoI_HF_name)
            plt.savefig("results/estimators_errorbar.pdf")

            # Number of additional simulations to reduce variance
            #std_reduction_range = np.linspace(0.01, 0.8, 10)
            std_reduction_range = np.linspace(MFMC_AE_std_optimal/MC_std, 0.9, 10)
            num_hf_mc = np.zeros_like(std_reduction_range)
            num_hf_mfmc = np.zeros_like(std_reduction_range)
            num_hf_mfmc_ae = np.zeros_like(std_reduction_range)
            std_mfmc_fixed_HF_budget = np.zeros_like(std_reduction_range)
            std_mfmc_fixed_HF_budget_ae = np.zeros_like(std_reduction_range)
            std_mfmc_fixed_budget = np.zeros_like(std_reduction_range)
            std_mfmc_fixed_budget_ae = np.zeros_like(std_reduction_range)
            std_hf = np.std(QoI_HF[:N_HF])

            # Find number of LF simulations for given standard deviation with following data: 
            # number of HF samples (num_HF), correlation (rho), required Std. (std_reqd), Std. of HF samples (std_HF)
            def find_num_LF_MFMC(num_HF, rho, std_reqd, std_HF):
                std_MC = std_HF/np.sqrt(num_HF) # Monte Carlo SD
                num_LF_MFMC = num_HF*(1.0-(1/rho**2)*(1.0-(std_reqd/std_MC)**2))**(-1)
                return num_LF_MFMC

            for i, std_reduc in enumerate(std_reduction_range):
                # Required standard deviation
                std_new = std_reduc*MC_std

                # Required number of HF sims for MC
                num_hf_mc[i] = (std_hf/std_new)**2 
             
                num_additional_HF = -1 # Add HF sims if target std. cannot be reached with only new LF sims
                num_lf_mfmc = -1 # Negative value indicates target std. not reached
                while (num_lf_mfmc < 0):
                    num_additional_HF = num_additional_HF + 1
                    num_lf_mfmc = find_num_LF_MFMC(N_HF+num_additional_HF, rho, std_new, std_hf)
                #print(std_reduc, num_additional_HF, num_lf_mfmc, num_lf_mfmc*cost_ratio)
                # Equivalent cost in HF sims
                num_hf_mfmc[i] = N_HF + num_additional_HF + num_lf_mfmc*cost_ratio 
                
                num_additional_HF = -1 # Add HF sims if target std. cannot be reached with only new LF sims
                num_lf_mfmc_ae = -1 # Negative value indicates target std. not reached
                while (num_lf_mfmc_ae < 0):
                    num_additional_HF = num_additional_HF + 1
                    num_lf_mfmc_ae = find_num_LF_MFMC(N_HF+num_additional_HF, rho_AE, std_new, std_hf)
                #print(std_reduc, num_additional_HF, num_lf_mfmc_ae, num_lf_mfmc_ae*cost_ratio)
                # Equivalent cost in HF sims
                num_hf_mfmc_ae[i] = N_HF + num_additional_HF + num_lf_mfmc_ae*cost_ratio 
                
                # Convert MC cost into additional LF simulations beyond N_HF pilot
                # Fixed HF budget (plus additional LF simulations)
                num_propagation_LF = (num_hf_mc[i] - N_HF)/cost_ratio
                # Compute std. of estimator with N_HF + num_propagation_LF sims
                std_mfmc_fixed_HF_budget[i] = MC_std*np.sqrt(1-((num_propagation_LF - N_HF)/num_propagation_LF)*rho**2)
                std_mfmc_fixed_HF_budget_ae[i] = MC_std*np.sqrt(1-((num_propagation_LF - N_HF)/num_propagation_LF)*rho_AE**2)
                
                # Use MC cost in optimal allocation for MFMC and MFMC-AE
                num_HF, num_LF, budget_used = find_optimal_allocation(num_hf_mc[i], rho)
                std_mfmc_fixed_budget[i] = (std_hf/np.sqrt(num_HF))*np.sqrt(1-((num_LF-num_HF)/num_LF)*rho**2)
                num_HF, num_LF, budget_used = find_optimal_allocation(num_hf_mc[i], rho_AE)
                std_mfmc_fixed_budget_ae[i] = (std_hf/np.sqrt(num_HF))*np.sqrt(1-((num_LF-num_HF)/num_LF)*rho_AE**2)

            plt.figure()
            plt.plot(num_hf_mc, (std_reduction_range*MC_std)**2, '-o', label="MC")
            #plt.plot(num_hf_mc, (std_mfmc_fixed_HF_budget)**2, '-o', label="MFMC")
            plt.plot(num_hf_mc, (std_mfmc_fixed_budget)**2, '-o', label="MFMC")
            #plt.plot(num_hf_mc, (std_mfmc_fixed_HF_budget_ae)**2, '-o', label="MFMC-AE")
            plt.plot(num_hf_mc, (std_mfmc_fixed_budget_ae)**2, '-o', label="MFMC-AE")
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel("Estimator variance")
            plt.xlabel("HF-equivalent simulations")
            plt.legend()
            plt.savefig("results/std_fixed_budget.pdf")

            plt.figure()
            plt.plot(num_hf_mc - N_HF, (std_reduction_range)**2, '-o', label="MC")
            plt.plot(num_hf_mfmc - N_HF, (std_reduction_range)**2, '-o', label="MFMC")
            plt.plot(num_hf_mfmc_ae - N_HF, (std_reduction_range)**2, '-o', label="MFMC-AE")
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel("Variance reduction factor")
            plt.xlabel("Additional HF-equivalent simulations")
            plt.legend()
            plt.savefig("results/budget_fixed_std.pdf")

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
            np.savez("results/propagation.npz", means=means, std=std, k_99=k_99, k_95=k_95)
            np.savez("results/cost_analysis.npz", std_reduction_range=std_reduction_range, MC_std=MC_std, 
                    std_mfmc_fixed_budget=std_mfmc_fixed_budget, std_mfmc_fixed_budget_ae=std_mfmc_fixed_budget_ae,
                    N_HF=N_HF, num_hf_mc=num_hf_mc, num_hf_mfmc=num_hf_mfmc, num_hf_mfmc_ae=num_hf_mfmc_ae)

    else:
        final_correlation[trial_idx] = rho_AE

if repeated_trials > 1:
    np.savetxt(trial_files[0], np.column_stack((trials_data,final_correlation,MC_mean)), header = trials_header+', final_correlation (pilot), MC mean')
