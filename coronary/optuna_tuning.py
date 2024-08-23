from help_functions import find_surrogate_reduced_model, find_surrogate_reduced_correlated_models_invCDF
from help_functions import find_normalizing_flow, find_normalizing_flow_1D, find_normalizing_flow_spline
import optuna
from sklearn.model_selection import train_test_split
import os

# Tuning

def tune_hyperparameter(f_data, g_data, f_output, g_output, dim_reduced, activation, flow_type, layers_max, neurons_max, lr_min, lr_max, gamma_min, gamma_max, epochs, k_splits, max_evals, name, together, sequential, absolute_value, tune_alpha = False, show=True):

    k = 1/k_splits
    hyperparameters = {}

    if not os.path.exists("./optuna"):
        os.mkdir("./optuna")

    f_data_train, f_data_test = train_test_split(f_data, test_size=k, shuffle=False)
    g_data_train, g_data_test = train_test_split(g_data, test_size=k, shuffle=False)
    f_output_train, f_output_test = train_test_split(f_output, test_size=k, shuffle=False)
    g_output_train, g_output_test = train_test_split(g_output, test_size=k, shuffle=False)
    
    #%% Surrogate model & Autoencoder
    
    print("Finding best surrogate and reduced models ...")

    def objective(params):
        layers_surrogate = params.suggest_int('layers_surrogate', 1, layers_max)
        neurons_surrogate = params.suggest_int('neurons_surrogate', 1, neurons_max)
        layers_AE = params.suggest_int('layers_AE', 1, layers_max)
        neurons_AE = params.suggest_int('neurons_AE', 1, neurons_max)
        lr = params.suggest_float('lr', lr_min, lr_max)
        gamma = params.suggest_float('gamma', gamma_min, gamma_max)
        if together:
            if tune_alpha:
                alpha = params.suggest_float('alpha', 0.5, 2.0)
            else:
                alpha = 1.0
            f_surrogate, model_f, g_surrogate, model_g, loss = find_surrogate_reduced_correlated_models_invCDF(f_data_train, g_data_train, f_output_train, g_output_train, 'together', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, alpha, epochs, sequential, absolute_value, show=False, X_f_test=f_data_test, X_g_test=g_data_test, Y_f_test=f_output_test, Y_g_test=g_output_test)
        else:
            f_surrogate, model_f, loss_f = find_surrogate_reduced_model(f_data_train, f_output_train, 'f', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, absolute_value, show=False, X_test=f_data_test, Y_test=f_output_test)
            g_surrogate, model_g, loss_g = find_surrogate_reduced_model(g_data_train, g_output_train, 'g', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, absolute_value, show=False, X_test=g_data_test, Y_test=g_output_test)
            loss = loss_f + loss_g
        return loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=max_evals)

    df = study.trials_dataframe()
    df = df.drop(['datetime_start', 'datetime_complete'], axis=1)
    if together:
        if tune_alpha:
            df.columns = ['trial', 'loss', 'time', 'alpha', 'gamma', 'layers_AE', 'layers_surrogate', 'lr', 'neurons_AE', 'neurons_surrogate', 'state']
        else:
            df.columns = ['trial', 'loss', 'time', 'gamma', 'layers_AE', 'layers_surrogate', 'lr', 'neurons_AE', 'neurons_surrogate', 'state']
    else:
        df.columns = ['trial', 'loss', 'time', 'gamma', 'layers_AE', 'layers_surrogate', 'lr', 'neurons_AE', 'neurons_surrogate', 'state']
    df['time'] = df['time'].dt.total_seconds()
    df.to_csv('./optuna/' + name + '_SURROGATE_AE.csv', index=False)

    layers_surrogate_best = study.best_params['layers_surrogate']
    neurons_surrogate_best = study.best_params['neurons_surrogate']
    layers_AE_best = study.best_params['layers_AE']
    neurons_AE_best = study.best_params['neurons_AE']
    lr_best = study.best_params['lr']
    gamma_best = study.best_params['gamma']

    hyperparameters['layers_surrogate'] = layers_surrogate_best
    hyperparameters['neurons_surrogate'] = neurons_surrogate_best
    hyperparameters['layers_AE'] = layers_AE_best
    hyperparameters['neurons_AE'] = neurons_AE_best
    hyperparameters['lr'] = lr_best
    hyperparameters['gamma'] = gamma_best
    
    if together:
        if tune_alpha:
            alpha_best = study.best_params['alpha']
        else:
            alpha_best = 1.0
        hyperparameters['alpha'] = alpha_best
        f_surrogate, model_f, g_surrogate, model_g, _ = find_surrogate_reduced_correlated_models_invCDF(f_data_train, g_data_train, f_output_train, g_output_train, 'together', dim_reduced, activation, layers_surrogate_best, neurons_surrogate_best, layers_AE_best, neurons_AE_best, lr_best, gamma_best, alpha_best, epochs, sequential, absolute_value, show=False)
    
    else:
        f_surrogate, model_f, loss_f = find_surrogate_reduced_model(f_data_train, f_output_train, 'f', dim_reduced, activation, layers_surrogate_best, neurons_surrogate_best, layers_AE_best, neurons_AE_best, lr_best, gamma_best, epochs, absolute_value, show=False)
        g_surrogate, model_g, loss_g = find_surrogate_reduced_model(g_data_train, g_output_train, 'g', dim_reduced, activation, layers_surrogate_best, neurons_surrogate_best, layers_AE_best, neurons_AE_best, lr_best, gamma_best, epochs, absolute_value, show=False)

    #%% Normalizing flow for AE
    
    if flow_type != 'invCDF':

        print("Finding best normalizing flows for AE ...")
    
        data_reduced_f = model_f.encode(f_data).detach()
        data_reduced_g = model_g.encode(g_data).detach()
    
        data_reduced_f_train, data_reduced_f_test = train_test_split(data_reduced_f, test_size=1/k_splits, shuffle=False)
        data_reduced_g_train, data_reduced_g_test = train_test_split(data_reduced_g, test_size=1/k_splits, shuffle=False)
    
        def objective(params):
            if flow_type == 'RealNVP':
                layers = params.suggest_int('layers', 1, layers_max)
                neurons = params.suggest_int('neurons', 1, neurons_max)
            lr = params.suggest_float('lr', lr_min, lr_max)
            gamma = params.suggest_float('gamma', gamma_min, gamma_max)
            if flow_type == 'spline':
                T_f, T_inverse_f, loss_f = find_normalizing_flow_spline(data_reduced_f_train, epochs, 'f', lr, gamma, show=False, data_test=data_reduced_f_test)
                T_g, T_inverse_g, loss_g = find_normalizing_flow_spline(data_reduced_g_train, epochs, 'g', lr, gamma, show=False, data_test=data_reduced_g_test)
            elif flow_type == 'RealNVP':
                if dim_reduced == 1:
                    T_f, T_inverse_f, loss_f = find_normalizing_flow_1D(data_reduced_f_train, layers, neurons, epochs, 'f', lr, gamma, show=False, data_test=data_reduced_f_test)
                    T_g, T_inverse_g, loss_g = find_normalizing_flow_1D(data_reduced_g_train, layers, neurons, epochs, 'g', lr, gamma, show=False, data_test=data_reduced_g_test)
                else:
                    T_f, T_inverse_f, loss_f = find_normalizing_flow(data_reduced_f_train, layers, neurons, epochs, 'f', lr, gamma, show=False, data_test=data_reduced_f_test)
                    T_g, T_inverse_g, loss_g = find_normalizing_flow(data_reduced_g_train, layers, neurons, epochs, 'g', lr, gamma, show=False, data_test=data_reduced_g_test)
            loss = loss_f + loss_g
            return loss
    
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=max_evals)
    
        df = study.trials_dataframe()
        df = df.drop(['datetime_start', 'datetime_complete'], axis=1)
        if flow_type == 'spline':
            df.columns = ['trial', 'loss', 'time', 'gamma', 'lr', 'state']
        elif flow_type == 'RealNVP':
            df.columns = ['trial', 'loss', 'time', 'gamma', 'layers', 'lr', 'neurons', 'state']
        df['time'] = df['time'].dt.total_seconds()
        df.to_csv('./optuna/' + name + '_NF_AE.csv', index=False)
    
        if flow_type == 'RealNVP':
            neurons_NF_AE_best = study.best_params['neurons']
            layers_NF_AE_best = study.best_params['layers']
        lr_best = study.best_params['lr']
        gamma_best = study.best_params['gamma']
    
        if flow_type == 'RealNVP':
            hyperparameters['neurons_NF_AE'] = neurons_NF_AE_best
            hyperparameters['layers_NF_AE'] = layers_NF_AE_best
        hyperparameters['lr_NF_AE'] = lr_best
        hyperparameters['gamma_NF_AE'] = gamma_best

    if show:
        print("HYPERPARAMETERS:")
        for key, value in hyperparameters.items():
            print(key + ': ' + str(value))

    return hyperparameters
