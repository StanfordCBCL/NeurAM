import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Autoencoder and surrogate model

def compute_reduction(d, data, f_data, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs, show=True, X_test=None, Y_test=None):

    class Autoencoder(torch.nn.Module):
        
        def __init__(self):
            super().__init__()
            
            model_structure = []
            for i in range(layers_AE-1):
                model_structure += [torch.nn.Linear(neurons_AE, neurons_AE), activation]
    
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(d, neurons_AE),
                activation,
                *model_structure,
                torch.nn.Linear(neurons_AE, r),
            )
    
            self.decoder_ = torch.nn.Sequential(
                torch.nn.Linear(r, neurons_AE),
                activation,
                *model_structure,
                torch.nn.Linear(neurons_AE, d),
                torch.nn.Sigmoid()
            )
          
        def decoder(self, t):
            return 2*self.decoder_(t) - 1
          
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
    class Surrogate(torch.nn.Module):
        
        def __init__(self):
            super().__init__()
            
            model_structure = [torch.nn.Linear(1, neurons_surrogate), activation]
            for i in range(layers_surrogate-1):
                model_structure += [torch.nn.Linear(neurons_surrogate, neurons_surrogate), activation]
            model_structure += [torch.nn.Linear(neurons_surrogate, 1)]
            self.net = torch.nn.Sequential(*model_structure)
          
        def forward(self, x):
            output = self.net(x)
            return output
    
    autoencoder = Autoencoder()
    surrogate = Surrogate()
    
    optimizer = torch.optim.Adam(list(autoencoder.parameters()) + list(surrogate.parameters()))
    loss_function = torch.nn.MSELoss()
    losses = [[], []]
    
    for epoch in tqdm(range(epochs), disable = not show):  
        reconstructed = autoencoder(data)   
        loss =   loss_function(f_data, torch.squeeze(surrogate(autoencoder.encoder(reconstructed)))) \
               + loss_function(f_data, torch.squeeze(surrogate(autoencoder.encoder(data)))) \
               + loss_function(reconstructed, autoencoder(reconstructed))
        losses[0].append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if X_test is not None:
            reconstructed_test = autoencoder(X_test)
            loss_output =  (loss_function(Y_test, torch.squeeze(surrogate(autoencoder.encoder(reconstructed_test)))) \
                          + loss_function(Y_test, torch.squeeze(surrogate(autoencoder.encoder(X_test)))) \
                          + loss_function(reconstructed_test, autoencoder(reconstructed_test))).item()
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)
        
    if show:
        plt.figure()
        plt.semilogy(losses[0], label='Train')
        if X_test is not None:
            plt.semilogy(losses[1], label='Test')
        plt.legend()
        plt.show(block=False)
        plt.close()
    
    return autoencoder, surrogate, losses[1][-1]

# Hyperparameter tuning

def tune_hyperparameter(d, data, f_data, r, activation, layers_max, neurons_max, epochs, k_splits, max_evals, name):
    
    k = 1/k_splits
    hyperparameters = {}

    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/" + name):
        os.mkdir("./results/" + name)

    data_train, data_test = train_test_split(data, test_size=k, shuffle=False)
    f_data_train, f_data_test = train_test_split(f_data, test_size=k, shuffle=False)
    
    def objective(params):
        layers_AE = params.suggest_int('layers_AE', 1, layers_max)
        neurons_AE = params.suggest_int('neurons_AE', 1, neurons_max)
        layers_surrogate = params.suggest_int('layers_surrogate', 1, layers_max)
        neurons_surrogate = params.suggest_int('neurons_surrogate', 1, neurons_max)
        autoencoder, surrogate, loss = compute_reduction(d, data_train, f_data_train, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs, show=False, X_test=data_test, Y_test=f_data_test)
        return loss
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=max_evals)
    
    df = study.trials_dataframe()
    df = df.drop(['datetime_start', 'datetime_complete'], axis=1)
    df.columns = ['trial', 'loss', 'time', 'layers_AE', 'layers_surrogate', 'neurons_AE', 'neurons_surrogate', 'state']
    df['time'] = df['time'].dt.total_seconds()
    df.to_csv('./results/' + name + '/tuning.csv', index=False)

    hyperparameters['layers_AE'] = study.best_params['layers_AE']
    hyperparameters['neurons_AE'] = study.best_params['neurons_AE']
    hyperparameters['layers_surrogate'] = study.best_params['layers_surrogate']
    hyperparameters['neurons_surrogate'] = study.best_params['neurons_surrogate']
    
    print("HYPERPARAMETERS:")
    for key, value in hyperparameters.items():
        print(key + ': ' + str(value))

    return hyperparameters

# Multifidelity Monte Carlo

def find_CDF(data):

    Finv = lambda a: torch.sort(data, dim=0)[0][(data.shape[0]*a).type(torch.int)]
    F = lambda x: torch.sum((data <= x[:,0]).type(torch.int), dim=0)/(data.shape[0] + 1)

    return F, Finv

def compute_MC(f, d, B):
    
    samples = 2*torch.rand((B, d)) - 1
    MC = torch.mean(f(samples))

    return MC.item()

def compute_MFMC(f_HF, f_LF, d, B, w):
    
    pilot_samples = 2*torch.rand((B, d)) - 1
    cov_matrix = np.cov(f_HF(pilot_samples), f_LF(pilot_samples))
    rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])
    beta = cov_matrix[0,1]/cov_matrix[1,1]
    
    gamma = np.sqrt((rho**2)/(w*(1 - rho**2)))
    N_HF = int(round(B/(1 + w*gamma)))
    N_LF = int(round(gamma*N_HF))
    
    samples = 2*torch.rand((N_LF, d)) - 1
    MFMC = torch.nanmean(f_HF(samples[:N_HF])) - beta*(torch.nanmean(f_LF(samples[:N_HF])) - torch.nanmean(f_LF(samples[:N_LF])))
    
    return MFMC.item(), rho
