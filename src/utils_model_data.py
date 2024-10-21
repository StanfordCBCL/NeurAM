import torch
import torchmetrics
from tqdm import tqdm
import normflows as nf
import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import numpy as np
import json
import optuna
from sklearn.model_selection import train_test_split
import os

# Surrogate model & autoencoder

class Autoencoder(torch.nn.Module):

    def __init__(self, layers_AE, neurons_AE, dim, dim_reduced, activation):
        super().__init__()

        model_structure = []
        for i in range(layers_AE-1):
            model_structure += [torch.nn.Linear(neurons_AE, neurons_AE), activation]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim, neurons_AE),
            activation,
            *model_structure,
            torch.nn.Linear(neurons_AE, dim_reduced)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(dim_reduced, neurons_AE),
            activation,
            *model_structure,
            torch.nn.Linear(neurons_AE, dim),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z):
        decoded = 2*self.decoder(z) - 1
        return decoded

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

class Surrogate_NN(torch.nn.Module):

    def __init__(self, layers_surrogate, neurons_surrogate, dim_reduced, activation):
        super().__init__()

        model_structure = [torch.nn.Linear(dim_reduced, neurons_surrogate), activation]
        for i in range(layers_surrogate-1):
            model_structure += [torch.nn.Linear(neurons_surrogate, neurons_surrogate), activation]
        model_structure += [torch.nn.Linear(neurons_surrogate, 1)]
        self.net = torch.nn.Sequential(*model_structure)

    def forward(self, x):
        output = self.net(x)
        return output

# Functions to train surrogate model & autoencoder

def find_surrogate_reduced_model(X, Y, name, dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, show=True, X_test=None, Y_test=None, return_surrogate_NN=False):

    _, dim = X.shape

    model = Autoencoder(layers_AE, neurons_AE, dim, dim_reduced, activation)
    surrogate = Surrogate_NN(layers_surrogate, neurons_surrogate, dim_reduced, activation)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(surrogate.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    loss_function = torch.nn.MSELoss()
    losses = [[], []]

    for epoch in tqdm(range(epochs), disable = not show):

        reconstructed = model(X)
        loss =   loss_function(Y, torch.squeeze(surrogate(model.encoder(reconstructed)))) \
               + loss_function(Y, torch.squeeze(surrogate(model.encoder(X)))) \
               + loss_function(torch.squeeze(surrogate(model.encoder(reconstructed))), torch.squeeze(surrogate(model.encoder(X)))) \
               + loss_function(model(reconstructed), reconstructed)    
        losses[0].append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
       
        if X_test is not None:
            reconstructed_test = model(X_test)
            loss_output =  (loss_function(Y_test, torch.squeeze(surrogate(model.encoder(reconstructed_test)))) \
                          + loss_function(Y_test, torch.squeeze(surrogate(model.encoder(X_test)))) \
                          + loss_function(torch.squeeze(surrogate(model.encoder(reconstructed_test))), torch.squeeze(surrogate(model.encoder(X_test)))) \
                          + loss_function(model(reconstructed_test), reconstructed_test)).item()    
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)

   #if show:
   #    plt.figure()
   #    plt.semilogy(losses[0], label='Train')
   #    if X_test is not None:
   #        plt.semilogy(losses[1], label='Test')
   #    plt.title("Loss surrogate + autoencoder " + name)
   #    plt.legend()
   #    plt.show(block=False)
   #    #plt.show()
   #    plt.close()

    def f_surrogate(x):
        model.eval()
        surrogate.eval()
        y = torch.squeeze(surrogate(model.encoder(x))).detach()
        return y
    
    save_data = {"autoencoder" : model.state_dict(),
            "surrogate" : surrogate.state_dict()}
    
    if return_surrogate_NN:
        return f_surrogate, model, losses, surrogate
    else:
        return f_surrogate, model, losses, save_data

def find_surrogate_reduced_correlated_models_invCDF(X_f, X_g, Y_f, Y_g, name, dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, alpha, epochs, sequential, show=True, X_f_test=None, X_g_test=None, Y_f_test=None, Y_g_test=None):

    if sequential:
        _, model_f, losses_f, surrogate_f = find_surrogate_reduced_model(X_f, Y_f, 'together_f', dim_reduced, activation, 
                layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, show, 
                X_test=X_f_test, Y_test=Y_f_test, return_surrogate_NN=True)  
        _, model_g, losses_g, surrogate_g = find_surrogate_reduced_model(X_g, Y_g, 'together_g', dim_reduced, activation, 
                layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, show, 
                X_test=X_g_test, Y_test=Y_g_test, return_surrogate_NN=True)  
        losses = [[], [], losses_f, losses_g]

    else:
        _, dim_f = X_f.shape
        _, dim_g = X_g.shape

        model_f = Autoencoder(layers_AE, neurons_AE, dim_f, dim_reduced, activation)
        model_g = Autoencoder(layers_AE, neurons_AE, dim_g, dim_reduced, activation)
        surrogate_f = Surrogate_NN(layers_surrogate, neurons_surrogate, dim_reduced, activation)
        surrogate_g = Surrogate_NN(layers_surrogate, neurons_surrogate, dim_reduced, activation)
        losses = [[], []]
    
    optimizer = torch.optim.Adam(list(model_f.parameters()) + list(model_g.parameters()) + list(surrogate_f.parameters()) + list(surrogate_g.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    loss_function = torch.nn.MSELoss()
    loss_pearson = torchmetrics.PearsonCorrCoef()

    for epoch in tqdm(range(epochs), disable = not show):

        reconstructed_f = model_f(X_f)
        reconstructed_g = model_g(X_g)
        
        data_reduced_f = model_f.encode(X_f)
        data_reduced_g = model_g.encode(X_g)
        
        CDF_f = lambda x: torch.sum((data_reduced_f <= x[:,0]).type(torch.int), dim=0)/(data_reduced_f.shape[0] + 1)
        T_inverse_f = lambda x: torch.unsqueeze(np.sqrt(2)*torch.erfinv(2*CDF_f(x) - 1), 1)
        
        invCDF_g = lambda a: torch.sort(data_reduced_g, dim=0)[0][(data_reduced_g.shape[0]*a).type(torch.int)]
        T_g = lambda z: invCDF_g((torch.erf(z[:,0]/np.sqrt(2)) + 1)/2)
        
        loss =   loss_function(Y_f, torch.squeeze(surrogate_f(model_f.encoder(reconstructed_f)))) \
               + loss_function(Y_f, torch.squeeze(surrogate_f(model_f.encoder(X_f)))) \
               + loss_function(torch.squeeze(surrogate_f(model_f.encoder(reconstructed_f))), torch.squeeze(surrogate_f(model_f.encoder(X_f)))) \
               + loss_function(model_f(reconstructed_f), reconstructed_f) \
               + loss_function(Y_g, torch.squeeze(surrogate_g(model_g.encoder(reconstructed_g)))) \
               + loss_function(Y_g, torch.squeeze(surrogate_g(model_g.encoder(X_g)))) \
               + loss_function(torch.squeeze(surrogate_g(model_g.encoder(reconstructed_g))), torch.squeeze(surrogate_g(model_g.encoder(X_g)))) \
               + loss_function(model_g(reconstructed_g), reconstructed_g) \
               - alpha*abs(loss_pearson(Y_f, torch.squeeze(surrogate_g(model_g.encoder(model_g.decoder(T_g(T_inverse_f(data_reduced_f))))))))
        
        losses[0].append(loss.item() + alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if X_f_test is not None and X_g_test is not None:
            
            reconstructed_f_test = model_f(X_f_test)
            reconstructed_g_test = model_g(X_g_test)
            
            data_reduced_f_test = model_f.encode(X_f_test)
            data_reduced_g_test = model_g.encode(X_g_test)
            
            CDF_f_test = lambda x: torch.sum((data_reduced_f_test <= x[:,0]).type(torch.int), dim=0)/(data_reduced_f_test.shape[0] + 1)
            T_inverse_f_test = lambda x: torch.unsqueeze(np.sqrt(2)*torch.erfinv(2*CDF_f_test(x) - 1), 1)
            
            invCDF_g_test = lambda a: torch.sort(data_reduced_g_test, dim=0)[0][(data_reduced_g_test.shape[0]*a).type(torch.int)]
            T_g_test = lambda z: invCDF_g_test((torch.erf(z[:,0]/np.sqrt(2)) + 1)/2)
            
            loss_output =   (loss_function(Y_f_test, torch.squeeze(surrogate_f(model_f.encoder(reconstructed_f_test)))) \
                          + loss_function(Y_f_test, torch.squeeze(surrogate_f(model_f.encoder(X_f_test)))) \
                          + loss_function(torch.squeeze(surrogate_f(model_f.encoder(reconstructed_f_test))), torch.squeeze(surrogate_f(model_f.encoder(X_f_test)))) \
                          + loss_function(model_f(reconstructed_f_test), reconstructed_f_test) \
                          + loss_function(Y_g_test, torch.squeeze(surrogate_g(model_g.encoder(reconstructed_g_test)))) \
                          + loss_function(Y_g_test, torch.squeeze(surrogate_g(model_g.encoder(X_g_test)))) \
                          + loss_function(torch.squeeze(surrogate_g(model_g.encoder(reconstructed_g_test))), torch.squeeze(surrogate_g(model_g.encoder(X_g_test)))) \
                          + loss_function(model_g(reconstructed_g_test), reconstructed_g_test) \
                          - alpha*abs(loss_pearson(Y_f_test, torch.squeeze(surrogate_g(model_g.encoder(model_g.decoder(T_g_test(T_inverse_f_test(data_reduced_f_test))))))))).item()
            
            loss_output += alpha
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)

#   if show:
#       plt.figure()
#       plt.semilogy([x + alpha for x in losses[0]], label='Train')
#       if X_f_test is not None and X_g_test is not None:
#           plt.semilogy([x + alpha for x in losses[1]], label='Test')
#       plt.title("Loss surrogate + autoencoder " + name)
#       plt.legend()
#       plt.show(block=False)
#       #plt.show()
#       plt.close()

    def f_surrogate(x):
        model_f.eval()
        surrogate_f.eval()
        y = torch.squeeze(surrogate_f(model_f.encoder(x))).detach()
        return y

    def g_surrogate(x):
        model_g.eval()
        surrogate_g.eval()
        y = torch.squeeze(surrogate_g(model_g.encoder(x))).detach()
        return y

    save_data = {"autoencoder_HF" : model_f.state_dict(),
            "autoencoder_LF" : model_g.state_dict(),
            "surrogate_HF" : surrogate_f.state_dict(),
            "surrogate_LF" : surrogate_g.state_dict()}

    return f_surrogate, model_f, g_surrogate, model_g, losses, save_data

# Load autoencoder and surrogate model from saved PyTorch files

def load_surrogate_reduced_correlated_models_invCDF(NN_model_path, dim_f, dim_g, dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE):
   
    # Initialize models
    model_f = Autoencoder(layers_AE, neurons_AE, dim_f, dim_reduced, activation)
    model_g = Autoencoder(layers_AE, neurons_AE, dim_g, dim_reduced, activation)
    surrogate_f = Surrogate_NN(layers_surrogate, neurons_surrogate, dim_reduced, activation)
    surrogate_g = Surrogate_NN(layers_surrogate, neurons_surrogate, dim_reduced, activation)
   
    # Read models
    saved_models = torch.load(NN_model_path+"/NN_models.pt", weights_only=True)
    model_f.load_state_dict(saved_models["autoencoder_HF"])
    model_g.load_state_dict(saved_models["autoencoder_LF"])
    surrogate_f.load_state_dict(saved_models["surrogate_HF"])
    surrogate_g.load_state_dict(saved_models["surrogate_LF"])
    model_f.eval()
    model_g.eval()

    def f_surrogate(x):
        model_f.eval()
        surrogate_f.eval()
        y = torch.squeeze(surrogate_f(model_f.encoder(x))).detach()
        return y

    def g_surrogate(x):
        model_g.eval()
        surrogate_g.eval()
        y = torch.squeeze(surrogate_g(model_g.encoder(x))).detach()
        return y
    
    return f_surrogate, model_f, g_surrogate, model_g

def load_surrogate_reduced_model(NN_model_path, key, dim, dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE):
    
    # Initialize models
    model = Autoencoder(layers_AE, neurons_AE, dim, dim_reduced, activation)
    surrogate = Surrogate_NN(layers_surrogate, neurons_surrogate, dim_reduced, activation)
    
    # Read models
    saved_models = torch.load(NN_model_path+"NN_models_"+key+".pt", weights_only=True)
    model.load_state_dict(saved_models["autoencoder"])
    surrogate.load_state_dict(saved_models["surrogate"])
    model.eval()

    def surrogate(x):
        model.eval()
        surrogate.eval()
        y = torch.squeeze(surrogate(model.encoder(x))).detach()
        return y

    return surrogate, model 

# Hyperparameter tuning

def tune_hyperparameters(base_path, f_data, g_data, f_output, g_output, dim_reduced, activation, flow_type, layers_max, neurons_max, lr_min, lr_max, gamma_min, gamma_max, epochs, k_splits, max_evals, name, together, sequential, tune_alpha = False, show=True):

    k = 1/k_splits
    hyperparameters = {}

    if not os.path.exists(base_path + "/optuna"):
        os.mkdir(base_path + "/optuna")

    f_data_train, f_data_test = train_test_split(f_data, test_size=k, shuffle=False)
    g_data_train, g_data_test = train_test_split(g_data, test_size=k, shuffle=False)
    f_output_train, f_output_test = train_test_split(f_output, test_size=k, shuffle=False)
    g_output_train, g_output_test = train_test_split(g_output, test_size=k, shuffle=False)
    
    # Surrogate model & autoencoder
    
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
            f_surrogate, model_f, g_surrogate, model_g, loss_all = find_surrogate_reduced_correlated_models_invCDF(f_data_train, g_data_train, f_output_train, g_output_train, 'together', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, alpha, epochs, sequential, show=False, X_f_test=f_data_test, X_g_test=g_data_test, Y_f_test=f_output_test, Y_g_test=g_output_test)
            loss = loss_all[1][-1]
        else:
            f_surrogate, model_f, loss_f = find_surrogate_reduced_model(f_data_train, f_output_train, 'f', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, show=False, X_test=f_data_test, Y_test=f_output_test)
            g_surrogate, model_g, loss_g = find_surrogate_reduced_model(g_data_train, g_output_train, 'g', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, show=False, X_test=g_data_test, Y_test=g_output_test)
            loss = loss_f[1][-1] + loss_g[1][-1]
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
    df.to_csv(base_path + '/optuna/' + name + '_SURROGATE_AE.csv', index=False)

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
        f_surrogate, model_f, g_surrogate, model_g, _ = find_surrogate_reduced_correlated_models_invCDF(f_data_train, g_data_train, f_output_train, g_output_train, 'together', dim_reduced, activation, layers_surrogate_best, neurons_surrogate_best, layers_AE_best, neurons_AE_best, lr_best, gamma_best, alpha_best, epochs, sequential, show=False)
    
    else:
        f_surrogate, model_f, loss_f = find_surrogate_reduced_model(f_data_train, f_output_train, 'f', dim_reduced, activation, layers_surrogate_best, neurons_surrogate_best, layers_AE_best, neurons_AE_best, lr_best, gamma_best, epochs, show=False)
        g_surrogate, model_g, loss_g = find_surrogate_reduced_model(g_data_train, g_output_train, 'g', dim_reduced, activation, layers_surrogate_best, neurons_surrogate_best, layers_AE_best, neurons_AE_best, lr_best, gamma_best, epochs, show=False)

    # Normalizing flow for AE
    
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
        df.to_csv(base_path + '/optuna/' + name + '_NF_AE.csv', index=False)
    
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


# Normalizing flow

def find_normalizing_flow(data, layers, neurons, epochs, name, lr, gamma, show=True, data_test=None):

    dim = data.shape[1]
    dist_base = nf.distributions.base.DiagGaussian(dim)

    flows = []
    b = torch.tensor([1 if i%2 == 0 else 0 for i in range(dim)])
    for i in range(layers):
        s = nf.nets.MLP([dim, 4, dim], init_zeros=True)
        t = nf.nets.MLP([dim, 4, dim], init_zeros=True)
        if i%2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(dim)]

    flow = nf.NormalizingFlow(dist_base, flows)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    losses = [[], []]

    for step in tqdm(range(epochs), disable = not show):

        optimizer.zero_grad()
        loss = flow.forward_kld(data)
        losses[0].append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

        if data_test is not None:
            loss_output = flow.forward_kld(data_test).item()
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)

    def T(z):
        x = flow.forward(z)
        if z.ndim == 1:
            return x[0]
        else:
            return x

    def T_inverse(x):
        z = flow.inverse(x)
        if x.ndim == 1:
            return z[0]
        else:
            return z

#   if show:
#       plt.figure()
#       plt.plot(losses[0], label='Train')
#       plt.plot(losses[1], label='Test')
#       plt.title("Loss normalizing flow " + name)
#       plt.legend()
#       plt.show(block=False)
#       plt.close()

    return T, T_inverse, losses[1][-1]

def find_normalizing_flow_1D(data, layers, neurons, epochs, name, lr, gamma, show=True, data_test=None):

    dist_base = nf.distributions.DiagGaussian(2)

    flows = []
    b = torch.tensor([1 if i%2 == 0 else 0 for i in range(2)])
    for i in range(layers):
        s = nf.nets.MLP([2, neurons, 2], init_zeros=True)
        t = nf.nets.MLP([2, neurons, 2], init_zeros=True)
        if i%2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(2)]

    flow = nf.NormalizingFlow(dist_base, flows)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    losses = [[], []]

    data_augmented = torch.stack((data[:,0], torch.normal(0, 1, (data.shape[0],)))).T
    if data_test is not None:
        data_test_augmented = torch.stack((data_test[:,0], torch.normal(0, 1, (data_test.shape[0],)))).T

    for step in tqdm(range(epochs), disable = not show):

        optimizer.zero_grad()
        loss = flow.forward_kld(data_augmented)
        losses[0].append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

        if data_test is not None:
            loss_output = flow.forward_kld(data_test_augmented).item()
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)

    def T(z):
        with torch.no_grad():
            z_augmented = torch.stack((z[:,0], torch.zeros(z.shape[0]))).T
            x = flow.forward(z_augmented)
            return torch.unsqueeze(x[:,0], 1)

    def T_inverse(y):
        x = y.clone()
        x_augmented = torch.stack((x[:,0], torch.zeros(x.shape[0]))).T
        z = flow.inverse(x_augmented)
        return torch.unsqueeze(z[:,0], 1)

#   if show:
#       plt.figure()
#       plt.plot(losses[0], label='Train')
#       plt.plot(losses[1], label='Test')
#       plt.title("Loss normalizing flow " + name)
#       plt.legend()
#       plt.show(block=False)
#       plt.close()

    return T, T_inverse, losses[1][-1]

def find_normalizing_flow_spline(data, epochs, name, lr, gamma, show=True, data_test=None):

    dim = data.shape[1]
    dist_base = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)), 1)
    
    bijector = bij.Spline()
    flow = dist.Flow(dist_base, bijector)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    losses = [[], []]

    for step in tqdm(range(epochs), disable = not show):
        
        optimizer.zero_grad()
        loss = -flow.log_prob(data).mean()
        losses[0].append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if data_test is not None:
            loss_output = -flow.log_prob(data_test).mean().item()
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)

    def T(z):
        if z.dim() == 1:
            z = torch.unsqueeze(z, 0)
        x = flow.bijector.forward(z)
        if z.ndim == 1:
            return x[0]
        else:
            return x

    def T_inverse(x):
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        z = flow.bijector.inverse(x)
        if x.ndim == 1:
            return z[0]
        else:
            return z

#   if show:
#       plt.figure()
#       plt.plot(losses[0], label='Train')
#       plt.plot(losses[1], label='Test')
#       plt.title("Loss normalizing flow " + name)
#       plt.legend()
#       plt.show(block=False)
#       plt.close()

    return T, T_inverse, losses[1][-1]

def find_normalizing_flow_invCDF(data):
    
    Finv = lambda a: torch.sort(data, dim=0)[0][(data.shape[0]*a).type(torch.int)]
    T = lambda z: Finv((torch.erf(z[:,0]/np.sqrt(2)) + 1)/2)
    
    F = lambda x: torch.sum((data <= x[:,0]).type(torch.int), dim=0)/(data.shape[0] + 1)
    T_inverse = lambda x: torch.unsqueeze(np.sqrt(2)*torch.erfinv(2*F(x) - 1), 1)

    return T, T_inverse

# Functions to read data

def read_data(mode, data_file, QoI_name = None, expected_samples = None, expected_data_fields = None):
    f = open(data_file)
    data = json.load(f)

    samples = list(data.keys())
    data_fields = list(data[samples[0]].keys()) 

    num_samples = len(samples)
    num_data_fields = len(data_fields)

    if (expected_samples):
        if (not (samples == expected_samples)):
            raise RuntimeError("ERROR: samples != expected_samples for "+data_file)

    if (expected_data_fields):
        if (not (data_fields == expected_data_fields)):
            raise RuntimeError("ERROR: data_fields != expected_data_fields for "+data_file)

    if (mode == "simulations"):
        QoI = np.zeros(num_samples)
        for sample_idx, sample in enumerate(samples):
            QoI[sample_idx] = data[sample][QoI_name]
        return samples, data_fields, QoI
    
    elif (mode == "parameters"):
        parameters = np.zeros((num_samples, num_data_fields))
        for sample_idx, sample in enumerate(samples):
            for name_idx, name in enumerate(data_fields):
                parameters[sample_idx, name_idx] = data[sample][name]
        return samples, data_fields, parameters

    else:
        raise RuntimeError("ERROR: Invalid option for 'mode'. Should be 'simulations'/'parameters'.")

def read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_LF_file, parameters_HF_file, all_LF_data_file, all_HF_data_file, parameters_file_propagation = None, pilot_AE_LF_data_file = None, prop_LF_data_file = None, prop_AE_LF_data_file = None):

    #samples_params, param_names, parameters = read_data("parameters", parameters_file)
    samples_params, param_names, parameters_LF = read_data("parameters", parameters_LF_file)
    samples_params, param_names, parameters_HF = read_data("parameters", parameters_HF_file)
    samples_LF, data_fields_LF, QoI_LF = read_data("simulations", all_LF_data_file, QoI_LF_name)
    samples_HF, data_fields_HF, QoI_HF = read_data("simulations", all_HF_data_file, QoI_HF_name)

    #num_params = len(param_names)
    num_samples_LF = len(samples_LF)
    num_samples_HF = len(samples_HF)

    num_samples = min(num_samples_LF, num_samples_HF)
    
    if (not (samples_LF[0:num_samples] == samples_HF[0:num_samples])):
        raise RuntimeError("ERROR: not (samples_LF[0:num_samples] == samples_HF[0:num_samples])")
    samples = samples_LF[0:num_samples]
    QoI_LF = QoI_LF[0:num_samples]
    parameters_LF = parameters_LF[0:num_samples,:]
    parameters_HF = parameters_HF[0:num_samples,:]

    if (parameters_file_propagation):
        samples_params_prop, param_names_prop, parameters_prop = read_data("parameters", parameters_file_propagation, None, None, param_names)
    else:
        parameters_prop = None

    if (pilot_AE_LF_data_file):
        _, _, pilot_AE_QoI_LF = read_data("simulations", pilot_AE_LF_data_file, QoI_LF_name)
    else:
        pilot_AE_QoI_LF = None

    if (prop_LF_data_file):
        _, _, prop_QoI_LF = read_data("simulations", prop_LF_data_file, QoI_LF_name, None, data_fields_LF)
    else:
        prop_QoI_LF = None

    if (prop_AE_LF_data_file):
        _, _, prop_AE_QoI_LF = read_data("simulations", prop_AE_LF_data_file, QoI_LF_name, None, data_fields_LF)
    else:
        prop_AE_QoI_LF = None
    
    return samples, parameters_LF, parameters_HF, QoI_LF, QoI_HF, parameters_prop, pilot_AE_QoI_LF, prop_QoI_LF, prop_AE_QoI_LF

def write_normalized_data(base_path, data_files_json, QoI_LF_name, QoI_HF_name, num_pilot_samples_to_use, trial_name_str = ""):

    parameters_HF_file = data_files_json["HF_inputs"]
    parameters_LF_file = data_files_json["HF_inputs"]
    all_LF_data_file = data_files_json["LF_outputs_pilot"]
    all_HF_data_file = data_files_json["HF_outputs"]
    parameters_file_propagation = data_files_json["LF_inputs_propagation"]

    # Read data

    samples, parameters_LF, parameters_HF, QoI_LF, QoI_HF, parameters_propagation, _, _, _ = read_simulation_data(QoI_LF_name, QoI_HF_name, 
            parameters_LF_file, parameters_HF_file, all_LF_data_file, all_HF_data_file, parameters_file_propagation)
    num_params_LF = parameters_LF.shape[1]
    num_params_HF = parameters_HF.shape[1]

    # Number of pilot samples to use
    if num_pilot_samples_to_use != -1: # If -1, use all samples
        num_samples = parameters_HF.shape[0]
        sample_idxs = np.random.default_rng().integers(0, num_samples, num_pilot_samples_to_use)
        QoI_LF = QoI_LF[sample_idxs]
        QoI_HF = QoI_HF[sample_idxs]
        parameters_LF = parameters_LF[sample_idxs,:]
        parameters_HF = parameters_HF[sample_idxs,:]

    # Values parameters

    R_min = 0.5
    R_max = 2.0

    R_cor_min = 0.70
    R_cor_max = 1.25

    min_parameters_LF = np.array([R_min]*(num_params_LF-1))
    min_parameters_LF = np.append(min_parameters_LF, R_cor_min)
    max_parameters_LF = np.array([R_max]*(num_params_LF-1))
    max_parameters_LF = np.append(max_parameters_LF, R_cor_max)

    min_parameters_HF = np.array([R_min]*(num_params_HF-1))
    min_parameters_HF = np.append(min_parameters_HF, R_cor_min)
    max_parameters_HF = np.array([R_max]*(num_params_HF-1))
    max_parameters_HF = np.append(max_parameters_HF, R_cor_max)

    # Normalization

    min_QoI_HF = np.min(QoI_HF) - 0.01
    max_QoI_HF = np.max(QoI_HF) + 0.01
    min_QoI_LF = np.min(QoI_LF) - 0.01
    max_QoI_LF = np.max(QoI_LF) + 0.01

    parameters_LF_normalized = np.empty(parameters_LF.shape)
    parameters_HF_normalized = np.empty(parameters_HF.shape)
    parameters_propagation_normalized = np.empty(parameters_propagation.shape)
    
    # Normalize LF parameters
    for j in range(parameters_LF.shape[1]):
        parameters_LF_normalized[:,j] = (2*parameters_LF[:,j] - min_parameters_LF[j] - max_parameters_LF[j])/(max_parameters_LF[j] - min_parameters_LF[j])
        parameters_propagation_normalized[:,j] = (2*parameters_propagation[:,j] - min_parameters_LF[j] - max_parameters_LF[j])/(max_parameters_LF[j] - min_parameters_LF[j])
    
    # Normalize HF parameters
    for j in range(parameters_HF.shape[1]):
        parameters_HF_normalized[:,j] = (2*parameters_HF[:,j] - min_parameters_HF[j] - max_parameters_HF[j])/(max_parameters_HF[j] - min_parameters_HF[j])
        
    QoI_HF_normalized = (2*QoI_HF - min_QoI_HF - max_QoI_HF)/(max_QoI_HF - min_QoI_HF)
    QoI_LF_normalized = (2*QoI_LF - min_QoI_LF - max_QoI_LF)/(max_QoI_LF - min_QoI_LF)

    # Save normalized data
    
    base_path = os.path.abspath(base_path)

    if not os.path.exists(base_path + "/results"):
        os.mkdir(base_path + "/results")

    np.savetxt(base_path + "/results/parameters_LF_normalized"+trial_name_str+".csv", parameters_LF_normalized, delimiter=",")
    np.savetxt(base_path + "/results/parameters_HF_normalized"+trial_name_str+".csv", parameters_HF_normalized, delimiter=",")
    np.savetxt(base_path + "/results/parameters_propagation_normalized"+trial_name_str+".csv", parameters_propagation_normalized, delimiter=",")

    np.savetxt(base_path + "/results/QoI_HF_normalized"+trial_name_str+".csv", QoI_HF_normalized, delimiter=",")
    np.savetxt(base_path + "/results/QoI_LF_normalized"+trial_name_str+".csv", QoI_LF_normalized, delimiter=",")

    if num_pilot_samples_to_use != -1:
        np.savetxt(base_path + "/results/sample_idxs"+trial_name_str+".csv", sample_idxs, fmt='%i', delimiter=",")


def write_unnormalized_data(base_path, config_string, trial_name_str = ""):

    base_path = os.path.abspath(base_path)
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
