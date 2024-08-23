import torch
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import normflows as nf
import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import numpy as np
import json

# Surrogate model & autoencoder

def find_surrogate_reduced_model(X, Y, name, dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, absolute_val, show=True, X_test=None, Y_test=None, return_surrogate_NN=False):

    _, dim = X.shape

    class Autoencoder(torch.nn.Module):

        def __init__(self):
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

        def __init__(self):
            super().__init__()

            model_structure = [torch.nn.Linear(dim_reduced, neurons_surrogate), activation]
            for i in range(layers_surrogate-1):
                model_structure += [torch.nn.Linear(neurons_surrogate, neurons_surrogate), activation]
            model_structure += [torch.nn.Linear(neurons_surrogate, 1)]
            self.net = torch.nn.Sequential(*model_structure)

        def forward(self, x):
            output = self.net(x)
            return output

    model = Autoencoder()
    surrogate = Surrogate_NN()

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
       
        if (absolute_val):
            for p in surrogate.parameters():
                #p.data.clamp_(0.02)
                p.data.abs_()

        if X_test is not None:
            reconstructed_test = model(X_test)
            loss_output =  (loss_function(Y_test, torch.squeeze(surrogate(model.encoder(reconstructed_test)))) \
                          + loss_function(Y_test, torch.squeeze(surrogate(model.encoder(X_test)))) \
                          + loss_function(torch.squeeze(surrogate(model.encoder(reconstructed_test))), torch.squeeze(surrogate(model.encoder(X_test)))) \
                          + loss_function(model(reconstructed_test), reconstructed_test)).item()    
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)

    if show:
        plt.figure()
        plt.semilogy(losses[0], label='Train')
        if X_test is not None:
            plt.semilogy(losses[1], label='Test')
        plt.title("Loss surrogate + autoencoder " + name)
        plt.legend()
        plt.show(block=False)
        #plt.show()
        plt.close()

    def f_surrogate(x):
        model.eval()
        surrogate.eval()
        y = torch.squeeze(surrogate(model.encoder(x))).detach()
        return y
    
    if return_surrogate_NN:
        return f_surrogate, model, losses[1][-1], surrogate
    else:
        return f_surrogate, model, losses[1][-1]

def find_surrogate_reduced_correlated_models_invCDF(X_f, X_g, Y_f, Y_g, name, dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, alpha, epochs, sequential, absolute_val, show=True, X_f_test=None, X_g_test=None, Y_f_test=None, Y_g_test=None):

    if sequential:
        _, model_f, _, surrogate_f = find_surrogate_reduced_model(X_f, Y_f, 'together_f', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, show, X_test=X_f_test, Y_test=Y_f_test, return_surrogate_NN=True)  
        _, model_g, _, surrogate_g = find_surrogate_reduced_model(X_g, Y_g, 'together_g', dim_reduced, activation, layers_surrogate, neurons_surrogate, layers_AE, neurons_AE, lr, gamma, epochs, show, X_test=X_g_test, Y_test=Y_g_test, return_surrogate_NN=True)  
    else:
        _, dim_f = X_f.shape
        _, dim_g = X_g.shape

        class Autoencoder(torch.nn.Module):

            def __init__(self, dim):
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

            def __init__(self):
                super().__init__()

                model_structure = [torch.nn.Linear(dim_reduced, neurons_surrogate), activation]
                for i in range(layers_surrogate-1):
                    model_structure += [torch.nn.Linear(neurons_surrogate, neurons_surrogate), activation]
                model_structure += [torch.nn.Linear(neurons_surrogate, 1)]
                self.net = torch.nn.Sequential(*model_structure)

            def forward(self, x):
                output = self.net(x)
                return output

        model_f = Autoencoder(dim_f)
        model_g = Autoencoder(dim_g)
        surrogate_f = Surrogate_NN()
        surrogate_g = Surrogate_NN()
    
    optimizer = torch.optim.Adam(list(model_f.parameters()) + list(model_g.parameters()) + list(surrogate_f.parameters()) + list(surrogate_g.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    loss_function = torch.nn.MSELoss()
    loss_pearson = torchmetrics.PearsonCorrCoef()
    losses = [[], []]

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
        
        losses[0].append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if absolute_val:
            for p in surrogate_f.parameters():
                #p.data.clamp_(0.02)
                p.data.abs_()
            for p in surrogate_g.parameters():
                #p.data.clamp_(0.02)
                p.data.abs_()

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
                          
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)

    if show:
        plt.figure()
        plt.semilogy([x + alpha for x in losses[0]], label='Train')
        if X_f_test is not None and X_g_test is not None:
            plt.semilogy([x + alpha for x in losses[1]], label='Test')
        plt.title("Loss surrogate + autoencoder " + name)
        plt.legend()
        plt.show(block=False)
        #plt.show()
        plt.close()

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

    return f_surrogate, model_f, g_surrogate, model_g, losses[1][-1]
    #return f_surrogate, model_f, g_surrogate, model_g, None

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

    if show:
        plt.figure()
        plt.plot(losses[0], label='Train')
        plt.plot(losses[1], label='Test')
        plt.title("Loss normalizing flow " + name)
        plt.legend()
        plt.show(block=False)
        plt.close()

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

    if show:
        plt.figure()
        plt.plot(losses[0], label='Train')
        plt.plot(losses[1], label='Test')
        plt.title("Loss normalizing flow " + name)
        plt.legend()
        plt.show(block=False)
        plt.close()

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

    if show:
        plt.figure()
        plt.plot(losses[0], label='Train')
        plt.plot(losses[1], label='Test')
        plt.title("Loss normalizing flow " + name)
        plt.legend()
        plt.show(block=False)
        plt.close()

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

def read_simulation_data(QoI_LF_name, QoI_HF_name, parameters_file, all_0d_data_file, all_3d_data_file, parameters_file_propagation = None, pilot_AE_0d_data_file = None, prop_0d_data_file = None, prop_AE_0d_data_file = None):

    samples_params, param_names, parameters = read_data("parameters", parameters_file)
    samples_0d, data_fields_0d, QoI_LF = read_data("simulations", all_0d_data_file, QoI_LF_name)
    samples_3d, data_fields_3d, QoI_HF = read_data("simulations", all_3d_data_file, QoI_HF_name)

    num_params = len(param_names)
    num_samples_0d = len(samples_0d)
    num_samples_3d = len(samples_3d)

    num_samples = min(num_samples_0d, num_samples_3d)
    
    if (not (samples_0d[0:num_samples] == samples_3d[0:num_samples])):
        raise RuntimeError("ERROR: not (samples_0d[0:num_samples] == samples_3d[0:num_samples])")
    samples = samples_0d[0:num_samples]
    QoI_LF = QoI_LF[0:num_samples]
    parameters = parameters[0:num_samples,:]

    if (parameters_file_propagation):
        samples_params_prop, param_names_prop, parameters_prop = read_data("parameters", parameters_file_propagation, None, None, param_names)
    else:
        parameters_prop = None

    if (pilot_AE_0d_data_file):
        #_, _, pilot_AE_QoI_LF = read_data("simulations", pilot_AE_0d_data_file, QoI_LF_name, samples)
        _, _, pilot_AE_QoI_LF = read_data("simulations", pilot_AE_0d_data_file, QoI_LF_name)
    else:
        pilot_AE_QoI_LF = None

    if (prop_0d_data_file):
        _, _, prop_QoI_LF = read_data("simulations", prop_0d_data_file, QoI_LF_name, None, data_fields_0d)
    else:
        prop_QoI_LF = None

    if (prop_AE_0d_data_file):
        _, _, prop_AE_QoI_LF = read_data("simulations", prop_AE_0d_data_file, QoI_LF_name, None, data_fields_0d)
    else:
        prop_AE_QoI_LF = None
    
    return samples, parameters, QoI_LF, QoI_HF, parameters_prop, pilot_AE_QoI_LF, prop_QoI_LF, prop_AE_QoI_LF

