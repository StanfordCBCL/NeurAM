import torch
import numpy as np

def get_model(name):
    
    d = 5
    
    left = [0.05, 1, 0.5, 0.5, 0.1]
    right = [0.2, 5, 3, 3, 1]
    
    def omega(x):
        y = torch.empty(x.shape)
        if y.ndim == 1:
            for i in range(d):
                y[i] = torch.exp(np.log(left[i]) + ((np.log(right[i]) - np.log(left[i]))/2)*(x[i] + 1))
        else:
            for i in range(d):
                y[:,i] = torch.exp(np.log(left[i]) + ((np.log(right[i]) - np.log(left[i]))/2)*(x[:,i] + 1))
        return y
                    
    if name == 'Hartmann_flow_velocity':
        
        g = lambda y: - y[2]*(y[3]/(y[4]**2))*(1 - (y[4]/torch.sqrt(y[3]*y[0]))*(torch.cosh(y[4]/torch.sqrt(y[3]*y[0]))/torch.sinh(y[4]/torch.sqrt(y[3]*y[0])))) if y.ndim==1 else \
                      - y[:,2]*(y[:,3]/(y[:,4]**2))*(1 - (y[:,4]/torch.sqrt(y[:,3]*y[:,0]))*(torch.cosh(y[:,4]/torch.sqrt(y[:,3]*y[:,0]))/torch.sinh(y[:,4]/torch.sqrt(y[:,3]*y[:,0]))))
        
    elif name == 'Hartmann_magnetic_field':
        
        g = lambda y: y[2]*(1/(2*y[4]))*(1 - 2*(torch.sqrt(y[3]*y[0])/y[4])*(torch.sinh(y[4]/torch.sqrt(4*y[3]*y[0]))/torch.cosh(y[4]/torch.sqrt(4*y[3]*y[0])))) if y.ndim==1 else \
                      y[:,2]*(1/(2*y[:,4]))*(1 - 2*(torch.sqrt(y[:,3]*y[:,0])/y[:,4])*(torch.sinh(y[:,4]/torch.sqrt(4*y[:,3]*y[:,0]))/torch.cosh(y[:,4]/torch.sqrt(4*y[:,3]*y[:,0]))))

    else:

        raise RuntimeError('ERROR: Invalid QoI name.')
    
    f = lambda x: g(omega(x))
        
    return d, f
