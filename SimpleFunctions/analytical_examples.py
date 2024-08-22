# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:22:44 2024

@author: andre
"""

#%% Moduels

import torch
import numpy as np

#%% Models

def get_model(name):
        
    if name == 'Q_HF':
        d = 2
        f = lambda x: torch.exp(0.7*x[0] + 0.3*x[1]) + 0.15*torch.sin(2*np.pi*x[0]) if x.ndim==1 else \
                      torch.exp(0.7*x[:,0] + 0.3*x[:,1]) + 0.15*torch.sin(2*np.pi*x[:,0])
                      
    elif name == 'Q_LF':
        d = 2
        f = lambda x: torch.exp(0.01*x[0] + 0.99*x[1]) + 0.15*torch.sin(3*np.pi*x[1]) if x.ndim==1 else \
                      torch.exp(0.01*x[:,0] + 0.99*x[:,1]) + 0.15*torch.sin(3*np.pi*x[:,1])
        
    return d, f