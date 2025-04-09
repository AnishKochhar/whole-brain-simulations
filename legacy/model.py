# Anish Kochhar, Imperial College London, February 2025

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

class ModelParams:
    """
    Parameters container for Whole-Brain Model equations
    Current Models:
        1. Dynamic Mean Field (Reduced Wong-Wang)
        
        2. Balloon-Windkessel Haemodynamic model
    """

    def __init__(self, **kwargs):
        ## DMF parameters
        #  starting states taken from Griffiths et al. 2022
        self.std_in    = [0.02, 0]         # Input noise standard deviation

        self.W_E       = [1.0, 0]          # Scale for external input to excitatory population
        self.tau_E     = [100.0, 0]        # Decay time (ms) for excitatory synapses
        self.gamma_E   = [0.641/1000.0, 0] # Kinetic parameter for excitatory dynamics

        self.W_I       = [0.7, 0]          # Scale for external input to inhibitory population
        self.tau_I     = [10.0, 0]         # Decay time for inhibitory synapses
        self.gamma_I   = [1.0/1000.0, 0]   # Kinetic parameter for inhibitory dynamics

        self.I_0       = [0.32, 0]         # Constant external input

        # Coupling parameters:
        self.g         = [20.0, 0]         # Global coupling (long-range)
        self.g_EE      = [0.1, 0]          # Local excitatory self-feedback
        self.g_IE      = [0.1, 0]          # Inhibitory-to-excitatory coupling
        self.g_EI      = [0.1, 0]          # Excitatory-to-inhibitory coupling

        # Sigmoid parameters for conversion of current to firing rate:
        self.std_out   = [0.02, 0]         # Output noise standard deviation
        self.aE        = [310.0, 0]
        self.bE        = [125.0, 0]
        self.dE        = [0.16, 0]
        self.aI        = [615.0, 0]
        self.bI        = [177.0, 0]
        self.dI        = [0.087, 0]

        self.mu        = [0.5, 0]          # Parameter used in delay computation
        
        ## Balloon-Windkessel parameters
        self.alpha     = [0.32, 0]
        self.rho       = [0.34, 0]
        self.k1        = [2.38, 0]
        self.k2        = [2.0, 0]
        self.k3        = [0.48, 0]
        self.V         = [0.02, 0]         # V0 in the BOLD equation
        self.E0        = [0.34, 0]
        self.tau_s     = [0.65, 0]
        self.tau_f     = [0.41, 0]
        self.tau_0     = [0.98, 0]

        for k, v in kwargs.items():
            setattr(self, k, v)


# MARK: Model Implementations
class DMF(nn.Module):
    """
    Dynamics Mean Field (DMF) model for neural activity
    We model two states per node: E (excitatory) and I (inhibitory)
    The ODEs are:

        dE/dt = -E/tau_E + (1 - E) * gamma_E * R_E + noise
        dI/dt = -I/tau_I + R_I + noise
    
    where the firing rates R_E and R_I are given by:
    
        R_E = h_tf(aE, bE, dE, I_E)
        R_I = h_tf(aI, bI, dI, I_I)
    
    and the input currents are:
    
        I_E = W_E * I_0 + g_EE * E + g * (Laplacian @ E) - g_IE * I
        I_I = W_I * I_0 + g_EI * E - I
    """

    def __init__(self, SC: np.ndarray, delays: np.ndarray, params: ModelParams, dt: float=0.05, T: int=100):
        """
        Parameters:
        SC:             Structural connectivity as (nodes x nodes) numpy array
        params:         ModelParams instance
        dt:             Integration timestep
        T:              Number of integration steps per batch
        """
        super(DMF, self).__init__()
        self.params = params
        self.dt = dt
        self.T = T
        self.node_size = SC.shape[0]

        # Store SC as fixed tensor
        self.register_buffer('SC', torch.tensor(SC, dtype=torch.float32))

        # Precompute Laplacian L = D - C (assume SC is log-transformed and normalized)
        D = np.diag(np.sum(SC, axis=1))
        L = D - SC
        self.register_buffer('L', torch.tensor(L, dtype=torch.float32))

        # Delays provided as numpy array of (node_size, node_size)
        self.register_buffer('delays', torch.tensor(delays, dtype=torch.int64))

        # Register all parameters as torch tensors
        parameters = [a for a in dir(params) if not a.startswith('__') and not callable(getattr(params, a))]
        for parameter in parameters:
            mean, var = getattr(params, parameter)
            setattr(self, parameter, Parameter(torch.tensor(mean \
                        + 1 / np.sqrt(var) * np.random.randn(1, )[0], dtype=torch.float32)))
            print(f'Set attribute {parameter} with mean {getattr(self, parameter)}')

        # State dimension (E and I)
        self.state_size = 2

    def h_tf(self, a, b, d, current):
        """
        Transformation for firing rates of excitatory and inhibitory pools
        Takes variables a, b, current and convert into a linear equation (a * current - b) while adding a small
        amount of noise (1e-5) while dividing that term to an exponential of itself multiplied by constant d for
        the appropriate dimensions
        """
        num = 1e-5 + torch.abs(a * current - b)
        den = 1e-5 * d + torch.abs(1.000 - torch.exp(-d * (a * current - b)))
        return torch.divide(num, den + 1e-8)
    
    def forward_batch(self, x0: torch.tensor=None, hE: torch.tensor=None, noise: torch.Tensor=None):
        """
        Integrates DMF ODEs over T time steps
        
        Parameters:
            x0: initial state of shape (node_size, 2). If None, initialize to zeros
            hE: delay buffer from previous batches (node_size, max_delay)
            noise: noise tensor of shape (node_size, 2). If None, use zeros
        
        Returns:
            X: tensor of shape (T, node_size, 2) containing state time series
        """
        N = self.node_size
        p = self.params
        dt = self.dt

        if x0 is None:
            x = torch.zeros(N, self.state_size, dtype=torch.float32)
        else:
            x = x0.clone()
        if noise is None:
            noise = torch.randn_like(x) * self.std_in * np.sqrt(dt) 


        X = [x.unsqueeze(0)] # record initial state

        for t in range(self.T - 1):
            E = x[:, 0:1] # (N, 1)
            I = x[:, 1:2] # (N, 1)

            # Compute delayed excitatory input
            delayed_E = torch.gather(hE, 1, self.delays) # (N, N)
            delayed_E = delayed_E.mean(dim=1, keepdim=True) # (N, 1)

            # Compute input currents
            I_E = self.W_E * self.I_0 + self.g_EE * E + self.g * (self.L @ delayed_E) - self.g_IE * I
            I_I = self.W_I * self.I_0 + self.g_EI * E - I
            
            R_E = self.h_tf(self.aE, self.bE, self.dE, I_E)
            R_I = self.h_tf(self.aI, self.bI, self.dI, I_I)
            
            dE = -E / self.tau_E + (1 - E) * self.gamma_E * R_E
            dI = -I / self.tau_I + R_I
            
            x = x + dt * torch.cat([dE, dI], dim=1) + noise

            # Update display buffer, discarding oldest value
            hE = torch.cat([x[:, 0:1], hE[:, :-1]], dim=1)

            X.append(x.unsqueeze(0))

        X = torch.cat(X, dim=0)
        return X, x, hE


    def forward(self, x0: torch.Tensor, hE: torch.Tensor, batches: int):
        """
        Run DMF forward pass over multiple batches

        Parameters:
            x0: Initial state (node_size, 2)
            hE: Initial delay buffer (node_size, L)
            batches: Number of batches

        Returns:
            X_all: Concatenated activity (T_total, node_size, 2)
            x_final: Final output state
            hE_final: Final delay buffer
        """
        X_all = []
        x = x0.clone()
        for _ in range(batches):
            X_batch, x, hE = self.forward_batch(x, hE)
            X_all.append(X_batch)
        X_all = torch.cat(X_all, dim=0)
        return X_all, x, hE
    

class Balloon(nn.Module):
    """
    Forward model from excitatory activity to BOLD signal using simplified Balloon-Windkessel equations

    Maintains four states per node: [x, f, v, q], which can be used to calcualte the final BOLD signal as:

        BOLD = V * [ k1*(1 - q) + k2*(1 - q/v) + k3*(1 - v) ]
    """
    def __init__(self, params: ModelParams, dt: float=0.05, T: int=50):
        super(Balloon, self).__init__()
        self.params = params
        self.dt = dt
        self.T = T  # number of time steps in one batch

        parameters = [a for a in dir(params) if not a.startswith('__') and not callable(getattr(params, a))]
        for parameter in parameters:
            mean, var = getattr(params, parameter)
            setattr(self, parameter, Parameter(torch.tensor(mean \
                        + 1 / np.sqrt(var) * np.random.randn(1, )[0], dtype=torch.float32)))
            print(f'Set attribute {parameter} with mean {getattr(self, parameter)}')


    
    def forward(self, E: torch.Tensor, x0: torch.Tensor=None, noise: torch.Tensor=None):
        """
        Parameters:
            E: Excitatory activity (T, N, 2)
            x0: Initial haemodynamic state (N, 4)
        
        Returns:
            BOLD: Final bold signal for each node
            xN: Final haemodynamic state (N, 4)
        """
        T, N, _ = E.shape
        p = self.params
        dt = self.dt

        if x0 is None:
            x = torch.zeros(N, dtype=torch.float32)
            f = torch.ones(N, dtype=torch.float32)
            v = torch.ones(N, dtype=torch.float32)
            q = torch.ones(N, dtype=torch.float32)
        else:
            x, f, v, q = x0[:, 0], x0[:, 1], x0[:, 2], x0[:, 3]
        if noise is None:
            noise = torch.randn_like(x) * self.std_out * np.sqrt(dt) 

        for t in range(T):
            E_t = E[t, :, 0]  # excitatory channel, shape (N, )
            dx = E_t - torch.reciprocal(self.tau_s) * x - torch.reciprocal(self.tau_f) * (f - 1)
            df = x
            dv = (f - torch.pow(v, torch.reciprocal(self.alpha))) * torch.reciprocal(self.tau_0)
            dq = (f * (1 - torch.pow(1 - self.rho, torch.reciprocal(f))) * torch.reciprocal(self.rho) \
                   - q * torch.pow(v, torch.reciprocal(self.alpha)) * torch.reciprocal(v+1e-8)) \
                     * torch.reciprocal(self.tau_0)
            
            # dx = E_t - (1 / p.tau_s) * x - (1 / p.tau_f) * (f - 1)
            # df = x
            # dv = (f - torch.pow(v, 1 /p.alpha)) / p.tau_0
            # dq = (f*(1 - np.power(1-p.rho, 1/(f+1e-8)))/p.rho - q*torch.pow(v, 1/p.alpha)/(v+1e-8)) / p.tau_0

            # dq = (f * (1 - np.power(1 - p.rho, 1 / (f + 1e-8))) / p.rho \
            #        - q * torch.pow(v, 1 / p.alpha) / (v + 1e-8)) \
            #          / p.tau_0
            
            
            x = x + dt * dx
            f = f + dt * df
            v = v + dt * dv
            q = q + dt * dq
            
        
        BOLD = self.V * 100.0 * torch.reciprocal(self.E0) * (self.k1 * (1 - q) + \
                                                      (self.k2 * (1 - q * torch.reciprocal(v))) + \
                                                      (self.k3 * (1 - v))) \
                + noise
        state_final = torch.stack([x, f, v, q], dim=1)
        return BOLD.unsqueeze(1), state_final
