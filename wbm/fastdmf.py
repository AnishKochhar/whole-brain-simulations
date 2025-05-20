import torch
import torch.nn as nn
import numpy as np
import psutil, os

from wbm.plotter import Plotter
from wbm.utils import DEVICE

def debug_snapshot(step_ms: float, rho: torch.Tensor,
                   **tensors):
    """
    Pretty print a full numerical health report for the DMF/BW step.

    Parameters
    ----------
    step_ms : float
        simulation time in milliseconds (for the header)
    rho : torch.Tensor
        the rho scalar (needed to evaluate the nasty (1-rho)^(1/f) term)
    tensors : keyword tensors
        any number of named tensors, e.g.   E=E, I=I, s=s, f=f, v=v, q=q,
        I_E=I_E, I_I=I_I, R_E=R_E, R_I=R_I, BOLD=BOLD
    """
    bar = "─" * 109

    for name, x in tensors.items():
        x_detached = x.detach()
        n_nan  = torch.isnan(x_detached).sum().item()
        n_inf  = torch.isinf(x_detached).sum().item()
        n_neg  = (x_detached < 0).sum().item()

        mn, av, mx = (val.item() for val in
                      (x_detached.min(), x_detached.mean(), x_detached.max()))

        flag = ""
        if n_nan or n_inf:
            flag += "  **NaN/Inf!**"
        if name in ("f", "v", "q") and n_neg:
            flag += f"  ({n_neg} neg)"
        if flag:
            print(f"\n{bar}\nNUMERICS CHECK @ {step_ms:7.1f} ms")
            print(f"{name:6} :  min={mn:9.3e}  mean={av:9.3e}  "
              f"max={mx:9.3e}  NaN={n_nan:4}  Inf={n_inf:4}{flag}")

    # explicit test for the dangerous term in dq
    if "f" in tensors:
        f = tensors["f"].detach()
        with torch.no_grad():
            bad_mask = f <= 0
            if bad_mask.any():
                print(f"\n[!]  {bad_mask.sum().item()} voxel(s) with f <= 0 "
                      f"(min f = {f.min().item():.3e})")
            else:
                term = (1 - (1 - rho.item()) ** (1.0 / f)).abs()
                if torch.isinf(term).any() or torch.isnan(term).any():
                    print("\n[!]  (1-rho)^(1/f) produced NaN/Inf")
                elif term.max() > 1e4:
                    print(f"\n[!]  (1-rho)^(1/f) very large (max={term.max().item():.3e})")

    
    if step_ms % 1000 == 0:
        rss = psutil.Process(os.getpid()).memory_info().rss / 2**20
        gpu = torch.cuda.memory_allocated() / 2**20
        print(f"RSS = {rss:,.0f} MB   GPU alloc = {gpu:,.0f} MB\n")


class FastDMFParams:
    def __init__(
        self,
        dt: float = 0.1, # neural time step (ms)
        dtt: float = 1.0, # haemodynamic step (ms), must be >= dt
        tr: float = 750.0, # TR duration (ms)

        # synaptic / DMF parameters
        W_E:   float = 1.0,
        W_I:   float = 0.7,
        I_0:   float = 0.382,
        tau_E: float = 100.0,
        tau_I: float = 10.0,
        gamma_E: float = 0.641 / 1000.0,
        gamma_I: float = 1.0   / 1000.0,
        sigma_E: float = 0.005,
        sigma_I: float = 0.005,

        # sigmoid transform
        aE: float = 310.0, bE: float = 125.0, dE: float = 0.16,
        aI: float = 615.0, bI: float = 177.0, dI: float = 0.087,

        # coupling
        g   : float = 2.0,    # global
        JN  : float = 0.15,   # J_nmda: excitatory synpatic coupling (nA)
        g_EE: float = 1.4,    # w+: local excitatory recurrence
        g_EI: float = 1.0,    # E→I
        g_IE: float = 0.75,   # I→E

        # hemodynamic (Balloon–Windkessel)
        tau_s: float = 0.65,   # s
        tau_f: float = 0.41,   # s
        tau_0: float = 0.98,   # s
        alpha: float = 0.32,
        rho:   float = 0.34,    # 0.4
        k1:    float = 2.38,    # 2.77
        k2:    float = 2.0,     # 0.4
        k3:    float = 0.48,    # 1
        V0:    float = 0.2,
        E0:    float = 0.34,
        
        # variants + logging
        use_delay_based:        bool = False,
        inhibitory_gain_scalar: bool = False,
        verbose:                bool = False

    ):

        # time constants
        self.dt = dt
        self.dtt = dtt
        self.tr = tr
        self.hidden_steps = int(np.round(tr / dt))
        assert dtt >= dt, "dtt must be >= dt"
        self.hemo_steps_per_tr = int(np.round(tr / dtt))
        self.dt_ratio = int(np.round(dtt / dt)) # Number of neural timesteps per haemodynamic step

        # synaptic
        self.W_E, self.W_I = W_E, W_I
        self.I_0 = I_0
        self.tau_E, self.tau_I = tau_E, tau_I
        self.gamma_E, self.gamma_I = gamma_E, gamma_I
        self.sigma_E, self.sigma_I = sigma_E, sigma_I

        # sigmoid
        self.aE, self.bE, self.dE = aE, bE, dE
        self.aI, self.bI, self.dI = aI, bI, dI

        # coupling
        self.JN = JN
        self.g, self.g_EE, self.g_EI, self.g_IE = g, g_EE, g_EI, g_IE

        # haemodynamics
        self.tau_s = tau_s # * 1000.0
        self.tau_f = tau_f # * 1000.0
        self.tau_0 = tau_0 # * 1000.0
        self.alpha, self.rho = alpha, rho
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.V0, self.E0 = V0, E0
        
        # variants
        self.use_delay_based        = use_delay_based
        self.inhibitory_gain_scalar = inhibitory_gain_scalar
        self.verbose                = verbose


class WholeBrainFastDMF(nn.Module):
    def __init__(self, params: FastDMFParams, distance_matrix: torch.Tensor, \
                node_size: int, input_size: int, batch_size: int, delays_max: int):
        super().__init__()
        self.params = params
        self.node_size = node_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = params.hidden_steps
        self.delays_max = delays_max
        self.state_size = 6 # [E, I, s, f, v, q]

        self.dt = torch.tensor(params.dt)

        self.register_buffer('dist', distance_matrix.unsqueeze(0)) # (1, N, N)

        speed = 1.5  # m/s
        delay_ms = self.dist / speed  # ms
        delay_steps = (delay_ms / params.dt).floor().clamp(0, delays_max-1).long()
        self.register_buffer('delay_steps', delay_steps)  # (1, N, N)

        # Plotter.plot_matrix(delay_steps.squeeze(0), title=f"Delay Steps dt = {params.dt}", cmap="inferno")

        def T(x): return torch.tensor(x, device=self.dist.device, dtype=torch.float32)
        
        self.dt      = T(params.dt)
        self.dtt     = T(params.dtt)
        self.dtt_s   = T(params.dtt / 1000.0)
        self.g       = nn.Parameter(T(params.g))
        self.JN      = T(params.JN)
        self.g_EE    = T(params.g_EE)
        self.g_EI    = T(params.g_EI)
        self.g_IE    = T(params.g_IE)
        self.W_E     = T(params.W_E)
        self.W_I     = T(params.W_I)
        self.I_0     = T(params.I_0)
        self.tau_E   = T(params.tau_E)
        self.tau_I   = T(params.tau_I)
        self.gamma_E = T(params.gamma_E)
        self.gamma_I = T(params.gamma_I)
        self.sigma_E = T(params.sigma_E)
        self.sigma_I = T(params.sigma_I)
        
        self.aE, self.bE, self.dE = T(params.aE), T(params.bE), T(params.dE)
        self.aI, self.bI, self.dI = T(params.aI), T(params.bI), T(params.dI)
        
        self.itau_s_h, self.itau_f_h, self.itau_0_h = T(1.0 / params.tau_s), T(1.0 / params.tau_f), T(1.0 / params.tau_0)
        
        self.ialpha_h, self.rho_h = T(1 / params.alpha), T(params.rho)
        self.k1_h, self.k2_h, self.k3_h = T(params.k1), T(params.k2), T(params.k3)
        self.V0_h, self.E0_h = T(params.V0), T(params.E0)

        # Log all hyperparameters and parameters
        param_dict = {k: v for k, v in vars(params).items() if not k.startswith('__') and not callable(v)}
        model_args = {
            'node_size': node_size,
            'input_size': input_size,
            'batch_size': batch_size,
            'delays_max': delays_max,
        }
        print("[Model] Params: " + ', '.join(f"{k}={v}" for k, v in {**param_dict, **model_args}.items()))

    def generate_initial_states(self):
        """
        Generates the initial state for RWW (DMF) foward function. Uses same initial states as in the Griffiths et al. code

        Returns:
            initial_state: torch.Tensor of shape (node_size, state_size, batch_size)
        """
        initial_state = 0.45 * np.random.uniform(0, 1, (self.node_size, self.state_size, self.batch_size))
        baseline = np.array([0, 0, 0, 1.0, 1.0, 1.0]).reshape(1, self.state_size, 1)
        initial_state = initial_state + baseline
        state_means = initial_state.mean(axis=(0, 2))
        E_mean, I_mean, x_mean, f_mean, v_mean, q_mean = state_means
        print(f"BASE | E={E_mean:.4f} I={I_mean:.4f} x={x_mean:.4f} f={f_mean:.4f} v={v_mean:.4f} q={q_mean:.4f}")
        return torch.tensor(initial_state, dtype=torch.float32, device=DEVICE)

    def firing_rate(self, a, b, d, current):
        """
        Transformation for firing rates of excitatory and inhibitory pools
        Takes variables a, b, current and convert into a linear equation (a * current - b) while adding a small
        amount of noise (1e-8) while dividing that term to an exponential of itself multiplied by constant d for
        the appropriate dimensions
        """
        x = a * current - b
        with torch.no_grad():
            bad_x = x[(torch.abs(1 - torch.exp(-self.dE * x)) < 1e-6) | (x < -178)]
            if bad_x.numel() > 0:                
                print('problematic x samples', bad_x[:10])
                print('exp max', torch.exp(-self.dE * bad_x).max().item())

        return x / (1.000 - torch.exp(-d * x) + 1e-8)
    
    def forward(self, state: torch.Tensor, delays: torch.Tensor, noise_in: torch.Tensor, noise_out: torch.Tensor, sc: torch.Tensor):
        """ 
            state: (N, 6, B)
            delays: (N, D, B)
            noise_in: (N, hidden_steps, input_size, B)
            noise_out: (N, B)
            sc: (B, N, N)
        """
        params = self.params
        haemo_steps = params.dt_ratio
        
        E = state[:, 0:1, :]  # (N, 1, B)
        I = state[:, 1:2, :]
        s = state[:, 2:3, :]
        f = state[:, 3:4, :]
        v = state[:, 4:5, :]
        q = state[:, 5:6, :]

        for t in range(self.hidden_size):
            E_noise = noise_in[:, t, 0:1, :] # (node_size, 1, batch_size)
            I_noise = noise_in[:, t, 1:2, :] # (node_size, 1, batch_size)

            # I_E calculations: delay-based vs. instaneous (E)
            if params.use_delay_based:
                write_index = t % self.delays_max
                delay_indices = (write_index - self.delay_steps) % self.delays_max  # (1, N, N)
                delay_indices = delay_indices.expand(self.batch_size, -1, -1) # (B, N, N)
                E_buffer = delays.permute(2, 1, 0) # (B, delays_max, N)
                E_delayed = E_buffer.gather(dim=1, index=delay_indices) # (B, N, N)
                connectivity = (sc * E_delayed).sum(-1).permute(1, 0).unsqueeze(1) # (N, 1, B)
            else:
                # Instantaneous FastDMF style
                E_b = E.permute(2, 0, 1) # (B, N, 1)
                weighted = torch.bmm(sc, E_b).squeeze(-1)             # (B, N)
                connectivity = weighted.permute(1,0).unsqueeze(1)     # (N, 1, B)

            # Inhibition variant
            if params.inhibitory_gain_scalar:
                inh_term = self.g_IE * I
            else:
                row_sum = sc.sum(-1) # (B, N)
                Ji  = 1.0 + 0.75 * self.g * row_sum        # (B, N)
                # Plotter.plot_vector(Ji[0, :].detach().cpu().numpy(), "Ji (batch = 0)")
                inh_term = Ji.permute(1, 0).unsqueeze(1) * I        # (N, 1, B)

            I_E = torch.relu(self.W_E * self.I_0 + self.g_EE * self.JN * E + self.g * self.JN * connectivity - inh_term)
            I_I = torch.relu(self.W_I * self.I_0 + self.JN * E - I)

            R_E = self.firing_rate(self.aE, self.bE, self.dE, I_E)
            R_I = self.firing_rate(self.aI, self.bI, self.dI, I_I)

            # if params.verbose and t % 10 == 0:
            #     print(f"I_E [{I_E.min().item():8.3f} - {I_E.max().item():8.3f}]     {I_E.mean().item():8.3f}\n")
            #     print(f"R_E [{R_E.min().item():8.3f} - {R_E.max().item():8.3f}]     {R_E.mean().item():8.3f}")

            dE = -E / self.tau_E + (1 - E) * self.gamma_E * R_E
            dI = -I / self.tau_I + self.gamma_I * R_I

            E = torch.relu(E + self.dt * dE + self.sigma_E * E_noise * torch.sqrt(self.dt))
            I = torch.relu(I + self.dt * dI + self.sigma_I * I_noise * torch.sqrt(self.dt))

            # Update delay buffer
            write_index = t % self.delays_max
            delays[:, write_index, :] = E.squeeze(1).detach()
            # delays = torch.cat([E, delays[:, :-1, :]], dim=1)

            if (t % haemo_steps) == 0:
                # Convert dt to seconds for haemodynamics
                ds = R_E - s * self.itau_s_h - (f - 1) * self.itau_f_h
                df = s
                dv = (f - v ** self.ialpha_h) * self.itau_0_h
                dq = (f * (1 - (1 - self.rho_h) ** (1 / f)) / self.rho_h
                    - q * (v ** (self.ialpha_h - 1))
                ) * self.itau_0_h

                s = s + ds * self.dtt_s
                f = f + df * self.dtt_s
                v = v + dv * self.dtt_s
                q = q + dq * self.dtt_s

                debug_snapshot(t * self.dt.item(),
                                rho=self.rho_h,
                                E=E, I=I, s=s, f=f, v=v, q=q,
                                I_E=I_E, R_E=R_E)    

                if params.verbose:
                    step_idx = t // haemo_steps
                    if step_idx % 50 == 0:
                        print(f"[{(t / 10):5.1f}ms]  E={E.mean():6.3f} I={I.mean():6.3f} I_E={I_E.mean():6.3f} I_I={I_I.mean():6.3f} R_E={R_E.mean():6.3f} R_I={R_I.mean():6.3f}", end="  |  ")
                        print(f"s={s.mean():6.3f} f={f.mean():6.3f} v={v.mean():6.3f} q={q.mean():6.3f}") # [HAEMO {step_idx}/{params.hemo_steps_per_tr}] 
                        
            if params.verbose: # and (t == 0 or t == params.hidden_steps - 1):
                if t == self.hidden_size - 1:    
                    rss = psutil.Process(os.getpid()).memory_info().rss / 2**20
                    print(f"RSS = {rss:,.0f} MB   GPU mem = {torch.cuda.memory_allocated()/2**20:,.0f} MB",
                        flush=True)

        BOLD = self.V0_h * (
            self.k1_h * (1 - q) + 
            self.k2_h * (1 - q / v) + 
            self.k3_h * (1 - v)
        ).squeeze(1)
        BOLD = BOLD + noise_out

        if params.verbose:
            print(f"[TR end] BOLD mean = {BOLD.mean():.3f}")

        new_state = torch.cat([E, I, s, f, v, q], dim=1)
        return new_state, BOLD, delays
