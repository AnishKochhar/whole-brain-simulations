# Anish Kochhar, Imperial College London, March 2025

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from costs import *

class Trainer:
    """
    Training loop:
        - Divides empirical BOLD into batches (at the TR scale)
        - Run DMF model in batches, and run haemodynamic forward model to generate simulated BOLD
        - Compute loss comparing FC
    """
    def __init__(self, model: nn.Module, balloon: nn.Module, costs: Costs, BOLD: np.ndarray, lr: float = 0.001, epochs: int = 20, 
                 batch_time: int = 50, plot: bool = True, verbose: bool = True):
        self.model = model
        self.balloon = balloon
        self.costs = costs
        self.BOLD = BOLD # (N, T_emp)
        self.lr = lr
        self.epochs = epochs
        self.batch_time = batch_time
        self.plot = plot 
        self.verbose = verbose

        # Optimizer over model and foward modules
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.balloon.parameters()), lr=self.lr)

    def train(self, x0: torch.Tensor = None, delays_E: torch.Tensor = None, delays_max = 500, delay_lb = 0.5, delay_ub = 2):
        if x0 is None:
            x0 = torch.zeros(self.model.node_size, self.model.state_size, dtype=torch.float32)
        if delays_E is None:
            delays_E = torch.FloatTensor(self.model.node_size, delays_max).uniform_(delay_lb, delay_ub)

        N, T = self.BOLD.shape
        assert N == self.model.node_size, f"Empirical BOLD node size {N} does not match model node size {self.model.node_size}"
        assert T >= self.batch_time, "Batch time steps exceeds empirical BOLD length"


        loss_history = []
        batch_size = 1
        num_batches = int(T // batch_size)
        print(f"[DEBUG] Running {num_batches} batches...")

        for epoch in tqdm(range(self.epochs), desc='Training epochs'):
            self.optimizer.zero_grad()

            x_batch = x0.clone()
            delays_E_batch = delays_E.clone()
            output_state = None # State of haemodynamic model

            bold_output = []
            # DEBUG
            dmf_history_epoch = []    # will store DMF state averages per time step for each batch
            balloon_history_epoch = []  # will store Balloon state (averaged over nodes) per batch


            for b in range(num_batches):
                X, x_batch, delays_E_batch = self.model.forward(x_batch, delays_E_batch, batches=batch_size)
                # X: (batch_time, node_size, state_size) (15, 100, 2)
                dmf_mean = torch.mean(X, dim=1)  # shape: (batch_time, 2)
                dmf_history_epoch.append(dmf_mean.detach().cpu().numpy())

                BOLD_batch, output_state = self.balloon.forward(X, output_state) # (node_size, 1)
                balloon_mean = torch.mean(output_state, dim=0)  # shape: (4,)
                balloon_history_epoch.append(balloon_mean.detach().cpu().numpy())


                bold_output.append(BOLD_batch)
                print(f"[DEBUG] Batch {b+1}: BOLD mean = {torch.mean(BOLD_batch).item():.4f}")


            simulated_BOLD = torch.cat(bold_output, dim=1) # (node_size, num_batches)

            loss = self.costs.compare_bold(simulated_BOLD, self.BOLD, self.plot, self.verbose)
            loss.backward()
            self.optimizer.step()

            loss_history.append(loss.item())

            if self.verbose:
                print(f"Epoch {epoch + 1} / {self.epochs}, Loss: {loss.item():.6f}")

            dmf_history_epoch = np.concatenate(dmf_history_epoch, axis=0)  # shape: (num_batches*batch_time, 2)
            time_axis = np.arange(dmf_history_epoch.shape[0]) * self.model.dt
            plt.figure()
            plt.plot(time_axis, dmf_history_epoch[:, 0], label="E (DMF)")
            plt.plot(time_axis, dmf_history_epoch[:, 1], label="I (DMF)")
            plt.title(f"Epoch {epoch+1}: DMF state time series (averaged over nodes)")
            plt.xlabel("Time (s)")
            plt.ylabel("Activity")
            plt.legend()
            plt.show()

            balloon_history_epoch = np.array(balloon_history_epoch)
            batch_axis = np.arange(balloon_history_epoch.shape[0])
            plt.figure()
            plt.plot(batch_axis, balloon_history_epoch[:, 0], label="x")
            plt.plot(batch_axis, balloon_history_epoch[:, 1], label="f")
            plt.plot(batch_axis, balloon_history_epoch[:, 2], label="v")
            plt.plot(batch_axis, balloon_history_epoch[:, 3], label="q")
            plt.title(f"Epoch {epoch+1}: Balloon state averages per batch")
            plt.xlabel("Batch index")
            plt.ylabel("State value")
            plt.legend()
            plt.show()



        return loss_history

    def test(self, x0: torch.Tensor = None, delays_E: torch.Tensor = None, delays_max = 500, delay_lb = 0.5, delay_ub = 2):
        if x0 is None:
            x0 = torch.zeros(self.model.node_size, self.model.state_size, dtype=torch.float32)
        if delays_E is None:
            delays_E = torch.FloatTensor(self.model.node_size, delays_max).uniform_(delay_lb, delay_ub)

        test_batches = int(self.BOLD.shape[1] // self.batch_time)
        print(f"[DEBUG] Running {test_batches} batches...")

        x_batch = x0.clone()
        delays_E_batch = delays_E.clone()

        bold_output = []
        for _ in range(test_batches):
            X, x_batch, delays_E_batch = self.model.forward(x_batch, delays_E_batch, batches=1)
            BOLD_batch = self.balloon.forward(X, batches=1)
            bold_output.append(BOLD_batch)
        simulated_BOLD = torch.cat(bold_output, dim=1) # (node_size, num_batches)
        if self.verbose:
            print(f"Test simulation: simulated BOLD shape: {simulated_BOLD.shape}")

        loss = self.costs.compare_bold(simulated_BOLD, self.BOLD, self.plot, self.verbose)
        
        return loss

