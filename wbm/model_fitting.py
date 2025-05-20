# Anish Kochhar, Imperial College London, March 2025
import time, gc
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# from tqdm import tqdm
from wbm.data_loader import BOLDDataLoader
from wbm.utils import DEVICE
from wbm.costs import Costs
from wbm.plotter import Plotter

class ModelFitting:
    def __init__(self, model, data_loader: BOLDDataLoader, num_epochs: int, lr: float, cost_function: Costs, \
                 smoothing_window: int = 1, finetune_steps: int = 5, finetune_batch: int = 32, batch_iters: int = None, log_state: bool = False, device = DEVICE):
        """
        Parameters:
            model: WholeBrainModel instance
EXPIRED     discriminator: DiscriminatorHook instance, containing all functionality to get classification loss (real vs. fake)
            data_loader: BOLDDataLoader instance providing sample_minibatch()
            num_epochs: Number of training epochs
            lr: Learning rate
            cost_function: compute() function for metrics comparision between simulated and empirical BOLD
            smoothing_window: size of moving-average window (1 = no smoothing)
            log_state: If True, logs the evolution of state variables over TR chunks
        """
        self.model = model
        self.loader = data_loader
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.cost_function = cost_function # Costs.compute
        self.smoothing_window = smoothing_window
        self.log_state = log_state

        self.finetune_steps = finetune_steps # per epoch
        self.finetune_batch = finetune_batch
        self.batch_iters = batch_iters

        self.logs = { "losses": [], "fc_correlation": [], "rmse": [], "roi_correlation": [], "hidden_states": [], "adv_loss": [] }
        self.parameters_history = { name: [] for name in ["g", "g_EE", "g_EI", "g_IE"] }
        self.device = device


    def smooth(self, bold: torch.Tensor):
        """ Applies moving average along time dimension (dim = 1) """
        if self.smoothing_window <= 1:
            return bold

        N, T, B = bold.shape
        x = bold.permute(0, 2, 1).reshape(-1, 1, T) # (N * B, T)
        
        # Asymmetric pad
        left_pad = (self.smoothing_window - 1) // 2
        right_pad = self.smoothing_window - 1 - left_pad
        x = F.pad(x, (left_pad, right_pad), mode='replicate')
        smoothed = F.avg_pool1d(x, kernel_size=self.smoothing_window, stride=1)
        smoothed = smoothed.reshape(N, B, T).permute(0, 2, 1)
        return smoothed

    def compute_fc(self, matrix: torch.Tensor):
        """ Builds the FC matrix  """
        zero_centered = matrix - matrix.mean(dim=1, keepdim=True)
        covariance = zero_centered @ zero_centered.T
        std = torch.sqrt(torch.diag(covariance)).unsqueeze(0)
        return (covariance / (std.T * std + 1e-8)).detach().cpu().numpy()

    def train(self, delays_max: int = 500, batch_size: int = 20):
        """
        Train the model over multiple minibatches, iterating over TR chunks for each sample

        Parameters:
            delays_max: Maximum delay stored for residual connections
            batch_size: Minibatch size
        """
        torch.autograd.set_detect_anomaly(True)

        num_batches = self.loader.batched_dataset_length(batch_size)  # Minibatches per epoch
        num_batches = min(num_batches, self.batch_iters) if self.batch_iters else num_batches
        delays = torch.zeros(self.model.node_size, delays_max, batch_size, device=self.device)

        for epoch in range(1, self.num_epochs + 1):
            batch_losses = []
            batch_fc_corrs = []
            batch_roi_corrs = []
            batch_rmses = []
            batch_adv_losses = []
            epoch_state_log = []
            # batch_iter = tqdm(range(num_batches), desc=f"Epochs [{epoch}/{self.num_epochs}]", unit="batch", leave=False)

            for batch_index in range(num_batches):
                # Initial state
                state = self.model.generate_initial_states().to(self.device)
                delays.zero_()

                self.optimizer.zero_grad()
            
                empirical_bold, normalised_sc, sampled = self.loader.sample_minibatch(batch_size)

                num_TRs = empirical_bold.shape[1]   # chunk_length = 50

                simulated_bold_chunks = []
                # noise_in shape: (node_size, hidden_size, batch_size, input_size) with input_size = 6
                noise_in = torch.empty(self.model.node_size, self.model.hidden_size, self.model.input_size, batch_size, device=self.device)
                # noise_out shape: (node_size, batch_size)
                noise_out = torch.empty(self.model.node_size, batch_size, device=self.device)

                print(f"[Trainer] g = {self.model.g.item():.4f}")

                for tr_index in range(num_TRs):
                    print(f"[TR {tr_index + 1}/{num_TRs}]")
                    # noise_in = torch.zeros_like(noise_in)
                    noise_in.normal_()

                    noise_out = torch.zeros_like(noise_out)
                    # noise_out.normal_() 

                    # external_current = torch.zeros(self.model.node_size, self.model.hidden_size, batch_size, device=self.device)

                    # state, bold_chunk, delays = self.model(state, external_current, noise_in, noise_out, delays, normalised_sc)
                    state, bold_chunk, delays = self.model(state, delays, noise_in, noise_out, normalised_sc)
                    simulated_bold_chunks.append(bold_chunk)

                    if self.log_state:
                        state_means = state.mean(dim=(0, 2)).detach().cpu().numpy()
                        # if tr_index % 10 == 0:
                        E_mean, I_mean, s_mean, f_mean, v_mean, q_mean = state_means
                        print(f"TR {tr_index:02d} | E={E_mean:.4f}  I={I_mean:.4f}  s={s_mean:.4f}  f={f_mean:.4f}  v={v_mean:.4f}  q={q_mean:.4f}" )
                        epoch_state_log.append(state_means)
                        
                    # if tr_index % 10 == 0:
                    #     print(f"SNR {((bold_chunk - noise_out).std()/noise_out.std()).item():.1f}",
                    #     f"|corr(E)| {torch.corrcoef(state[:,0,:]).abs().mean():.2f}",
                    #     f"|corr(BOLD)| {torch.corrcoef(bold_chunk).abs().mean():.2f}")


                # Stack TR chunks to form a time series: (node_size, num_TRs, batch_size)
                simulated_bold_epoch = torch.stack(simulated_bold_chunks, dim=1)
                smoothed_simulated_bold_epoch = self.smooth(simulated_bold_epoch)

                # Compare distribution of simulated vs empirical BOLD at final TR across all nodes and batches
                sim_final = smoothed_simulated_bold_epoch[:, -1, :].detach().cpu().numpy().flatten()
                emp_final = empirical_bold[:, -1, :].detach().cpu().numpy().flatten()
                print("Simulated BOLD (final TR) - mean: {:.4f}, std: {:.4f}".format(sim_final.mean(), sim_final.std()))
                print("Empirical BOLD (final TR) - mean: {:.4f}, std: {:.4f}".format(emp_final.mean(), emp_final.std()))

                # Compute cost
                metrics = self.cost_function.compute(smoothed_simulated_bold_epoch, empirical_bold)
                loss = metrics["loss"]

                print("MEMORY SUMMARY", torch.cuda.memory_summary())
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                print(f"Loss = {loss.item():.4f}")
                print(f"Backward {(time.perf_counter()-start):.2f} s")
                
                # Print gradient norms for each parameter
                grad_info = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_info.append(f"{name}: {grad_norm:.2e}")
                    else:
                        grad_info.append(f"{name}: None")
                print("Grad norms | " + ", ".join(grad_info))

                for n, p in self.model.named_parameters():
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print('Bad grad in', n)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                batch_losses.append(loss.item())
                batch_fc_corrs.append(metrics["average_fc_correlation"].item())
                batch_roi_corrs.append(metrics["average_rois_correlation"])
                batch_rmses.append(metrics["rmse"])
                # batch_adv_losses.append(total_adversarial_loss)

                # batch_iter.set_postfix(
                #     loss=f"{loss.item():.4f}",
                #     # adv=f"{total_adversarial_loss:.4f}",
                #     rmse=f"{metrics['rmse']:.4f}",
                #     fc_corr=f"{metrics['average_fc_correlation'].item():.4f}"
                # )

                delays = delays.detach()
                state = state.detach()

                # Plot FC matrix heatmaps for final epoch for batch 0
                simulated_fc = self.compute_fc(smoothed_simulated_bold_epoch[:, :, 0])
                empirical_fc = self.compute_fc(empirical_bold[:, :, 0])
                sampled_id = sampled[0]
                
                Plotter.plot_functional_connectivity_heatmaps(simulated_fc, empirical_fc, sampled_id)

                Plotter.plot_node_comparison(
                    empirical_bold[:, :, 0].unsqueeze(-1),
                    smoothed_simulated_bold_epoch[:, :, 0].unsqueeze(-1),
                    node_indices=list(np.random.choice(range(self.model.node_size), size=10, replace=False))
                )

                del simulated_bold_chunks[:], simulated_bold_epoch, smoothed_simulated_bold_epoch
                gc.collect()

            for parameter_name in self.parameters_history:
                self.parameters_history[parameter_name].append(getattr(self.model, parameter_name).item())
            
            self.logs["losses"].append(np.mean(batch_losses))
            self.logs["fc_correlation"].append(np.mean(batch_fc_corrs))
            self.logs["roi_correlation"].append(np.mean(batch_roi_corrs))
            self.logs["rmse"].append(np.mean(batch_rmses))
            self.logs["adv_loss"].append(np.mean(batch_adv_losses))

            if self.log_state:
                self.logs["hidden_states"].append(np.stack(epoch_state_log, axis=0))

            print(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Loss: {self.logs['losses'][-1]:.4f} | "
                f"RMSE: {self.logs['rmse'][-1]:.4f} | "
                f"ROI Corr: {self.logs['roi_correlation'][-1]:.4f} | "
                f"FC Corr: {self.logs['fc_correlation'][-1]:.4f}"
            )

            # Plot FC matrix heatmaps for final epoch for batch 0
            simulated_fc = self.compute_fc(smoothed_simulated_bold_epoch[:, :, 0])
            empirical_fc = self.compute_fc(empirical_bold[:, :, 0])
            sampled_id = sampled[0]
            
            Plotter.plot_functional_connectivity_heatmaps(simulated_fc, empirical_fc, sampled_id)

            Plotter.plot_node_comparison(
                empirical_bold[:, :, 0].unsqueeze(-1),
                smoothed_simulated_bold_epoch[:, :, 0].unsqueeze(-1),
                node_indices=list(np.random.choice(range(self.model.node_size), size=6, replace=False))
            )

            for name, param in self.model.named_parameters():
                print(name, param.item())


        # Final epoch visualisations
        epochs = list(range(1, self.num_epochs + 1))
        Plotter.plot_loss_curve(epochs, self.logs["losses"])
        Plotter.plot_fc_correlation_curve(epochs, self.logs["fc_correlation"])
        Plotter.plot_roi_correlation_curve(epochs, self.logs["roi_correlation"])
        Plotter.plot_rmse_curve(epochs, self.logs["rmse"])

        if self.log_state:
            Plotter.plot_hidden_states(self.logs["hidden_states"])

        Plotter.plot_coupling_parameters(self.parameters_history)

        # Plot FC matrix heatmaps for final epoch for batch 0
        simulated_fc = self.compute_fc(smoothed_simulated_bold_epoch[:, :, 0])
        empirical_fc = self.compute_fc(empirical_bold[:, :, 0])
        sampled_id = sampled[0]
        
        Plotter.plot_functional_connectivity_heatmaps(simulated_fc, empirical_fc, sampled_id)

        Plotter.plot_node_comparison(
            empirical_bold[:, :, 0].unsqueeze(-1),
            smoothed_simulated_bold_epoch[:, :, 0].unsqueeze(-1),
            node_indices=list(np.random.choice(range(self.model.node_size), size=6, replace=False))
        )

        for name, param in self.model.named_parameters():
            print(name, param.item())
