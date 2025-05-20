import torch
import numpy as np
from wbm.utils import DEVICE

class Costs:

    @staticmethod
    def ledoit_wolf_shrinkage_torch(X: torch.Tensor, shrinkage_value: float = 0.1) -> torch.Tensor:
        """
        Compute Ledoit-Wolf shrunk covariance matrix in PyTorch (with manual shrinkage)
        
        Parameters:
            X: torch.Tensor of shape (T, N) or (B, T, N)
            shrinkage_value: float in [0, 1], amount of shrinkage

        Returns:
            shrunk_cov: torch.Tensor of shape (N, N) or (B, N, N)
        """
        X = X.to(DEVICE)
        if X.dim() == 2:
            X = X.unsqueeze(0)  # (1, T, N)
        B, T, N = X.shape

        X_mean = X.mean(dim=1, keepdim=True)
        X_centered = X - X_mean  # (B, T, N)
        empirical_cov = torch.matmul(X_centered.transpose(1, 2), X_centered) / (T - 1)  # (B, N, N)

        # Target: identity scaled by average variance
        avg_var = torch.mean(torch.diagonal(empirical_cov, dim1=1, dim2=2), dim=1)  # (B,)
        target = torch.stack([torch.eye(N, device=X.device) * avg_var[i] for i in range(B)], dim=0)  # (B, N, N)

        shrunk_cov = (1 - shrinkage_value) * empirical_cov + shrinkage_value * target
        return shrunk_cov.squeeze(0) if shrunk_cov.shape[0] == 1 else shrunk_cov


    @staticmethod
    def compute(simulated_bold, empirical_bold):
        """
        Compare two BOLD time series and calcuate Pearson correlation between FC matrices

        Parameters:
            simulated_bold: torch.Tensor shape (N, T, B)
            empirical_bold: torch.Tensor shape (N, T, B)

        Returns:
            loss: torch scalar, Pearson's correlation loss between FC matrices
                calculated as -log(0.5 + 0.5 * global_corr)
            root mean squared error
            average node-wise Pearson correlation
            average functional connectivity Pearson correlation
        """
        if not isinstance(simulated_bold, torch.Tensor):
            simulated_bold = torch.tensor(simulated_bold, dtype=torch.float32)
        if not isinstance(empirical_bold, torch.Tensor):
            empirical_bold = torch.tensor(empirical_bold, dtype=torch.float32)
        
        assert simulated_bold.shape == empirical_bold.shape, f"Simulated and Empirical BOLD time series must have the same dimensions. Found EMP: {empirical_bold.shape}, SIM: {simulated_bold.shape}"
        # print(f"Simulated BOLD shape: ({simulated_bold.shape})")
        N, T, B = simulated_bold.shape

        rmse = torch.sqrt(torch.mean((simulated_bold - empirical_bold) ** 2))

        # Compute Pearon's correlation between per node
        rois_correlation = []
        for b in range(B):
            sim_batch = simulated_bold[:, :, b]
            emp_batch = empirical_bold[:, :, b]

            # Zero mean
            s_centered = sim_batch - torch.mean(sim_batch)
            e_centered = emp_batch - torch.mean(emp_batch)
            
            dot_product = (s_centered * e_centered).sum(dim=1)
            product = (s_centered.norm(dim=1) * e_centered.norm(dim=1) + 1e-8)
            rois_correlation.append((dot_product / product).mean().detach().item())

        average_rois_correlation = float(np.mean(rois_correlation))

        global_corrs = []
        
        for b in range(B):
            sim_b = simulated_bold[:, :, b].permute(1, 0) # (T, N)
            emp_b = empirical_bold[:, :, b].permute(1, 0) # (T, N)

            if torch.allclose(sim_b.std(dim=0), torch.zeros_like(sim_b[0]), atol=1e-5) or \
            torch.allclose(emp_b.std(dim=0), torch.zeros_like(emp_b[0]), atol=1e-5):
                print(f"[WARNING] Batch {b}: constant or near-zero signal in sim or emp BOLD.")
                global_corr_b = torch.tensor(0.0, device=sim_b.device, requires_grad=True)
                global_corrs.append(global_corr_b)
                continue


            # MARK: Ledoit-Wolf shrunk covariance estimation
            # cov_sim = Costs.ledoit_wolf_shrinkage_torch(sim_b, shrinkage_value=0.1)  # (N, N)
            # cov_emp = Costs.ledoit_wolf_shrinkage_torch(emp_b, shrinkage_value=0.1)
            # Compute global FC matrices
            sim_n = sim_b - torch.mean(sim_b, dim=1, keepdim=True)
            emp_n = emp_b - torch.mean(emp_b, dim=1, keepdim=True)
            cov_sim = sim_n @ sim_n.t()  # (N, N)
            cov_emp = emp_n @ emp_n.t()  # (N, N)
            std_sim = torch.sqrt(torch.diag(cov_sim) + 1e-8)
            std_emp = torch.sqrt(torch.diag(cov_emp) + 1e-8)
            FC_sim = cov_sim / (std_sim.unsqueeze(1) * std_sim.unsqueeze(0) + 1e-8)
            FC_emp = cov_emp / (std_emp.unsqueeze(1) * std_emp.unsqueeze(0) + 1e-8)
            
            # Extract lower triangular parts (excluding the diagonal)
            mask = torch.tril(torch.ones_like(FC_sim), diagonal=-1).bool()

            sim_vec = FC_sim[mask]
            emp_vec = FC_emp[mask]
            sim_vec = sim_vec - torch.mean(sim_vec)
            emp_vec = emp_vec - torch.mean(emp_vec)

            global_corr = torch.sum(sim_vec * emp_vec) / (torch.sqrt(torch.sum(sim_vec**2)) * torch.sqrt(torch.sum(emp_vec**2)) + 1e-8)


            # dot = torch.sum(sim_vec * emp_vec)
            # norm_sim = torch.sqrt(torch.sum(sim_vec ** 2)) #.clamp(min=1e-6)
            # norm_emp = torch.sqrt(torch.sum(emp_vec ** 2)) #.clamp(min=1e-6)
            # l2_product = norm_sim * norm_emp

            # global_corr_b = dot / l2_product

            # # Skip NaNs or Infs
            # if not torch.isfinite(global_corr_b):
            #     print(f"[WARNING] NaN or Inf in global_corr_b at batch {b}")
            #     global_corr_b = torch.tensor(0.0, device=sim_vec.device, requires_grad=True)

            # global_corrs.append(global_corr_b)

        # global_corr = torch.mean(torch.stack(global_corrs)).clamp(min=1e-8)
        # global_corr = torch.nan_to_num(global_corr, nan=0.0, posinf=0.0, neginf=0.0) # NAN -> 0
        # global_corr = torch.clamp(global_corr, min=-0.999, max=0.999) # Clamp to (-1, 1)

        correlation_loss = -torch.log(0.5 + 0.5 * global_corr + 1e-8)

        print(f"RMSE between BOLD time series: {rmse:.4f}")
        print(f"Average per-ROI Pearson correlation: {average_rois_correlation:.4f}")
        print(f"FC Pearson's correlation: {global_corr:.4f}")

        return {
            "loss": correlation_loss,
            "rmse": rmse.detach().cpu().numpy(),
            "average_rois_correlation": average_rois_correlation,
            "average_fc_correlation": global_corr.detach().cpu().numpy()
        }
        