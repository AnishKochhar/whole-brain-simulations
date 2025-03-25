import numpy as np
import torch
import matplotlib.pyplot as plt


class Costs:
    
    def cost_distance(self, sim, emp):
        """
        Compute RMSE between simulated and empirical signals
        """
        return torch.sqrt(torch.mean((sim - emp) ** 2))


    def compare_bold(self, simulated_bold, empirical_bold, plot=True, verbose=True):
        """
        Compare two BOLD time series and calcuate Pearson correlation between FC matrices

        Parameters:
            simulated_bold: torch.Tensor shape (N, T)
            empirical_bold: torch.Tensor shape (N, T)

        Returns:
            correlation_loss: torch scalar, Pearson's correlation loss between FC matrices
                calculated as -log(0.5 + 0.5 * global_corr)
        """
        if not isinstance(simulated_bold, torch.Tensor):
            simulated_bold = torch.tensor(simulated_bold, dtype=torch.float32)
        if not isinstance(empirical_bold, torch.Tensor):
            empirical_bold = torch.tensor(empirical_bold, dtype=torch.float32)
        
        assert simulated_bold.shape == empirical_bold.shape, f"Simulated and Empirical BOLD time series must have the same dimensions. Found EMP: {empirical_bold.shape}, SIM: {simulated_bold.shape}"
        N, T = simulated_bold.shape

        rmse = torch.sqrt(torch.mean((simulated_bold - empirical_bold) ** 2))

        # Compute Pearon's correlation between per node
        rois_correlation = []
        for node in range(N):
            sim = simulated_bold[node, :]
            emp = empirical_bold[node, :]

            # Zero mean
            s_centered = sim - torch.mean(sim)
            e_centered = emp - torch.mean(emp)
            corr = torch.dot(s_centered, e_centered) / (torch.sqrt(torch.sum(s_centered**2)) * torch.sqrt(torch.sum(e_centered**2)) + 1e-8)
            rois_correlation.append(corr)
            
        rois_correlation = torch.stack(rois_correlation)
        average_rois_correlation = torch.mean(rois_correlation)
        
        # Compute global FC matrices using torch operations.
        sim_n = simulated_bold - torch.mean(simulated_bold, dim=1, keepdim=True)
        emp_n = empirical_bold - torch.mean(empirical_bold, dim=1, keepdim=True)
        cov_sim = sim_n @ sim_n.t()  # (N, N)
        cov_emp = emp_n @ emp_n.t()  # (N, N)
        std_sim = torch.sqrt(torch.diag(cov_sim) + 1e-8)
        std_emp = torch.sqrt(torch.diag(cov_emp) + 1e-8)
        FC_sim = cov_sim / (std_sim.unsqueeze(1) * std_sim.unsqueeze(0) + 1e-8)
        FC_emp = cov_emp / (std_emp.unsqueeze(1) * std_emp.unsqueeze(0) + 1e-8)
        
        # Extract lower triangular parts (excluding the diagonal).
        mask = torch.tril(torch.ones_like(FC_sim), diagonal=-1).bool()
        sim_vec = FC_sim[mask]
        emp_vec = FC_emp[mask]
        sim_vec = sim_vec - torch.mean(sim_vec)
        emp_vec = emp_vec - torch.mean(emp_vec)
        global_corr = torch.sum(sim_vec * emp_vec) / (torch.sqrt(torch.sum(sim_vec**2)) * torch.sqrt(torch.sum(emp_vec**2)) + 1e-8)
        

        if plot:
            FC_sim_np = FC_sim.detach().cpu().numpy()
            FC_emp_np = FC_emp.detach().cpu().numpy()

            ## Visualisations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

            # Plot simulated FC matrix
            im1 = ax1.imshow(FC_sim_np, vmin=-1, vmax=1, cmap='coolwarm')
            ax1.set_title("Simulated FC")
            plt.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(FC_emp_np, vmin=-1, vmax=1, cmap='coolwarm')
            ax2.set_title("Empirical FC")
            plt.colorbar(im2, ax=ax2)

            plt.tight_layout()
            plt.show()

        if verbose:

            print(f"RMSE between BOLD time series: {rmse:.4f}")
            print(f"Average per-ROI Pearson correlation: {average_rois_correlation:.4f}")
            print(f"FC Pearson's correlation: {global_corr:.4f}")

        correlation_loss = -torch.log(0.5 + 0.5 * global_corr + 1e-8)
        return correlation_loss
        

    def compare_bold_np(self, simulated_bold, empirical_bold, plot=True, verbose=True):
        """
        Compare two BOLD time series and calcuate Pearson correlation between FC matrices

        Parameters:
            simulated_bold: torch.Tensor shape (N, T)
            empirical_bold: torch.Tensor shape (N, T)

        Returns:
            correlation_loss: torch scalar, Pearson's correlation loss between FC matrices
                calculated as -log(0.5 + 0.5 * global_corr)
        """
        if isinstance(simulated_bold, torch.Tensor):
            simulated_bold = simulated_bold.detach().cpu().numpy()
        if isinstance(empirical_bold, torch.Tensor):
            empirical_bold = empirical_bold.detach().cpu().numpy()
        
        assert simulated_bold.shape == empirical_bold.shape, f"Simulated and Empirical BOLD time series must have the same dimensions. Found EMP: {empirical_bold.shape}, SIM: {simulated_bold.shape}"
        N, T = simulated_bold.shape

        rmse = np.sqrt(np.mean((simulated_bold - empirical_bold) ** 2))

        # Compute Pearon's correlation between per node
        rois_correlation = []
        for node in range(N):
            sim = simulated_bold[node, :]
            emp = empirical_bold[node, :]

            # Zero mean
            sim = sim - np.mean(sim)
            emp = emp - np.mean(emp)

            correlation = np.corrcoef(sim, emp)[0, 1]
            rois_correlation.append(correlation)
        average_rois_correlation = np.mean(rois_correlation)

        # Global FC matrices (correlation across time for each pair of ROIs)
        FC_sim = np.corrcoef(simulated_bold)
        FC_emp = np.corrcoef(empirical_bold)

        # Extract lower triangles (excluding diagonal)
        triu_idx = np.triu_indices(N, k=1)
        FC_simulated_vec = FC_sim[triu_idx]
        FC_empirical_vec = FC_emp[triu_idx]
        
        # Global FC similarity
        FC_simulated_vec = FC_simulated_vec - np.mean(FC_simulated_vec)
        FC_empirical_vec = FC_empirical_vec - np.mean(FC_empirical_vec)
        global_corr = np.sum(FC_simulated_vec * FC_empirical_vec) / (np.sqrt(np.sum(FC_simulated_vec**2)) * np.sqrt(np.sum(FC_empirical_vec**2)) + 1e-8)
        
        if plot:
            ## Visualisations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

            # Plot simulated FC matrix
            cax1 = ax1.imshow(FC_sim, vmin=-1, vmax=1, cmap='coolwarm')
            ax1.set_title("Simulated FC")
            fig.colorbar(cax1, ax=ax1)

            cax2 = ax2.imshow(FC_emp, vmin=-1, vmax=1, cmap='coolwarm')
            ax2.set_title("Empirical FC")
            fig.colorbar(cax2, ax=ax2)

            plt.tight_layout()
            plt.show()

        if verbose:

            print(f"RMSE between BOLD time series: {rmse:.4f}")
            print(f"Average per-ROI Pearson correlation: {average_rois_correlation:.4f}")
            print(f"FC Pearson's correlation: {global_corr:.4f}")

        correlation_loss = -np.log(0.50 + 0.50 * global_corr)
        return correlation_loss