## General purpose plotting functions

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    @staticmethod
    def plot_loss_curve(epoch_indices, loss_values):
        plt.figure()
        plt.plot(epoch_indices, loss_values, marker='o')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    @staticmethod
    def plot_rmse_curve(epoch_indices, rmse_values):
        plt.figure()
        plt.plot(epoch_indices, rmse_values, marker='o')
        plt.title("RMSE over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.show()

    @staticmethod
    def plot_roi_correlation_curve(epoch_indices, roi_corr_values):
        plt.figure()
        plt.plot(epoch_indices, roi_corr_values, marker='o')
        plt.title("Average ROI Correlation over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Average ROI Pearson r")
        plt.show()

    @staticmethod
    def plot_fc_correlation_curve(epoch_indices, fc_corr_values):
        plt.figure()
        plt.plot(epoch_indices, fc_corr_values, marker='o')
        plt.title("Average FC Correlation over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Average FC Pearson r")
        plt.show()

    @staticmethod
    def plot_functional_connectivity_heatmaps(simulated_fc: np.ndarray, empirical_fc: np.ndarray):
        """
            Plots both simulated and empirical Functional Connectivity (heatmap) on horizontal axis
            sim_fc, emp_fc: np.ndarray
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(simulated_fc, vmin=-1, vmax=1, cmap='coolwarm', ax=axes[0])
        axes[0].set_title("Simulated FC")
        sns.heatmap(empirical_fc, vmin=-1, vmax=1, cmap='coolwarm', ax=axes[1])
        axes[1].set_title("Empirical FC")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_time_series(time_series: torch.Tensor, title: str, max_nodes: int = 6):
        """
            Plots batched BOLD time series on single plot:
            time_series: torch.Tensor, shape (N, T, B)
        """
        data = time_series.detach().cpu().numpy() if isinstance(time_series, torch.Tensor) else time_series
        N, T, B = data.shape
        for batch_idx in range(B):
            plt.figure(figsize=(10, 4))
            for node_idx in range(min(N, max_nodes)):
                plt.plot(np.arange(T), data[node_idx, :, batch_idx], label=f"Node {node_idx}")
            plt.title(f"{title} (Batch {batch_idx})")
            plt.xlabel("TR")
            plt.ylabel("BOLD signal")
            plt.legend()
            plt.show()

    @staticmethod
    def plot_hidden_states(hidden_state_logs: np.ndarray, state_names = ['E', 'I', 'x', 'f', 'v', 'q']):
        """
            hidden_state_logs: list of `epoch` elements, each (time_points, state_size = 6)
            Heatmaps each of six state variables (E, I, x, f, v, q) over TRs, where colour is the state value
        """
        logs = np.stack(hidden_state_logs, axis=0)
        num_epochs, T, state_size = logs.shape

        for dim in range(state_size):
            plt.figure(figsize=(6, 4))
            sns.heatmap(logs[:, :, dim], xticklabels=max(1, T//10), yticklabels=1, cmap="magma", cbar_kws={'label': state_names[dim]})
            plt.title(f"Evolution of hidden-state '{state_names[dim]}'")
            plt.xlabel("TR")
            plt.ylabel("Epoch")
            plt.tight_layout()
            plt.show()


    @staticmethod
    def plot_coupling_parameters(parameter_history):
        """
            parameter_history: dictionary of `epoch` elements each (time_points, parameter_size = 4)
            Plots each of the four core coupling parameters (g, g_EE, g_EI, g_IE)
        """
        plt.figure(figsize=(6, 4))
        for param_name, values in parameter_history.items():            
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, marker='o', label=param_name)
        plt.title("Coupling parameters over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Parameter Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
        

    @staticmethod
    def plot_node_comparison(empirical_bold: torch.Tensor, simulated_bold: torch.Tensor, node_indices=None):
        if node_indices is None:
            node_indices = list(range(min(6, empirical_bold.shape[0])))
        emp = empirical_bold.detach().cpu().numpy()
        sim = simulated_bold.detach().cpu().numpy()
        T = emp.shape[1]
        fig, axes = plt.subplots(len(node_indices), 1, figsize=(10, 2*len(node_indices)), sharex=True)
        for i, node in enumerate(node_indices):
            axes[i].plot(np.arange(T), emp[node, :, 0], label="Empirical")
            axes[i].plot(np.arange(T), sim[node, :, 0], label="Simulated")
            axes[i].set_ylabel(f"Node {node}")
            if i == 0:
                axes[i].legend()
        axes[-1].set_xlabel("TR")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_laplacian(subject_index: int, laplacian_matrix: np.ndarray, title = "Laplacian Heatmap (Subject {subject_index})"):
        plt.figure(figsize=(6, 5))
        sns.heatmap(laplacian_matrix, cmap='viridis')
        plt.title(title)
        plt.xlabel("Node")
        plt.ylabel("Node")
        plt.show()

    @staticmethod
    def plot_distance_matrix(subject_index: int, distance_matrix: np.ndarray):
        plt.figure(figsize=(6, 5))
        sns.heatmap(distance_matrix, cmap='magma')
        plt.title(f"Distance Matrix (Subject {subject_index})")
        plt.xlabel("Node")
        plt.ylabel("Node")
        plt.show()