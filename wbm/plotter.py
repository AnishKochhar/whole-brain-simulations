## General purpose plotting functions

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Plotter:
    save_figures = True
    save_figpath = "wbm_plots/"

    @staticmethod
    def _autofill(x):       # Accepts torch / numpy / list
        if isinstance(x, torch.Tensor):
            return x.cpu().float().numpy()
        return np.asarray(x)

    @staticmethod
    def _get_save_path(basename, ext=".png"):
        """
        Returns a non-overwriting file path for saving figures.
        Uses Plotter.save_figpath as directory if set, else current directory.
        """
        directory = Plotter.save_figpath or "."
        os.makedirs(directory, exist_ok=True)
        idx = 0
        while True:
            fname = f"{basename}{f'_{idx}' if idx > 0 else ''}{ext}"
            fpath = os.path.join(directory, fname)
            if not os.path.exists(fpath):
                return fpath
            idx += 1

    @staticmethod
    def plot_loss_curve(epoch_indices, loss_values):
        plt.figure()
        plt.plot(epoch_indices, loss_values, marker='o')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if Plotter.save_figures:
            fpath = Plotter._get_save_path("loss_curve")
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    @staticmethod
    def plot_rmse_curve(epoch_indices, rmse_values):
        plt.figure()
        plt.plot(epoch_indices, rmse_values, marker='o')
        plt.title("RMSE over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        if Plotter.save_figures:
            fpath = Plotter._get_save_path("rmse_curve")
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    @staticmethod
    def plot_roi_correlation_curve(epoch_indices, roi_corr_values):
        plt.figure()
        plt.plot(epoch_indices, roi_corr_values, marker='o')
        plt.title("Average ROI Correlation over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Average ROI Pearson r")
        if Plotter.save_figures:
            fpath = Plotter._get_save_path("roi_correlation_curve")
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    @staticmethod
    def plot_fc_correlation_curve(epoch_indices, fc_corr_values):
        plt.figure()
        plt.plot(epoch_indices, fc_corr_values, marker='o')
        plt.title("Average FC Correlation over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Average FC Pearson r")
        if Plotter.save_figures:
            fpath = Plotter._get_save_path("fc_correlation_curve")
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    @staticmethod
    def plot_functional_connectivity_heatmaps(simulated_fc: np.ndarray, empirical_fc: np.ndarray, subject: int = None):
        """
            Plots both simulated and empirical Functional Connectivity (heatmap) on horizontal axis
            sim_fc, emp_fc: np.ndarray
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if subject: fig.suptitle(f"Subject {subject}")
        sns.heatmap(simulated_fc, vmin=-1, vmax=1, cmap='coolwarm', ax=axes[0])
        axes[0].set_title("Simulated FC")
        sns.heatmap(empirical_fc, vmin=-1, vmax=1, cmap='coolwarm', ax=axes[1])
        axes[1].set_title("Empirical FC")
        plt.tight_layout()
        if Plotter.save_figures:
            base = f"fc_heatmaps_subject_{subject}" if subject is not None else "fc_heatmaps"
            fpath = Plotter._get_save_path(base)
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

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
            if Plotter.save_figures:
                base = f"time_series_batch_{batch_idx}"
                fpath = Plotter._get_save_path(base)
                plt.savefig(fpath)
            else:
                plt.show()
            plt.close('all')

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
            if Plotter.save_figures:
                base = f"hidden_state_{state_names[dim]}"
                fpath = Plotter._get_save_path(base)
                plt.savefig(fpath)
            else:
                plt.show()
            plt.close('all')


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
        if Plotter.save_figures:
            fpath = Plotter._get_save_path("coupling_parameters")
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')
        

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
        if Plotter.save_figures:
            fpath = Plotter._get_save_path("node_comparison")
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    @staticmethod
    def plot_matrix(matrix: np.ndarray, title = "Laplacian Heatmap", cmap = 'viridis'):
        m = Plotter._autofill(matrix)
        plt.figure(figsize=(6, 5))
        sns.heatmap(m, cmap=cmap)
        plt.title(title)
        plt.xlabel("Node")
        plt.ylabel("Node")
        if Plotter.save_figures:
            base = title.replace(" ", "_").lower()
            fpath = Plotter._get_save_path(base)
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    @staticmethod
    def plot_distance_matrix(subject_index: int, distance_matrix: np.ndarray):
        plt.figure(figsize=(6, 5))
        sns.heatmap(distance_matrix, cmap='magma')
        plt.title(f"Distance Matrix (Subject {subject_index})")
        plt.xlabel("Node")
        plt.ylabel("Node")
        if Plotter.save_figures:
            base = f"distance_matrix_subject_{subject_index}"
            fpath = Plotter._get_save_path(base)
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    def plot_vector(vector, title="Vector Plot", xlabel="Source Node", ylabel="Value"):
        v = Plotter._autofill(vector).flatten()
        plt.figure(figsize=(6,2))
        plt.plot(v, marker='o', lw=1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        if Plotter.save_figures:
            base = title.replace(" ", "_").lower()
            fpath = Plotter._get_save_path(base)
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    def plot_hist(array, bins=50, title=""):
        plt.figure()
        plt.hist(array.flatten(), bins=bins)
        plt.title(title)
        plt.tight_layout()
        if Plotter.save_figures:
            base = title.replace(" ", "_").lower() if title else "hist"
            fpath = Plotter._get_save_path(base)
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')

    @staticmethod
    def hist_triplet(arr1, arr2, arr3, step, titles=("I_E", "aE*I_E-bE", "r_E"), bins=40):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        for k, (data, title) in enumerate(zip((arr1, arr2, arr3), titles)):
            ax[k].hist(data.flatten(), bins=bins, color="tab:blue")
            ax[k].set_title(f"{title}  (step {step})")
        plt.tight_layout()
        if Plotter.save_figures:
            base = f"hist_triplet_step_{step}"
            fpath = Plotter._get_save_path(base)
            plt.savefig(fpath)
        else:
            plt.show()
        plt.close('all')
