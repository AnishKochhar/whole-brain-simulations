## Build PyTorch-Geometric graphs from BOLD chunks + SC matrices

import torch
from sklearn.decomposition import PCA
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

class GraphBuilder:
    """
    Converts (bold, sc) pairs into PyG Data objects:
        adjacenct = structural connectivity (N, N)
        x = Ledoit-Wolf FC column featrures (N, N) (+ optional PCA of raw BOLD) (N, p_dim)
        y = label (1 = real, 0 = synthetic)
    """
    def __init__(self, node_dim: int = 100, pca_dim: int = 8, use_pca: bool = True, device: str = "cuda"):
        self.node_dim = node_dim
        self.pca_dim = pca_dim
        self.use_pca = use_pca
        self.device = device
        self._pca = PCA(n_components=pca_dim) if use_pca else None

    def _ledoit_wolf_shrinkage_torch(self, X: torch.Tensor, shrinkage_value: float = 0.1):
        """
        Compute Ledoit-Wolf shrunk covariance matrix in PyTorch (with manual shrinkage)
        
        Parameters:
            X: torch.Tensor of shape (T, N) or (B, T, N)
            shrinkage_value: float in [0, 1], amount of shrinkage

        Returns:
            shrunk_cov: torch.Tensor of shape (N, N) or (B, N, N)
        """
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

    def build_graph(self, bold_chunk: torch.Tensor, sc_matrix: torch.Tensor, label: float = 1.0) -> Data:
        """
        Builds Data instance out of input BOLD time series and SC matrix
        Uses Ledoit Wolf shrinkage as functional connectivity matrix

        Parameters:
            bold_chunk: torch Tensor shape (N, T)
            sc: torch Tensor shape (N, N)
        
            Returns:

        """
        cov = self._ledoit_wolf_shrinkage_torch(bold_chunk.T)
        std = torch.sqrt(torch.diag(cov) + 1e-8)
        fc = cov / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-8)
        
        feats = [fc.t()]

        if self.use_pca:
            feats.append(self._pca.fit_transform(bold_chunk.T.detach().numpy()))
        
        x = torch.cat(feats, dim=1).to(self.device)
        edge_index, edge_weight = dense_to_sparse(sc_matrix)
        return Data(x = x.to(self.device), edge_index=edge_index.to(self.device), edge_weight=edge_weight.to(self.device), 
                    y=torch.tensor([label], dtype=torch.float32, device=self.device)).to(self.device)
