## Build PyTorch-Geometric graphs from BOLD chunks + SC matrices

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

class GraphBuilder:
    """
    Converts (bold, sc) pairs into PyG Data objects:
        adjacenct = structural connectivity (N, N)
        x = Ledoit-Wolf FC column featrures (N, N) 
            (+ optional PCA of raw BOLD) (N, p_dim)
        y = label (1 = real, 0 = synthetic)
    """
    def __init__(self, node_dim: int = 100, pca_dim: int = 8, use_pca: bool = True, device: str = "cpu"):
        self.node_dim = node_dim
        self.pca_dim = pca_dim
        self.use_pca = use_pca
        self.device = device
        self._pca = PCA(n_components=pca_dim) if use_pca else None

    @staticmethod
    def _ledoit_wolf_corr(bold_np: np.ndarray):
        """ 
        Ledoit-Wold shrunk covariance, used as correlation column features 
        Parameters:
            bold_np: shape (N, T)
        """
        lw = LedoitWolf(store_precision=False).fit(bold_np.T)
        covariance = lw.covariance_
        std = np.sqrt(np.diag(covariance) + 1e-12)
        correlation = covariance / np.outer(std, std) # shape (N, N)
        return correlation 
    
    def _node_features(self, bold_np: np.ndarray):
        """
        Feature 1: column of shrunk FC (N, N)
        Feature 2: p PCA components (N, p)
        """
        fc = self._ledoit_wolf_corr(bold_np)
        feats = [fc.T] # x_i columns
        if self.use_pca:
            pca_feats = self._pca.fit_transform(bold_np)
            feats.append(pca_feats)
        feat_mat = np.concatenate(feats, axis=1)      # (N, d)
        return torch.tensor(feat_mat, dtype=torch.float32, device=self.device)
    
    def build_graph(self, bold: np.ndarray, sc: np.ndarray, label: int):
        """
        Parameters:
            bold: ndarray, BOLD chunk with shape (N, T)
            sc: ndarray, SC matrix ([0, 1] normalised) shape (N, N)
            label: int, 1 = real, 0 = simualted
        """
        A = torch.tensor(sc, dtype=torch.float32)
        X = self._node_features(bold)
        edge_index, edge_weight = dense_to_sparse(A)
        data = Data(x = X, edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor([label], dtype=torch.float32))
        return data
