## VGAE encoder, unsupervise trainer, classifier head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.loader import DataLoader

import os
from pathlib import Path
from typing import List
from tqdm import tqdm


# Encoder
class _GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, latent):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv_mu = GCNConv(hidden, latent)
        self.conv_logvar = GCNConv(hidden, latent)

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv_mu(h, edge_index, edge_weight), self.conv_logvar(h, edge_index, edge_weight)
    

# VGAE
def build_vgae(in_dim, hidden=64, latent=32):
    encoder = _GCNEncoder(in_dim, hidden, latent)
    return VGAE(encoder)

# Classifier Head
class GraphClassifier(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return torch.sigmoid(self.mlp(z))
    
class DiscriminatorVGAE(nn.Module):
    """
    Wrapper that includes:
        VGAE encoder (frozen after pre-trian unless unfreeze = True)
        Graph-level MLP classifier
        helper exposing get_lost(real_graphs, simulated_graphs), returning a scalar loss
    """
    def __init__(self, encoder: VGAE, latent_dim: int = 32):
        super().__init__()
        self.encoder = encoder
        self.classifier = GraphClassifier(latent_dim)


    def encode(self, data: Data):
        """ Returs node-level latents (requires grad = False) """
        with torch.no_grad():
            z_nodes = self.encoder.encode(data.x, data.edge_index, data.edge_weight)
        return z_nodes
    
    def forward(self, data: Data):
        z = self.encode(data).mean(dim=0, keepdim=True) # (1, latent)
        confidence = self.classifier(z)
        return confidence


    # Training API
    @staticmethod
    def train_unsupervised(encoder: VGAE, dataset: List, epochs: int = 200, beta: float = 1e-3, 
                           batch_size: int = 16, lr: float = 1e-3, device: str = "cuda"):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        encoder = encoder.to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

        pbar = tqdm(range(epochs), "VGAE pre-train")

        for epoch in pbar:
            epoch_loss = 0
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                z = encoder.encode(data.x, data.edge_index, data.edge_weight)
                loss = ( encoder.recon_loss(z, data.edge_index, data.edge_weight)
                        + beta * encoder.kl_loss() )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            pbar.set_postfix(loss=f"{epoch_loss/len(loader):.4f}")
        
        # Freeze weights
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    @staticmethod
    def train_classifier(discriminator, dataset: List, epochs: int = 50, batch_size: int = 8, lr: float = 1e-3, device: str = "cuda"):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        discriminator = discriminator.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(discriminator.classifier.parameters(), lr=lr)

        pbar = tqdm(range(epochs), "Classifier train")
        losses = []
        for epoch in pbar:
            epoch_loss = 0.0
            for data in loader:
                data = data.to(device)
                confidence = discriminator(data)
                loss = criterion(confidence, data.y.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            losses.append(epoch_loss)
            pbar.set_postfix(loss=f"{epoch_loss:.4f}")
        return losses
    

    # Convenience function from WholeBrainModel
    def get_bce_loss(self, data_sim, device = "cuda"):
        """ Returns BCE loss w.r.t. label 1 (want to be real) """
        data_sim = data_sim.to(device)
        confidence = self(data_sim)
        target = torch.ones_like(confidence)
        return F.binary_cross_entropy(confidence, target)
    
    # Save + Load
    def save(self, checkpoint_path: str):
        Path(os.path.dirname(checkpoint_path)).mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path: str, map_location = "cpu"):
        self.load_state_dict(torch.load(checkpoint_path, map_location=map_location))