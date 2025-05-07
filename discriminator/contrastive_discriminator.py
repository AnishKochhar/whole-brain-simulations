import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool


## MARK: Graph Augmentations and Loss values

def alignment(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 2):
    """
    Alignment loss: encourages positive pairs (z1, z2) to be close. Assumes z1 and z2 are L2-normalized
    """
    return (F.normalize(z1, dim=1) * F.normalize(z2, dim=1)).sum(dim=1).mean().item()
    # return (z1 - z2).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(z: torch.Tensor, t: float = 2):
    """
    Uniformity loss: encourages representations to be spread out (Wang & Isola, 2020)
    Lower values mean more 'collapse' to the same point
    """
    z = F.normalize(z, dim=1)
    return (torch.pdist(z, p=2).pow(2).mul(-t).exp().mean()).log().item()


def graph_augment(data, drop_p = 0.2, mask_p = 0.2):
    """ Applies random edge dropout and node feature masking """
    edge_index, edge_weight = dropout_edge(data.edge_index, p=drop_p, training=True, force_undirected=True)
    x = data.x.clone()
    mask = torch.rand_like(x) < mask_p
    x[mask] = 0.0
    new_data = data.clone()
    new_data.edge_index, new_data.edge_weight, new_data.x = edge_index, edge_weight, x
    return new_data


# Encoder
class GraphEncoder(nn.Module):      # 4-layer GIN
    def __init__(self, in_dim, hidden=64, latent=32):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in range(4):
            dim_in = in_dim if k == 0 else hidden
            mlp = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
        self.readout = nn.Linear(hidden, latent)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index)
        z = F.normalize(self.readout(x), p=2, dim=-1)
        return z

class NTXentLoss(nn.Module):
    def __init__(self, temperature = 0.5):
        super().__init__()
        self.t = temperature
    
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.t # Cosine sims
        labels = torch.arange(B, device=z.device).repeat(2)
        loss = F.cross_entropy(sim, labels)
        return loss

class GraphAttentionPool(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(latent, latent), nn.Tanh(), nn.Linear(latent, 1))

    def forward(self, z: torch.Tensor, batch: torch.Tensor):
        """
        Attention weights normalised within each graph

        z: (N, latent), batch: (N, ) graph-id for every node, returns: (batch, latent)
        """
        # Attention score per node
        alpha = self.att(z).squeeze(-1)  # (N)
        alpha = torch.exp(alpha - alpha.max())

        alpha_sum = global_add_pool(alpha, batch) # (B, )
        alpha = alpha / (alpha_sum[batch] + 1e-8) # (N, )
        # Weighted node aggregation
        graph_z = global_add_pool(z * alpha.unsqueeze(1), batch) # (B, latent)
        return graph_z
    
class GraphClassifier(nn.Module):
    """ Attention-pool + 3-layer MLP, returns probability in [0, 1] """
    def __init__(self, latent = 32):
        super().__init__()
        self.pool = GraphAttentionPool(latent)
        self.mlp = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, z, batch):
        z_graph = self.pool(z, batch)
        return torch.sigmoid(self.mlp(z_graph)).view(-1) # (B, )

class CriticHead(nn.Module):  
    """ Single layer Wasserstein critic """
    def __init__(self, latent = 32):
        super().__init__()
        self.score = nn.Linear(latent, 1)
    
    def forward(self, graph_embeddings):
        return self.score(graph_embeddings).view(-1) # (B, )
    

class Discriminator(nn.Module):
    """ use_critic == False -> BCE Classifier
        use_critic == True  -> WGAN0GP critic (no sigmoid)
    """
    def __init__(self, encoder: nn.Module, latent = 32, use_critic = False):
        super().__init__()
        self.encoder = encoder
        self.use_critic = use_critic
        if use_critic:
            self.classifier = CriticHead(latent)
        else:
            self.classifier = GraphClassifier(latent)
    
    def forward(self, data: Data):
        z_nodes = self.encoder(data) # (N, latent)

        if hasattr(data, 'batch') and data.batch is not None:
            batch_vec = data.batch
        else:
            batch_vec = torch.zeros(z_nodes.size(0), dtype=torch.long, device=z_nodes.device)

        if self.use_critic:
            z_graph = global_add_pool(z_nodes, batch_vec)
            return self.classifier(z_graph) # Raw score
        else:
            return self.classifier(z_nodes, batch_vec) # Sigmoid prob
