# Single-call helper function for VGAE pre-training and classifier training

from typing import List
import torch
import matplotlib.pyplot as plt
from graph_builder import GraphBuilder
from vgae_discriminator import build_vgae, DiscriminatorVGAE

def build_and_train_discriminator(real_graphs: List, simualated_graphs: List, figure_path = "", device = "cuda"):
    """
    real_graphs, simulated_graphs: Lists of PyG Data objects
    Returns ready-to-use DiscriminatorVGAE instance
    """

    in_dim = real_graphs[0].x.size(1)
    vgae = build_vgae(in_dim=in_dim, hidden=64, latent=32)

    # 1. Unsupervised pre-train
    DiscriminatorVGAE.train_unsupervised(vgae, real_graphs, device=device)

    # 2. Classifer
    discriminator = DiscriminatorVGAE(vgae, latent_dim=32)
    losses = DiscriminatorVGAE.train_classifier(discriminator, real_graphs + simualated_graphs, device=device)

    plt.plot(losses)
    plt.title("Classifier BCE")
    plt.save(figure_path)
    
    return discriminator.to(device)