## Utils for saving / loading frozen models
import torch
import torch.nn as nn
from pathlib import Path
from discriminator.contrastive_discriminator import GraphEncoder, Discriminator


def save_frozen_models(encoder: nn.Module, discriminator: nn.Module, path_root = "checkpoints"):
    path_root = Path(path_root)
    path_root.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path_root / "encoder.pt")
    torch.save(discriminator.state_dict(), path_root / "disciminator.pt")
    print("Successfully saved encoder and discriminator!")

def load_encoder(path: str, in_dim: int, hidden: int, latent: int, freeze: bool = True,
                 device: str = "cuda") -> GraphEncoder:
    encoder = GraphEncoder(in_dim, hidden, latent).to(device)
    encoder.load_state_dict(torch.load(path, map_location=device))
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze
    return encoder

def load_discriminator(path: str, encoder: GraphEncoder, latent: int, use_critic: bool = False, 
                       freeze: bool = True, device: str = "cuda") -> Discriminator:
    discriminator = Discriminator(encoder, latent, use_critic=use_critic).to(device)
    discriminator.load_state_dict(torch.load(path, map_location=device))
    discriminator.eval()

    for parameter in discriminator.parameters():
        parameter.requires_grad = not freeze
    return discriminator