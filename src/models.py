import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import Dataset


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(64, latent_dim)  # Mean of latent distribution
        self.logvar_layer = nn.Linear(64, latent_dim)  # Log-variance for reparameterization

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),  # No activation
        )

    def reparameterize(self, mu, logvar):
        """ Implements the reparameterization trick: z = μ + σ * ε """
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        epsilon = torch.randn_like(std)  # Sample from a standard normal distribution
        return mu + std * epsilon  # Reparametrize

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.reparameterize(mu, logvar)  # Sampled latent vector

        decoded = self.decoder(z)

        return decoded, mu, logvar, z

def vae_loss(reconstructed_x, x, mu, logvar):
    """Loss function for VAE (Reconstruction Loss _ KL Divergence)"""
    reconstruction_loss = nn.MSELoss(reduction="sum")(reconstructed_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return reconstruction_loss + kl_divergence



class TorchDataset(Dataset):
    def __init__(self, data: np.ndarray, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels if labels is not None else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return self.data[idx]

        else:
            return self.data[idx], self.labels[idx]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)  # No activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
