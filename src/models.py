import torch
from torch import nn as nn


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
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Outputs should be between 0-1
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
