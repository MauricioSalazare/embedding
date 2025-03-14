import matplotlib

from src.models import VAE

matplotlib.use("MacOSX")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import numpy as np

class TorchDataset(Dataset):
    def __init__(self, data: np.ndarray, labels=None):
        self.data = data
        self.labels = labels if labels is not None else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return self.data[idx]

        else:
            return self.data[idx], self.labels[idx]

def standardize(data: np.ndarray, axis=0) -> np.ndarray:
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    return (data - mean) / std




def vae_loss(reconstructed_x, x, mu, logvar):
    """Loss function for VAE (Reconstruction Loss _ KL Divergence)"""
    reconstruction_loss = nn.MSELoss(reduction="sum")(reconstructed_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return reconstruction_loss + kl_divergence



# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

# Initialize model, loss function, and optimizer
input_dim = 28 * 28  # MNIST images are 28x28
latent_dim = 3  # ✅ Force a 3D latent space
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the VAE
epochs = 1
for epoch in range(epochs):
    total_loss = 0
    for images, _ in dataloader:
        optimizer.zero_grad()
        reconstructed_x, mu, logvar, _ = model(images)
        loss = vae_loss(reconstructed_x, images, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


#%%
# Get the 3D latent representation for visualization
torch.manual_seed(42)

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
new_dataloader = DataLoader(test_data, batch_size=3000, shuffle=False)  # ✅ No shuffling


data_samples, labels = next(iter(new_dataloader))
# data_samples, labels = next(iter(dataloader))  # Get a batch
with torch.no_grad():
    _, _, _, latent_vectors = model(data_samples)

# Convert to numpy for plotting
latent_vectors = latent_vectors.numpy()
labels = labels.numpy()

# Plot the 3D latent space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2], c=labels, cmap='tab10', alpha=0.7, s=2)
ax.set_title("3D Latent Space of VAE")
ax.set_xlabel("Z1")
ax.set_ylabel("Z2")
ax.set_zlabel("Z3")
plt.colorbar(scatter, label="Digit Label")
plt.show()
