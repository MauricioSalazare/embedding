import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.use("MacOSX")

import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


from sklearn.decomposition import PCA
import umap
from tqdm import trange

from sklearn.metrics import mean_squared_error, silhouette_score
from scipy.spatial import procrustes
from sklearn.manifold import trustworthiness
from sklearn.feature_selection import mutual_info_regression


import plotly.graph_objects as go
import base64
import io
from PIL import Image



#%%
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_data, batch_size=128, shuffle=True)


#%%
# Define Autoencoder
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
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#%%
# Initialize model, loss function, and optimizer
input_dim = 28 * 28  # MNIST images are 28x28
latent_dim = 3  # We want a 3D representation
model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%%
# Train Autoencoder
epochs = 1
for epoch in trange(epochs):
    for images, _ in dataloader:
        optimizer.zero_grad()
        encoded, decoded = model(images)
        loss = criterion(decoded, images)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#%%
# Get encoded 3D representations
# model_dir = "saved_models"
# model_path = os.path.join(model_dir, f"autoencoder_2025-03-10_14-04-28.pth")
# model = Autoencoder(input_dim, latent_dim)
# model.load_state_dict(torch.load(model_path))

# Get more samples and a deterministic portion for the data
model.eval()
torch.manual_seed(42)

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
new_dataloader = DataLoader(test_data, batch_size=3000, shuffle=False)  # ✅ No shuffling


data_samples, labels = next(iter(new_dataloader))

with torch.no_grad():
    encoded_data, decoded_data = model(data_samples)
encoded_data = encoded_data.numpy()
reconstructed_data = decoded_data.numpy()
labels = labels.numpy()

#%%
# Apply PCA to MNIST
pca = PCA(n_components=3)
pca_data = pca.fit_transform(data_samples.numpy())
pca_reconstructed = pca.inverse_transform(pca_data)

#%%
# Apply UMAP to MNIST
umap_model = umap.UMAP(n_components=3)
umap_data = umap_model.fit_transform(data_samples.numpy())


#%%
# Lock roll (Z-axis rotation) by overriding rotation behavior
def create_mouse_move_handler(fig, axes_list):
    def on_mouse_move(event):
        for ax in axes_list:
            if event.button == 1 and event.inaxes == ax:  # Left mouse button
                elev, azim = ax.elev, ax.azim  # Get current rotation angles
                ax.view_init(elev=elev, azim=azim)  # Lock roll (Z-axis)
        fig.canvas.draw_idle()  # Refresh all plots
    return on_mouse_move  # Return the handler function

# 3D Plot function
def plot_3D(data, labels, title, ax):

    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab10', alpha=0.7, s=2)
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')

    return scatter



#%%
# Compute Evaluation Metrics

# 1. Reconstruction Error (MSE)
autoencoder_mse = mean_squared_error(data_samples.numpy(), reconstructed_data)
pca_mse = mean_squared_error(data_samples.numpy(), pca_reconstructed)
print(f"Autoencoder MSE: {autoencoder_mse:.4f}")
print(f"PCA MSE: {pca_mse:.4f}")

# 2. Explained Variance (only for PCA)
explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
print(f"PCA Explained Variance Ratio: {explained_variance_ratio:.4f}")

# 3. Trustworthiness Score
trust_autoencoder = trustworthiness(data_samples.numpy(), encoded_data, n_neighbors=5)
trust_pca = trustworthiness(data_samples.numpy(), pca_data, n_neighbors=5)
trust_umap = trustworthiness(data_samples.numpy(), umap_data, n_neighbors=5)
print(f"Trustworthiness - Autoencoder: {trust_autoencoder:.4f}, PCA: {trust_pca:.4f}, UMAP: {trust_umap:.4f}")

# 4. Procrustes Distance
_, _, procrustes_autoencoder = procrustes(pca_data, encoded_data)
_, _, procrustes_pca = procrustes(pca_data, pca_data)
_, _, procrustes_umap = procrustes(pca_data, umap_data)
print(f"Procrustes Distance - Autoencoder: {procrustes_autoencoder:.4f}, PCA: {procrustes_pca:.4f}, UMAP: {procrustes_umap:.4f}")

# 5. Mutual Information (MI)
mi_autoencoder = np.mean([mutual_info_regression(data_samples.numpy(), encoded_data[:, i]) for i in range(3)])
mi_pca = np.mean([mutual_info_regression(data_samples.numpy(), pca_data[:, i]) for i in range(3)])
mi_umap = np.mean([mutual_info_regression(data_samples.numpy(), umap_data[:, i]) for i in range(3)])
print(f"Mutual Information - Autoencoder: {mi_autoencoder:.4f}, PCA: {mi_pca:.4f}, UMAP: {mi_umap:.4f}")

# 6. Silhouette Score
silhouette_autoencoder = silhouette_score(encoded_data, labels)
silhouette_pca = silhouette_score(pca_data, labels)
silhouette_umap = silhouette_score(umap_data, labels)
print(f"Silhouette Score - Autoencoder: {silhouette_autoencoder:.4f}, PCA: {silhouette_pca:.4f}, UMAP: {silhouette_umap:.4f}")





#%% Plot results
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')

fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(10, 10))
axes = axes.flatten()

plot_3D(encoded_data, labels, "Autoencoder 3D Representation", ax=axes[0])
plot_3D(pca_data, labels, "PCA 3D Representation", ax=axes[1])
scatter = plot_3D(umap_data, labels, "UMAP 3D Representation", ax=axes[2])
fig.canvas.mpl_connect("motion_notify_event", create_mouse_move_handler(fig, axes))
plt.colorbar(scatter)
plt.tight_layout()
plt.show()