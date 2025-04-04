import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("MacOSX")

from src.models import VAE, TorchDataset
from src.utils import StandardizationPipeline
from src.metrics import mape


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd


def vae_loss(reconstructed_x, x, mu, logvar):
    """Loss function for VAE (Reconstruction Loss _ KL Divergence)"""
    reconstruction_loss = nn.MSELoss(reduction="sum")(reconstructed_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return reconstruction_loss + kl_divergence



# Load dataset
clusters = pd.read_csv("./data/processed/rlps_clusters.csv", index_col=0)
dataset = pd.read_csv("./data/processed/rlps_filtered.csv", index_col=0)

dataset_clusters = dataset.join(clusters, how="inner")  # Merge clusters into df1

#%% Process
standardize_pipe = StandardizationPipeline()
dataset_array = standardize_pipe.fit_transform(dataset.values)
torch_data = TorchDataset(dataset_array)
dataloader = DataLoader(torch_data, batch_size=30, shuffle=True)


# Initialize model, loss function, and optimizer
input_dim = 96  # Load profiles at 15 min resolution
latent_dim = 3  # Force a 3D latent space
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the VAE
epochs = 1000
for epoch in range(epochs):
    total_loss = 0
    for images in dataloader:
        optimizer.zero_grad()
        reconstructed_x, mu, logvar, _ = model(images)
        loss = vae_loss(reconstructed_x, images, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


#%%
# Pull all data and do the 3D latent representation for visualization
torch.manual_seed(42)
data_samples = next(iter(DataLoader(torch_data, batch_size=len(torch_data), shuffle=False)))
with torch.no_grad():
    decoded, _, _, latent_vectors = model(data_samples)

# Convert to numpy for plotting
latent_vectors = latent_vectors.numpy()
decoded = decoded.numpy()
decoded_kw = standardize_pipe.inverse_transform(decoded)

# Dataframe latent vector
latent_vectors_df = pd.DataFrame(latent_vectors, index=dataset.index)
latent_vectors_df = latent_vectors_df.join(clusters, how="inner")

#%%
# Plot the 3D latent space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(latent_vectors_df.iloc[:, 0].values,
                     latent_vectors_df.iloc[:, 1].values,
                     latent_vectors_df.iloc[:, 2].values,
                     c=latent_vectors_df.iloc[:, -1].values,
                     alpha=0.7,
                     s=2)
ax.set_title("3D Latent Space of VAE")
ax.set_xlabel("Z1")
ax.set_ylabel("Z2")
ax.set_zlabel("Z3")
ax.set_aspect('equal')
plt.show()
#%%
vae_df = pd.DataFrame(latent_vectors, columns=["Z1", "Z2", "Z3"], index= dataset.index)
decoder_df = pd.DataFrame(decoded, index= dataset.index)
decoder_kw_df = pd.DataFrame(decoded_kw, index= dataset.index)
data_stamples_df = pd.DataFrame(data_samples.numpy(), index=dataset.index)


#%% Save for dash
vae_df.to_csv("./data/processed/vae_rlps.csv")

#%%
BOXID='ESD.001120-1'
fig, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(decoder_df.loc[BOXID].values, linewidth=0.5, color='red', label="Decoded [std]")
ax[0].plot(data_stamples_df.loc[BOXID].values, linewidth=0.5, color='black', label="Original [std]")
ax[0].legend(loc='upper left', fontsize='small')
ax[0].set_title(f"Scaled dataset [std] - {BOXID}", fontsize='small')

ax[1].plot(decoder_kw_df.loc[BOXID].values, linewidth=0.5, color='red', label="Decoded [kW]")
ax[1].plot(dataset.loc[BOXID].values, linewidth=0.5, color='black', label="Original [kW]")
ax[1].legend(loc='upper left', fontsize='small')
ax[1].set_title(f"Dataset in power [kW] - {BOXID}", fontsize='small')

#%%
print(f"MAPE: {mape(decoder_kw_df.loc[BOXID].values, dataset.loc[BOXID].values):.2f}%")