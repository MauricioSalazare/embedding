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
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from src.models import TorchDataset
from src.utils import scale, StandardizationPipeline
from src.metrics import mape
import pandas as pd



#%%
# Initialize TensorBoard writer
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"runs/autoencoder_{timestamp}"
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
model_path = os.path.join(model_dir, f"autoencoder_{timestamp}.pth")

writer = SummaryWriter(log_dir)

#%% Load dataset
clusters = pd.read_csv("./data/processed/rlps_clusters.csv", index_col=0)
dataset = pd.read_csv("./data/processed/rlps_filtered.csv", index_col=0)

dataset_clusters = dataset.join(clusters, how="inner")  # Merge clusters into df1

#%% Process
standardize_pipe = StandardizationPipeline()
dataset_array = standardize_pipe.fit_transform(dataset.values)
torch_data = TorchDataset(dataset_array)
dataloader = DataLoader(torch_data, batch_size=30, shuffle=True)


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
            nn.Linear(256, input_dim)  # No activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Initialize model, loss function, and optimizer
input_dim = 96  # MNIST images are 28x28
latent_dim = 3  # We want a 3D representation
model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.manual_seed(42)
# Training loop with TensorBoard logging
epochs = 1_000
for epoch in range(epochs):
    epoch_loss = 0
    for images in dataloader:
        optimizer.zero_grad()
        encoded, decoded = model(images)
        loss = criterion(decoded, images)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Log the loss to TensorBoard
    writer.add_scalar("Training Loss", avg_loss, epoch)

torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Close the TensorBoard writer



#%%
# Get the 3D latent representation for visualization
torch.manual_seed(42)
data_samples = next(iter(DataLoader(torch_data, batch_size=len(torch_data), shuffle=False)))
# data_samples, labels = next(iter(dataloader))  # Get a batch
with torch.no_grad():
   latent_vectors, decoded = model(data_samples)

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
                     c='grey',
                     alpha=0.7,
                     s=2)
ax.set_title("3D Latent Space of VAE")
ax.set_xlabel("Z1")
ax.set_ylabel("Z2")
ax.set_zlabel("Z3")
ax.set_aspect('equal')
plt.show()

#%%
autoencoder_df = pd.DataFrame(latent_vectors, columns=["Z1", "Z2", "Z3"], index= dataset.index)
decoder_df = pd.DataFrame(decoded, index= dataset.index)
decoder_kw_df = pd.DataFrame(decoded_kw, index= dataset.index)
data_stamples_df = pd.DataFrame(data_samples.numpy(), index=dataset.index)

#%% Save for dash
autoencoder_df.to_csv("./data/processed/autoencoder_rlps.csv")


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