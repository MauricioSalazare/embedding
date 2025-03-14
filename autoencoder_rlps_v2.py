import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

from src.utils import StandardizationPipeline
from meta_models import AutoencoderModel
import pandas as pd

from database import FileDataset

#%% Get data
data_provider = FileDataset()
data = data_provider.get_dataset()
data_values = data.drop(columns=['CLUSTER'])
dataset = data_values.values

#%% Initialize TensorBoard writer
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"runs/autoencoder_{timestamp}"
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
model_path = os.path.join(model_dir, f"autoencoder_{timestamp}.pth")

writer = SummaryWriter(log_dir)

#%% Process
standardize_pipe = StandardizationPipeline(order='row-column')
dataset_std = standardize_pipe.fit_transform(dataset)

#%% Modelling
vae = AutoencoderModel()
torch.manual_seed(42)
latent_vectors = vae.fit_transform(dataset_std, latent_dim=3)
decoded = vae.reconstruct(dataset_std)
decoded_kw = standardize_pipe.inverse_transform(decoded)

### Gather results
latent_vectors_df = pd.DataFrame(latent_vectors, index=data.index)
latent_vectors_df = latent_vectors_df.join(data['CLUSTER'])
decoded_df = pd.DataFrame(decoded, index=data.index)
decoded_kw_df = pd.DataFrame(decoded_kw, index=data.index)
dataset_std_df = pd.DataFrame(dataset_std, index=data.index)
#
#%% Plot the 3D latent space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
for cluster, color in zip(sorted(latent_vectors_df['CLUSTER'].unique().tolist()), colors):
    idx = latent_vectors_df['CLUSTER'] == cluster
    scatter = ax.scatter(latent_vectors_df[idx].iloc[:, 0].values,
                         latent_vectors_df[idx].iloc[:, 1].values,
                         latent_vectors_df[idx].iloc[:, 2].values,
                         c=color,
                         alpha=0.7,
                         s=2,
                         label=f"Cluster {cluster}")
ax.set_title("3D Latent Space fro VAE")
ax.legend(loc="upper left")
ax.set_xlabel("Z1")
ax.set_ylabel("Z2")
ax.set_zlabel("Z3")
ax.set_aspect('equal')
plt.show()

#%% Plot reconstruction
BOXID='ESD.001120-1'
fig, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].plot(decoded_df.loc[BOXID].values, linewidth=0.5, color='red', label="Decoded [std]")
ax[0].plot(dataset_std_df.loc[BOXID].values, linewidth=0.5, color='black', label="Original [std]")
ax[0].legend(loc='upper left', fontsize='small')
ax[0].set_title(f"Scaled dataset [std] - {BOXID}", fontsize='small')

ax[1].plot(decoded_kw_df.loc[BOXID].values, linewidth=0.5, color='red', label="Decoded [kW]")
ax[1].plot(data_values.loc[BOXID].values, linewidth=0.5, color='black', label="Original [kW]")
ax[1].legend(loc='upper left', fontsize='small')
ax[1].set_title(f"Dataset in power [kW] - {BOXID}", fontsize='small')

