import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

from src.plotting import create_mouse_move_handler, plot_3D
from src.utils import StandardizationPipeline
import pandas as pd
import numpy as np

import umap

#%% Load dataset
dataset_raw = pd.read_csv("./data/processed/rlps_2023_data_clean.csv")

dataset_clusters = dataset_raw[(dataset_raw["MONTH"] == 7) & (dataset_raw["GEMEENTE"] == 'ENSCHEDE')]
dataset = dataset_clusters.drop(columns=["MONTH", "GEMEENTE", "CLUSTER"]).set_index("BOXID")


#%% Process
standardize_pipe = StandardizationPipeline()
data_samples = standardize_pipe.fit_transform(dataset.values)

# Apply UMAP to MNIST
np.random.seed(42)
umap_model = umap.UMAP(n_components=3, n_neighbors=5, random_state=42)
umap_data = umap_model.fit_transform(data_samples)

umap_frame = pd.DataFrame(umap_data, index=dataset.index, columns=['Z1', 'Z2', 'Z3'])
umap_frame.to_csv('./data/processed/umap_rlps.csv')

#%% Plot results
fig, axes = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(10, 10))
# axes = axes.flatten()
scatter = plot_3D(umap_data, dataset_clusters['CLUSTER'].values, "UMAP 3D Representation", ax=axes)
fig.canvas.mpl_connect("motion_notify_event", create_mouse_move_handler(fig, axes))
plt.colorbar(scatter)
plt.tight_layout()
plt.show()