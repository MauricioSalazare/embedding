import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

from src.plotting import create_mouse_move_handler, plot_3D
from src.utils import StandardizationPipeline
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

#%% Load dataset
dataset_raw = pd.read_csv("./data/processed/rlps_2023_data_clean.csv")

dataset_clusters = dataset_raw[(dataset_raw["MONTH"] == 1) & (dataset_raw["GEMEENTE"] == 'ENSCHEDE')]
dataset = dataset_clusters.drop(columns=["MONTH", "GEMEENTE", "CLUSTER"]).set_index("BOXID")


#%% Process
standardize_pipe = StandardizationPipeline()
data_samples = standardize_pipe.fit_transform(dataset.values)

# Apply UMAP to MNIST
np.random.seed(42)
tsne_model = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
tsne_data = tsne_model.fit_transform(data_samples)

umap_frame = pd.DataFrame(tsne_data, index=dataset.index, columns=['Z1', 'Z2', 'Z3'])
umap_frame.to_csv('./data/processed/tsne_rlps.csv')

#%% Plot results
fig, axes = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(10, 10))
# axes = axes.flatten()
scatter = plot_3D(tsne_data, dataset_clusters['CLUSTER'].values, "t-SNE 3D Representation", ax=axes)
fig.canvas.mpl_connect("motion_notify_event", create_mouse_move_handler(fig, axes))
plt.colorbar(scatter)
plt.tight_layout()
plt.show()