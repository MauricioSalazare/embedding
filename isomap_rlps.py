import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

from src.plotting import create_mouse_move_handler, plot_3D
from src.utils import StandardizationPipeline
import pandas as pd
import numpy as np

from sklearn.manifold import Isomap
import umap

#%% Load dataset
clusters = pd.read_csv("./data/processed/rlps_clusters.csv", index_col=0)
dataset = pd.read_csv("./data/processed/rlps_filtered.csv", index_col=0)

dataset_clusters = dataset.join(clusters, how="inner")  # Merge clusters into df1


#%% Process
standardize_pipe = StandardizationPipeline(order="row")
data_samples = standardize_pipe.fit_transform(dataset.values)

# Apply UMAP to MNIST
np.random.seed(42)
isomap = Isomap(n_neighbors=10, n_components=3)
isomap_data = isomap.fit_transform(data_samples)


# umap_model = umap.UMAP(n_components=3, n_neighbors=5, random_state=42)
# umap_data = umap_model.fit_transform(data_samples)

umap_frame = pd.DataFrame(isomap_data, index=dataset.index, columns=['Z1', 'Z2', 'Z3'])
umap_frame.to_csv('./data/processed/isomap_rlps.csv')

#%% Plot results
fig, axes = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(10, 10))
# axes = axes.flatten()
scatter = plot_3D(isomap_data, dataset_clusters['CLUSTER'].values, "UMAP 3D Representation", ax=axes)
fig.canvas.mpl_connect("motion_notify_event", create_mouse_move_handler(fig, axes))
plt.colorbar(scatter)
plt.tight_layout()
plt.show()