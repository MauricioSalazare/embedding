import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

import pandas as pd
from src.utils import scale
from sklearn.cluster import KMeans


df = pd.read_csv("./data/processed/rlps_filtered.csv", index_col=0)

data = scale(df.values, axis=1)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(data)

#%%
fig, ax = plt.subplots(2,2, figsize=(10,10))
ax_ = ax.flatten()

for ii, ax_r in enumerate(ax_):
    idx = labels == ii
    ax_r.plot(data[idx,:].T, color=f"C{ii}", linewidth=0.5)

df_cluster = pd.DataFrame(labels, index=df.index, columns=['CLUSTER'])
df_cluster.to_csv('./data/processed/rlps_clusters.csv')


