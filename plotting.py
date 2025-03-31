import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

#%%
latent_vectors = pd.read_csv('./data/processed/latent_vectors.csv')
rlps_data = pd.read_csv('./data/processed/rlps_2023_data.csv')


#%% Define models and layout
models = ['isomap', 'sphere', 'umap', 'autoencoder']
titles = [f"Model: {model.capitalize()}" for model in models]

# HIGHLIGHT = 'ESD.001124-1'
HIGHLIGHT = None
HIGHLIGHT = '027.5610-1'
HIGHLIGHT = '150.559-1'

MONTH = 8
GEMEENTE = 'BREDA'
# Define matplotlib-like colors for clusters
cluster_colors = {
    0: plt.get_cmap("tab10")(0),
    1: plt.get_cmap("tab10")(1),
    2: plt.get_cmap("tab10")(2),
    3: plt.get_cmap("tab10")(3),
}

# Define layout for subplots (2x2 for 3D scatter, 1x2 for time series)
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=titles + ["Load Profile", ""],
    specs=[
        [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
        [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
        [{'type': 'xy', 'colspan': 2}, None]  # Last row is for time series
    ],
    row_heights=[0.4, 0.4, 0.2],  # Adjust height to give more space for the time series
)

# Process HIGHLIGHT data for the time-series plot
if HIGHLIGHT is not None:
    rlps = rlps_data[(rlps_data['GEMEENTE'] == GEMEENTE) &
                     (rlps_data['MONTH'] == MONTH) &
                     (rlps_data['BOXID'] == HIGHLIGHT)].drop(columns=["GEMEENTE", "MONTH", "BOXID", "CLUSTER"])
    date_range = pd.date_range(start=f"2023-{MONTH:02d}-01", periods=96, freq='15min')
    rlps_df = pd.DataFrame(rlps.values.flatten(), index=date_range, columns=['power'])

# Add a scatter plot for each model
for i, model in enumerate(models):
    row = i // 2 + 1
    col = i % 2 + 1

    # Filter data for the specific model
    data = latent_vectors[(latent_vectors['GEMEENTE'] == GEMEENTE) &
                          (latent_vectors['MONTH'] == MONTH) &
                          (latent_vectors['MODEL'] == model)]
    data_highlight = data[data['BOXID'] == HIGHLIGHT]

    # Create traces for each cluster separately
    for cluster_id in sorted(data['CLUSTER'].unique()):
        cluster_data = data[data['CLUSTER'] == cluster_id]

        fig.add_trace(
            go.Scatter3d(
                x=cluster_data['Z1'],
                y=cluster_data['Z2'],
                z=cluster_data['Z3'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=f"rgb{tuple(int(c * 255) for c in cluster_colors[cluster_id][:3])}",
                    opacity=0.8
                ),
                text=cluster_data['BOXID'],
                showlegend=False
            ),
            row=row, col=col
        )

        if row == 1 and col == 1:
            showlegend = True
        else:
            showlegend = False

        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    size=15,
                    color=f"rgb{tuple(int(c * 255) for c in cluster_colors[cluster_id][:3])}"
                ),
                name=f"Cluster {cluster_id}",
                showlegend=showlegend
            )
        )

    if HIGHLIGHT is not None:
        fig.add_trace(
            go.Scatter3d(
                x=data_highlight['Z1'],
                y=data_highlight['Z2'],
                z=data_highlight['Z3'],
                mode='markers',
                marker=dict(
                    size=5,
                    color="purple"
                ),
                name=f"Highlight",
                showlegend=False
            ),
            row=row, col=col
        )

# Add time-series plot if HIGHLIGHT is not None
if HIGHLIGHT is not None:
    fig.add_trace(
        go.Scatter(
            x=rlps_df.index,
            y=rlps_df['power'],
            mode='lines',
            line=dict(color="purple", width=2),
            name="Load Profile"
        ),
        row=3, col=1
    )

# Update layout
fig.update_layout(
    title="3D Scatter Plots for Different Models & Load Profile",
    height=1200, width=1100,
    showlegend=True
)

# Show figure
fig.show()