import dash
from dash import dcc, html, Input, Output, callback_context, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np

np.random.seed(42)
# df1 = pd.read_csv("./data/processed/umap_rlps.csv", index_col=0)
# df1 = pd.read_csv("./data/processed/vae_rlps.csv", index_col=0)
# df1 = pd.read_csv("./data/processed/autoencoder_rlps.csv", index_col=0)
df1 = pd.read_csv("./data/processed/isomap_rlps.csv", index_col=0)
df2 = pd.read_csv("./data/processed/ts_enschede_2023.csv")
df3 = pd.read_csv("./data/processed/rlps_clusters.csv", index_col=0)

# Ensure df1 and df3 align by index
df1 = df1.join(df3, how="inner")  # Merge clusters into df1


# Create Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='scatter_plot', style={'height': '70vh'})
    ], style={"width": "60%", "display": "inline-block"}),

    html.Div([
        dcc.Graph(id='time_series', style={'height': '70vh'})
    ], style={"width": "40%", "display": "inline-block"})
])

# Callback to update the time series based on scatter plot selection
@app.callback(
    Output('time_series', 'figure'),
    Input('scatter_plot', 'clickData')
)
def update_time_series(clickData):
    if not clickData or "points" not in clickData:
        return no_update


    object_id = clickData["points"][0]["text"]  # No need for [0] since we store it as a single value


    # Filter df2 for selected object
    filtered_df = df2[df2["BOXID"] == object_id]

    # Create time series plot using go.Figure()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df["QUARTER"],
        y=filtered_df["POWER"],
        mode='lines',
        name=f'Object {object_id}',
        line=dict(color='blue')
    ))

    # Update layout
    fig.update_layout(
        title=f"Time Series for Object {object_id}",
        xaxis_title="Time",
        yaxis_title="Value"
    )

    return fig

# Callback to create scatter plot
@app.callback(
    Output('scatter_plot', 'figure'),
    Input('scatter_plot', 'relayoutData')  # Placeholder to trigger initial load
)
def update_scatter(_):
    ctx = callback_context

    # If triggered by relayout (zoom, pan, rotate), do not update the figure
    if ctx.triggered and "relayoutData" in ctx.triggered[0]["prop_id"]:
        return no_update  # Prevent the figure from resetting

    cluster_labels = df1['CLUSTER']
    unique_clusters = cluster_labels.unique()

    # Assign a color to each cluster
    cluster_colors = {
        cluster: f'rgb({np.random.randint(0, 255)},{np.random.randint(0, 255)},{np.random.randint(0, 255)})' for cluster
        in unique_clusters}

    # Create 3D scatter plot using go.Figure()
    fig = go.Figure()

    for cluster in unique_clusters:
        cluster_data = df1[cluster_labels == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data["Z1"],
            y=cluster_data["Z2"],
            z=cluster_data["Z3"],
            mode='markers',
            marker=dict(size=4, color=cluster_colors[cluster]),
            text=cluster_data.index,  # Show object ID
            name=f"Cluster {cluster}"  # Legend entry
        ))
    # Update layout
    fig.update_layout(
        title="3D Scatter Plot",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )

    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
