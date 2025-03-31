import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

import pandas as pd
from src.utils import scale
from sklearn.cluster import KMeans
from database import FileDataset
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment


#%% Homogenize clusters
N_CLUSTERS = 4


def cluster_month(data_scaled, n_clusters=N_CLUSTERS):
    """Runs KMeans clustering and returns labels and centroids."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    return labels, kmeans.cluster_centers_


def align_clusters(prev_centroids, current_centroids):
    """Aligns cluster labels using the Hungarian algorithm."""
    cost_matrix = np.linalg.norm(prev_centroids[:, None] - current_centroids[None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create label mapping and reassign centroids
    label_mapping = {old: new for old, new in zip(col_ind, row_ind)}
    reordered_centroids = np.zeros_like(current_centroids)
    for old, new in label_mapping.items():
        reordered_centroids[new] = current_centroids[old]

    return label_mapping, reordered_centroids


def store_results(gemeente, month, df_month, data_scaled, labels, centroids):
    """Stores cluster results for visualization and analysis."""
    scaled_data = pd.DataFrame(data_scaled,
                               columns=df_month.drop(columns=['GEMEENTE', 'BOXID', 'MONTH', 'CLUSTER'],
                                                     errors='ignore').columns)
    scaled_data['GEMEENTE'] = gemeente
    scaled_data['CLUSTER'] = labels
    scaled_data['MONTH'] = month
    scaled_data['BOXID'] = df_month['BOXID'].tolist()  # Add BOXID here

    centroids_data = pd.DataFrame(centroids, columns=scaled_data.columns[:-4])  # Exclude non-feature columns
    centroids_data['GEMEENTE'] = gemeente
    centroids_data['CLUSTER'] = list(range(N_CLUSTERS))
    centroids_data['MONTH'] = month

    return scaled_data, centroids_data


def cluster_and_align(dataset: pd.DataFrame, n_clusters: int, reference_month:int=7):
    # ----------------- MAIN LOOP -----------------
    normalized_data_df, centroids_data_df = [], []

    gemeentes = dataset['GEMEENTE'].unique().tolist()

    for gemeente in tqdm(gemeentes):
        df = dataset[dataset['GEMEENTE'] == gemeente].dropna()

        # First Month (Reference)
        df_month = df[df['MONTH'] == reference_month].copy()
        if df_month.empty:
            raise ValueError(f"No data found for {gemeente} in January. Labels cannot be aligned.")

        data_scaled = scale(df_month.drop(columns=['GEMEENTE', 'BOXID', 'MONTH', 'CLUSTER'], errors='ignore').values,
                            axis=1)
        labels, centroids = cluster_month(data_scaled, n_clusters=n_clusters)

        # Store results
        scaled_data, centroids_data = store_results(gemeente, reference_month, df_month, data_scaled, labels, centroids)
        normalized_data_df.append(scaled_data)
        centroids_data_df.append(centroids_data)

        centroids_list = [centroids]  # Initialize aligned centroids

        remaining_months = [i for i in range(1, 13) if i != reference_month]

        # Process Remaining Months
        for month in remaining_months:
            df_month = df[df['MONTH'] == month].copy()
            data_scaled = scale(df_month.drop(columns=['GEMEENTE', 'BOXID', 'MONTH', 'CLUSTER'], errors='ignore').values,
                                axis=1)

            labels, current_centroids = cluster_month(data_scaled, n_clusters=n_clusters)
            label_mapping, reordered_centroids = align_clusters(centroids_list[-1], current_centroids)

            # Relabel clusters based on mapping
            aligned_labels = np.array([label_mapping[label] for label in labels])

            # Store results
            scaled_data, centroids_data = store_results(gemeente, month, df_month, data_scaled, aligned_labels,
                                                                    reordered_centroids)
            normalized_data_df.append(scaled_data)
            centroids_data_df.append(centroids_data)
            centroids_list.append(reordered_centroids)  # Update reference centroids

    # Concatenate all results
    normalized_data_df = pd.concat(normalized_data_df, ignore_index=True)
    centroids_data_df = pd.concat(centroids_data_df, ignore_index=True)

    return normalized_data_df, centroids_data_df


if __name__ == '__main__':
    dataset = FileDataset()
    data_to_cluster = dataset.get_dataset()

    normalized_data_df, centroids_data_df = cluster_and_align(data_to_cluster, N_CLUSTERS, reference_month=1)



    #%%
    GEMEENTE = 'ENSCHEDE'

    fix, ax = plt.subplots(12,4, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    for month in range(1, 13):
        for cluster in range(N_CLUSTERS):
            eigenprofile = centroids_data_df[(centroids_data_df['MONTH'] == month) &
                                             (centroids_data_df['GEMEENTE'] == GEMEENTE) &
                                             (centroids_data_df['CLUSTER'] == cluster)].drop(columns=['CLUSTER', 'GEMEENTE', 'MONTH']).values
            ax[month-1, cluster].plot(eigenprofile.flatten(), linewidth=0.5, color=f'C{cluster}')


    #%%


    fix, ax = plt.subplots(12,4, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    for month in range(1, 13):
        for cluster in range(N_CLUSTERS):
            eigenprofile = normalized_data_df[(normalized_data_df['MONTH'] == month) &
                                             (normalized_data_df['GEMEENTE'] == GEMEENTE) &
                                             (normalized_data_df['CLUSTER'] == cluster)].drop(columns=['CLUSTER', 'GEMEENTE', 'MONTH', 'BOXID']).values
            eigenprofile_one = eigenprofile.mean(axis=0)
            ax[month-1, cluster].plot(eigenprofile_one.flatten(), linewidth=0.5, color=f'C{cluster}', label=f"{eigenprofile.shape[0]}")

            ax[month-1, cluster].legend(fontsize='xx-small')

            ax[month - 1, cluster].set_xticks([])
            ax[month - 1, cluster].set_yticks([])
            ax[month - 1, cluster].set_xlabel('')

            ax[month - 1, cluster].set_ylabel(f"{month}")

