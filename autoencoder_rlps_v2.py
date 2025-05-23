import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

from typing import Type, Tuple, Optional
from src.utils import StandardizationPipeline
from meta_models import AutoencoderModel, IsomapModel, SphereModel, UmapModel, VariationalAutoencoder
import pandas as pd

from database import FileDataset
import numpy as np

def train_transform(cls: Type,
                    data_parameters:dict,
                    processing_parameters: dict,
                    model_parameters:dict,
                   ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

    process_seed = os.getpid() % 2 ** 32  # Ensure seed is within valid range
    torch.manual_seed(process_seed)  # Set a unique seed per process

    if not callable(cls):
        raise TypeError(f"{cls} is not a valid model class.")

    # Get data
    n_clusters = data_parameters.get("n_clusters", 4)
    force_clustering = data_parameters.get("force_clustering", False)
    write_file = data_parameters.get("write_file", True)
    month = data_parameters.get('month', 8)
    city = data_parameters.get('city', 'ENSCHEDE')

    data_provider = FileDataset(n_clusters=n_clusters, force_clustering=force_clustering, write_file=write_file)
    data = data_provider.get_dataset(month=month, gemeente=city)
    required_columns = {'GEMEENTE', 'BOXID', 'CLUSTER', 'MONTH'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Required columns {required_columns - set(data.columns)} not found in dataset")

    data_clusters = data[['CLUSTER', 'BOXID']].copy()

    data_values = data.drop(columns=['CLUSTER', 'GEMEENTE', 'MONTH', 'BOXID'])
    data_columns = data_values.columns  # Quarter values

    dataset = data_values.values

    # Process
    order_std = processing_parameters.get('order', 'row-column')
    standardize_pipe = StandardizationPipeline(order=order_std)
    dataset_std = standardize_pipe.fit_transform(dataset)

    # Train and transform
    try:
        model = cls(**model_parameters)
        latent_vectors = model.fit_transform(dataset_std)
    except Exception as e:
        print(f"Exception: {e}")
        latent_vectors = np.full_like(dataset_std[:, :3], None, dtype=object)

    # Prepare results
    results_parameters = dict(data_clusters=data_clusters,
                              gemeente=city,
                              month=month,
                              model=cls.name)


    latent_vectors = prepare_results(latent_vectors=latent_vectors,
                                     units='LD',
                                     column_names=['Z1', 'Z2', 'Z3'],
                                      **results_parameters)

    if hasattr(model, "reconstruct") and callable(getattr(model, "reconstruct")):
        decoded = model.reconstruct(dataset_std)
        decoded_kw = standardize_pipe.inverse_transform(decoded)

        decoded_vectors = prepare_results(latent_vectors=decoded,
                                          units='STD',
                                          column_names=data_columns.tolist(),
                                          **results_parameters)
        decoded_vectors_kw = prepare_results(latent_vectors=decoded_kw,
                                             units='KW',
                                             column_names=data_columns.tolist(),
                                             **results_parameters)
        decoded_vectors = pd.concat([decoded_vectors, decoded_vectors_kw], axis=0, ignore_index=True)

        return latent_vectors, decoded_vectors

    else:
        print("Warning: The model does not have a 'reconstruct' method.")
        return latent_vectors, None





def prepare_results(latent_vectors: np.array,
                    column_names: list[str],
                    units: str,
                    data_clusters: pd.DataFrame,
                    gemeente:str,
                    month:int,
                    model:str):
    latent_vectors_df = pd.DataFrame(latent_vectors,
                                     columns=column_names,
                                     index=data_clusters['BOXID'].tolist())

    data_clusters_indexed = data_clusters.set_index('BOXID')
    latent_vectors_df = latent_vectors_df.join(data_clusters_indexed)
    latent_vectors_df['GEMEENTE'] = gemeente
    latent_vectors_df['MONTH'] = month
    latent_vectors_df['MODEL'] = model
    latent_vectors_df['UNITS'] = units
    latent_vectors_df.reset_index(inplace=True, names=['BOXID'])
    latent_vectors_df = latent_vectors_df[['GEMEENTE', 'BOXID', 'MONTH', 'MODEL', 'UNITS'] + column_names + ['CLUSTER']]

    return latent_vectors_df



#%% Same but automatic:
data_parameters = dict(month=7,
                       city='ENSCHEDE',
                       n_clusters=4,
                       force_clustering=False,
                       write_file=False)
model_parameters = dict()
processing_parameters = dict(order='row')

model_class = SphereModel
latent, decoded = train_transform(cls=model_class,
                                  data_parameters=data_parameters,
                                  processing_parameters=processing_parameters,
                                  model_parameters=model_parameters,
                                  )


#% Plot the 3D latent space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
for cluster, color in zip(sorted(latent['CLUSTER'].unique().tolist()), colors):
    idx = latent['CLUSTER'] == cluster
    scatter = ax.scatter(latent[idx]['Z1'].values,
                         latent[idx]['Z2'].values,
                         latent[idx]['Z3'].values,
                         c=color,
                         alpha=0.7,
                         s=2,
                         label=f"Cluster {cluster} -> {sum(idx)}")
ax.set_title(f"3D Latent Space for {model_class.name.upper()} "
             f"- Month: {data_parameters['month']} "
             f"- City: {data_parameters['city']}")
ax.legend(loc="upper left")
ax.set_xlabel("Z1")
ax.set_ylabel("Z2")
ax.set_zlabel("Z3")
ax.set_aspect('equal')
plt.show()

