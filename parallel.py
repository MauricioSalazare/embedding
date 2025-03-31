import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

import torch
import os
import multiprocessing

from typing import Type, Tuple, Optional
from src.utils import StandardizationPipeline
from meta_models import AutoencoderModel, IsomapModel, SphereModel, UmapModel, VariationalAutoencoder, TSNEModel
import pandas as pd
from functools import partial

from database import FileDataset
import numpy as np

import time

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
    write_file = data_parameters.get("write_file", False)

    month = data_parameters['month']
    gemeente = data_parameters['gemeente']

    data_provider = FileDataset(n_clusters=n_clusters, force_clustering=force_clustering, write_file=write_file)
    data = data_provider.get_dataset(month=month, gemeente=gemeente)
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
        print(f"Something went wrong! Exception: {e}")
        print(f"Model: {cls}")
        print(f"Gemeente: {gemeente}")
        print(f"Month: {month}")
        latent_vectors = np.full_like(dataset_std[:, :3], None, dtype=object)

    # Prepare results
    results_parameters = dict(data_clusters=data_clusters,
                              gemeente=gemeente,
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

if __name__ == '__main__':


    #%% Same but automatic:
    _ = FileDataset(n_clusters=4, force_clustering=True, write_file=True)  # Force clustering once

    data_provider = FileDataset()
    gemeentes = data_provider.get_gemeente_list()

    tasks_sphere = [
        dict(cls=cls_model,
             data_parameters=dict(month=month, gemeente=gemeente),
             processing_parameters=dict(order='row'),
             model_parameters=dict())  # Leave default
        for month in range(1, 13)
        for gemeente in gemeentes
        for cls_model in [SphereModel, IsomapModel]
    ]

    tasks_fast = [
        dict(cls=cls_model,
             data_parameters=dict(month=month, gemeente=gemeente),
             processing_parameters=dict(order='row-column'),
             model_parameters=dict())  # Leave default
        for month in range(1, 13)
        for gemeente in gemeentes
        for cls_model in [UmapModel, TSNEModel]
    ]

    tasks_nn = [
        dict(cls=cls_model,
             data_parameters=dict(month=month, gemeente=gemeente),
             processing_parameters=dict(order='row-column'),
             model_parameters=dict())
        for month in range(1, 13)
        for gemeente in gemeentes
        for cls_model in [AutoencoderModel]
    ]

    # tasks = tasks_fast + tasks_nn
    tasks = tasks_sphere + tasks_fast

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use partial to adapt function to starmap
        func = partial(train_transform)
        results = pool.starmap(func, [(t["cls"],
                                       t["data_parameters"],
                                       t["processing_parameters"],
                                       t["model_parameters"]) for t in tasks])

    latent_vectors = [result[0] for result in results ]
    latent_vectors = pd.concat(latent_vectors, ignore_index=True)

    latent_vectors.to_csv('./data/processed/latent_vectors.csv', index=False)
