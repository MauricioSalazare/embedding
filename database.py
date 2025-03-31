from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import warnings
from tqdm import tqdm


load_dotenv()
DATASET_FILE_PATH = Path(os.getenv("DATASET_FILE_PATH"))
CLUSTERS_FILE_PATH = Path(os.getenv("CLUSTERS_FILE_PATH"))

class DataProvider(ABC):
    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_gemeente_list(self) -> list:
        pass

    @abstractmethod
    def get_ids_list(self) -> list:
        pass


class FileDataset(DataProvider):

    def __init__(self, n_clusters=4, force_clustering=False, write_file=False):
        self.n_clusters = n_clusters
        self.force_clustering = force_clustering
        self.write_file = write_file
        self.dataset_loaded = False

        self.load_dataset()


    def load_dataset(self):
        """This is a naive loading of the dataset. If it doesn't fit in memory, you are toasted"""
        self.dataset = pd.read_csv("./data/processed/rlps_2023_data_clean.csv")
        self.gemeentes = self.dataset['GEMEENTE'].unique().tolist()
        self.boxids = self.dataset['BOXID'].unique().tolist()

        self.dataset_loaded = True

        if 'CLUSTER' not in self.dataset.columns or self.force_clustering:
            warnings.warn("The dataset does not contain a column 'CLUSTER', or clustering was forced: "
                          "Clustering the dataset...")
            self._cluster_dataset()
        print("Dataset clustered... Load OK!")

    def _cluster_dataset(self):
        from clustering import cluster_and_align

        normalized_data_df, centroids_data_df = cluster_and_align(self.dataset, n_clusters=self.n_clusters)
        clustered_data_df = normalized_data_df[['GEMEENTE', 'BOXID', 'MONTH', 'CLUSTER']]

        if self.write_file:
            clustered_data_df.to_csv('./data/processed/clusters_rlps_2023.csv')

        if 'CLUSTER' in self.dataset.columns:  # Rewrite clusters
            print("Re-writing clusters...")
            self.dataset.drop(columns=['CLUSTER'], inplace=True)

        self.dataset = self.dataset.merge(clustered_data_df, on=['GEMEENTE', 'BOXID', 'MONTH'], how='inner')

        if self.write_file:
            print("Saving cluster file clusters...")
            self.dataset.to_csv('./data/processed/rlps_2023_data.csv', index=False)

    def get_dataset(self, gemeente=None, month=None, id=None)-> pd.DataFrame | None:
        query = self.dataset.copy()

        if gemeente is not None:
            if gemeente in self.gemeentes:
                query = query[query['GEMEENTE'] == gemeente].copy()
            else:
                warnings.warn(f"Gemeente {gemeente} not in database")
                return None

        if id is not None:
            if id in self.boxids:
                query = query[query['BOXID'] == id].copy()
            else:
                warnings.warn(f"Box id {id} not in database")
                return None

        if month is not None:
            if (1 <= month <= 12):
                query = query[query['MONTH'] == month].copy()
            else:
                raise ValueError("month must be between 1 and 12")

        required_columns = {'GEMEENTE', 'BOXID', 'CLUSTER', 'MONTH'}

        # Check if all required columns exist
        if not required_columns.issubset(query.columns):
            raise ValueError(f"Required columns {required_columns - set(query.columns)} not found in dataset")

        return query

    def get_gemeente_list(self) -> list:
        if not self.dataset_loaded:
            raise Exception("Dataset not loaded, run 'get_dataset'")
        return self.gemeentes



    def get_ids_list(self) -> list:
        if not self.dataset_loaded:
            raise Exception("Dataset not loaded, run 'get_dataset'")
        return self.boxids

    def add_clusters(self, clusters_df: pd.DataFrame):
        missing_columns = {'BOXID', 'GEMEENTE', 'CLUSTER', 'MONTH'} - set(clusters_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.dataset = self.dataset.merge(clusters_df, on=['GEMEENTE', 'BOXID', 'MONTH'], how='inner')
        self.dataset_clusterd = True




if __name__ == "__main__":
    dataset = FileDataset(n_clusters=4, force_clustering=True)
    print(dataset.get_dataset(gemeente='ENSCHEDE'))
