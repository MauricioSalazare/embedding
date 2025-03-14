from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

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

    def __init__(self):
        self.dataset_loaded = False
        self.load_dataset()

    def load_dataset(self):
        """This is a navie loading of the dataset. If it doesn't fit in memory, you are toasted"""
        self.clusters = pd.read_csv("./data/processed/rlps_clusters.csv", index_col=0)
        self.dataset = pd.read_csv("./data/processed/rlps_filtered.csv", index_col=0)
        self.dataset_clusters = self.dataset.join(self.clusters, how="inner")  # Merge clusters into df1
        self.dataset_loaded = True

    def get_dataset(self, gemeente=None, month=None)-> pd.DataFrame:
        return self.dataset_clusters.copy()

    def get_gemeente_list(self) -> list:
        if not self.dataset_loaded:
            raise Exception("Dataset not loaded, run 'get_dataset_fi'")

        raise NotImplementedError

    def get_ids_list(self) -> list:
        if not self.dataset_loaded:
            raise Exception("Dataset not loaded, run 'get_dataset_fi'")

        raise NotImplementedError

if __name__ == "__main__":
    dataset = FileDataset()

