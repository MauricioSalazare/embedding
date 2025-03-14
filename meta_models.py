from abc import ABC, abstractmethod
import numpy as np

from src.models import Autoencoder, TorchDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class DimensionReductionModel(ABC):

    @abstractmethod
    def fit(self, X:np.array, latent_dim:int, **kwargs):
        """Learns something from the data. e.g., parameters"""
        pass

    @abstractmethod
    def transform(self, X:np.array):
        """Applies the model to the data"""
        pass

    @abstractmethod
    def fit_transform(self, X:np.array, latent_dim:int, **kwargs):
        """Learns something from data and then transforms it."""
        pass

    @abstractmethod
    def decode(self, X:np.array):
        """Brings back the data from lower dimension to higher dimension"""
        pass

    @abstractmethod
    def score(self):
        """Compute a scoring system for the performance of the model"""
        pass


class AutoencoderModel(DimensionReductionModel):
    def __init__(self):
        self.model_fitted = False

    def fit(self,
            X:np.ndarray,
            latent_dim: int=3,
            epochs:int=1_000,
            learning_rate=0.001,
            writer:SummaryWriter|None=None, **kwargs):

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array array")

        if X.ndim != 2:
            raise ValueError("X must be a 2D-numpy array")

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.lower_dimension = latent_dim
        self.high_dimension = self.n_features

        # Load data:
        torch_data = TorchDataset(X)
        dataloader = DataLoader(torch_data, batch_size=30, shuffle=True)

        # Define model:
        self.model = Autoencoder(input_dim=self.n_features, latent_dim=latent_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            epoch_loss = 0
            for images in dataloader:
                self.optimizer.zero_grad()
                encoded, decoded = self.model(images)
                loss = self.criterion(decoded, images)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # Log the loss to TensorBoard
            if writer is not None:
                writer.add_scalar("Training Loss", avg_loss, epoch)

        if writer is not None:
            writer.close()

        self.model_fitted = True

        return


    def transform(self, X:np.ndarray) -> np.ndarray:
        if not self.model_fitted:
            raise ValueError("Model must be trained before transform")


        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)


        with torch.no_grad():
            latent_vectors = self.model.encoder(X_tensor)

        latent_vectors_array = latent_vectors.numpy()

        return latent_vectors_array


    def fit_transform(self,
                      X:np.array,
                      latent_dim:int=3,
                      epochs:int=1_000,
                      learning_rate=0.001,
                      writer:SummaryWriter|None=None, **kwargs):

        self.fit(X, latent_dim, epochs=epochs, learning_rate=learning_rate, writer=writer, **kwargs)
        return self.transform(X)


    def decode(self, X:np.ndarray) -> np.ndarray:
        if not self.model_fitted:
            raise ValueError("Model must be trained before transform")

        if X.shape[1] != self.lower_dimension:
            raise ValueError(f"X has {X.shape[1]} lower dimension(s) and it must be {self.lower_dimension}.")

        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)

        # with torch.no_grad():
        #     latent_vectors, decoded = self.model(X_tensor)

        with torch.no_grad():
            reconstructed = self.model.decoder(X_tensor)

        reconstructed_array = reconstructed.numpy()

        return reconstructed_array

    def reconstruct(self, X:np.ndarray) -> np.ndarray:
        if not self.model_fitted:
            raise ValueError("Model must be trained before transform")

        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)

        with torch.no_grad():
            latent_vectors, decoded = self.model(X_tensor)

        decoded_array = decoded.numpy()

        return decoded_array

    def score(self):
        raise NotImplementedError











