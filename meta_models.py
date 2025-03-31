from abc import ABC, abstractmethod
import numpy as np
from sklearn.manifold import Isomap, TSNE
import umap

from src.models import Autoencoder, TorchDataset, VAE, vae_loss
from spheral import Sphere

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _check_numpy_array(array: np.ndarray):
    if not isinstance(array, np.ndarray):
        raise ValueError("X must be a numpy array array")

    if array.ndim != 2:
        raise ValueError("X must be a 2D-numpy array")

def _check_is_fitted(self):
    if not self.model_fitted:
        raise ValueError("Model must be trained before transform")


class DimensionReductionModel(ABC):

    @abstractmethod
    def fit(self, X: np.array):
        """Learns something from the data. e.g., parameters"""
        pass

    @abstractmethod
    def transform(self, X: np.array):
        """Applies the model to the data"""
        pass

    @abstractmethod
    def fit_transform(self, X: np.array):
        """Learns something from data and then transforms it."""
        pass

    @abstractmethod
    def decode(self, X: np.array):
        """Brings back the data from lower dimension to higher dimension"""
        pass

    @abstractmethod
    def score(self):
        """Compute a scoring system for the performance of the model"""
        pass


class AutoencoderModel(DimensionReductionModel):
    name = 'autoencoder'

    def __init__(
        self,
        latent_dim: int = 3,
        epochs: int = 1_000,
        batch_size: int = 30,
        learning_rate=0.001,
        writer: SummaryWriter | None = None,
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.writer = writer

        self.model_fitted = False

    def fit(
        self,
        X: np.ndarray,
    ):
        _check_numpy_array(X)

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.lower_dimension = self.latent_dim
        self.high_dimension = self.n_features

        # Load data:
        torch_data = TorchDataset(X)
        dataloader = DataLoader(torch_data, batch_size=self.batch_size, shuffle=True)

        # Define model:
        self.model = Autoencoder(input_dim=self.n_features, latent_dim=self.latent_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for images in dataloader:
                self.optimizer.zero_grad()
                encoded, decoded = self.model(images)
                loss = self.criterion(decoded, images)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

            # Log the loss to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Training Loss", avg_loss, epoch)

        if self.writer is not None:
            self.writer.close()

        self.model_fitted = True

        return

    def transform(self, X: np.ndarray) -> np.ndarray:
        _check_numpy_array(X)
        _check_is_fitted(self)

        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)

        with torch.no_grad():
            latent_vectors = self.model.encoder(X_tensor)

        latent_vectors_array = latent_vectors.numpy()

        return latent_vectors_array

    def fit_transform(self, X: np.array) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def decode(self, X: np.ndarray) -> np.ndarray:
        _check_numpy_array(X)
        _check_is_fitted(self)

        if X.shape[1] != self.lower_dimension:
            raise ValueError(
                f"X has {X.shape[1]} lower dimension(s) and it must be {self.lower_dimension}."
            )

        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)

        with torch.no_grad():
            reconstructed = self.model.decoder(X_tensor)

        reconstructed_array = reconstructed.numpy()

        return reconstructed_array

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        _check_numpy_array(X)
        _check_is_fitted(self)

        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)

        with torch.no_grad():
            latent_vectors, decoded = self.model(X_tensor)

        decoded_array = decoded.numpy()

        return decoded_array

    def score(self):
        raise NotImplementedError


class VariationalAutoencoder(DimensionReductionModel):
    name = "vae"
    def __init__(
        self,
        latent_dim: int = 3,
        epochs: int = 1_000,
        batch_size: int = 30,
        learning_rate=0.001,
        writer: SummaryWriter | None = None,
        **kwargs,
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.writer = writer

        self.model_fitted = False

    def fit(
        self,
        X: np.ndarray,
    ):
        _check_numpy_array(X)

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.lower_dimension = self.latent_dim
        self.high_dimension = self.n_features

        # Load data:
        torch_data = TorchDataset(X)
        dataloader = DataLoader(torch_data, batch_size=self.batch_size, shuffle=True)

        # Define model:
        self.model = VAE(input_dim=self.n_features, latent_dim=self.latent_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            total_loss = 0
            for images in dataloader:
                self.optimizer.zero_grad()

                reconstructed_x, mu, logvar, _ = self.model(images)
                loss = vae_loss(reconstructed_x, images, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")

            # Log the loss to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Training Loss", total_loss, epoch)

        if self.writer is not None:
            self.writer.close()

        self.model_fitted = True

        return

    def transform(self, X: np.ndarray) -> np.ndarray:
        _check_numpy_array(X)
        _check_is_fitted(self)

        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)

        with torch.no_grad():
            decoded, _, _, latent_vectors = self.model(X_tensor)

        latent_vectors_array = latent_vectors.numpy()

        return latent_vectors_array

    def fit_transform(self, X: np.array) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """Receives low dimensional data and transform to high dimensions"""
        _check_numpy_array(X)
        _check_is_fitted(self)

        if X.shape[1] != self.lower_dimension:
            raise ValueError(
                f"X has {X.shape[1]} lower dimension(s) and it must be {self.lower_dimension}."
            )

        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)  # Low dimensional data

        with torch.no_grad():
            mu = self.model.mu_layer(X_tensor)
            logvar = self.model.logvar_layer(X_tensor)
            z = self.model.reparameterize(mu, logvar)  # Sampled latent vector
            decoded = self.model.decoder(z)

        reconstructed_array = decoded.numpy()

        return reconstructed_array

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        _check_numpy_array(X)
        _check_is_fitted(self)

        X_array = X.astype(np.float32)  # Ensure float32 type
        X_tensor = torch.from_numpy(X_array)

        with torch.no_grad():
            decoded, _, _, latent_vectors = self.model(X_tensor)

        decoded_array = decoded.numpy()

        return decoded_array

    def score(self):
        raise NotImplementedError


class IsomapModel(DimensionReductionModel):
    name = "isomap"
    def __init__(self, n_neighbors: int = 10, n_components: int = 3):
        self.name = "isomap"
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.model = Isomap(
            n_neighbors=self.n_neighbors, n_components=self.n_components
        )
        self.model_fitted = False

    def fit(self, X: np.array):
        _check_numpy_array(X)
        self.model.fit(X)
        self.model_fitted = True

    def transform(self, X: np.array):
        _check_numpy_array(X)
        _check_is_fitted(self)
        return self.model.transform(X)

    def fit_transform(self, X: np.array):
        _check_numpy_array(X)
        return self.model.fit_transform(X)

    def decode(self, X: np.array):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError


class UmapModel(DimensionReductionModel):
    name = "umap"
    def __init__(self, n_neighbors: int = 3, n_components: int = 3):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.model = umap.UMAP(
            n_neighbors=self.n_neighbors, n_components=self.n_components
        )
        self.model_fitted = False

    def fit(self, X: np.array):
        _check_numpy_array(X)
        self.model.fit(X)
        self.model_fitted = True

    def transform(self, X: np.array):
        _check_numpy_array(X)
        _check_is_fitted(self)
        return self.model.transform(X)

    def fit_transform(self, X: np.array):
        _check_numpy_array(X)
        return self.model.fit_transform(X)

    def decode(self, X: np.array):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError



class TSNEModel(DimensionReductionModel):
    name = "t-SNE"
    def __init__(self, n_components=3, perplexity=30, max_iter=1000):
        self.perplexity = perplexity
        self.n_components = n_components
        self.max_iter = max_iter
        self.model =  TSNE(n_components=self.n_components, perplexity=self.perplexity, n_iter=self.max_iter,)
        self.model_fitted = False

    def fit(self, X: np.array):
        _check_numpy_array(X)
        self.model.fit(X)
        self.model_fitted = True

    def transform(self, X: np.array):
        _check_numpy_array(X)
        _check_is_fitted(self)
        return self.model.transform(X)

    def fit_transform(self, X: np.array):
        _check_numpy_array(X)
        return self.model.fit_transform(X)

    def decode(self, X: np.array):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError


class SphereModel(DimensionReductionModel):
    name = "sphere"
    def __init__(self):
        self.model = Sphere()
        self.model_fitted = False

    def fit(self, X: np.array):
        _check_numpy_array(X)
        self.model.fit(X)
        self.model_fitted = True

    def transform(self, X: np.array):
        _check_numpy_array(X)
        _check_is_fitted(self)
        return self.model.transform(X)

    def fit_transform(self, X: np.array):
        _check_numpy_array(X)
        return self.model.fit_transform(X)

    def decode(self, X: np.array):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

