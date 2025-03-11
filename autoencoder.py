import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.use("MacOSX")

import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import os



#%%

# Initialize TensorBoard writer
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"runs/autoencoder_{timestamp}"
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
model_path = os.path.join(model_dir, f"autoencoder_{timestamp}.pth")

writer = SummaryWriter(log_dir)  # ✅ Unique log directory for each run

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Initialize model, loss function, and optimizer
input_dim = 28 * 28  # MNIST images are 28x28
latent_dim = 3  # We want a 3D representation
model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with TensorBoard logging
epochs = 1000
for epoch in range(epochs):
    epoch_loss = 0
    for images, _ in dataloader:
        optimizer.zero_grad()
        encoded, decoded = model(images)
        loss = criterion(decoded, images)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Log the loss to TensorBoard
    writer.add_scalar("Training Loss", avg_loss, epoch)

torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Close the TensorBoard writer
writer.close()
