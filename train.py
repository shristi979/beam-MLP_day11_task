import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np

from model import BeamMLP

# Set random seeds for reproducibility
torch.manual_seed(72)
np.random.seed(72)

# Load normalized datasets
train_data = pd.read_csv("train_normalized.csv")
val_data = pd.read_csv("validation_normalized.csv")

# Select input features
input_columns = ["x", "L", "E", "I", "P"]

# Convert inputs and outputs into NumPy arrays
X_train = train_data[input_columns].values
y_train = train_data[["y"]].values

X_val = val_data[input_columns].values
y_val = val_data[["y"]].values

print("Training input shape:", X_train.shape)
print("Training output shape:", y_train.shape)

print("Validation input shape:", X_val.shape)
print("Validation output shape:", y_val.shape)

# Convert NumPy arrays into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

print("Training tensor shapes:")
print("X:", X_train_tensor.shape)
print("y:", y_train_tensor.shape)

print("\nValidation tensor shapes:")
print("X:", X_val_tensor.shape)
print("y:", y_val_tensor.shape)

# Combine inputs and targets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Define batch size
batch_size = 128

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("\nNumber of training batches:", len(train_loader))
print("Number of validation batches:", len(val_loader))

# Initialize the neural network
model = BeamMLP()

# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay = 1e-5
)

print("Model initialized successfully!")
print("Loss function:", criterion)
print("Optimizer:", optimizer)

# Number of training epochs
epochs = 300

# Store losses for plotting later
train_losses = []
val_losses = []

# Track the best validation loss
best_val_loss = float("inf")
best_epoch = 0

# Training loop
for epoch in range(epochs):

    # -------------------------
    # TRAINING
    # -------------------------

    model.train()

    running_train_loss = 0.0

    for inputs, targets in train_loader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        predictions = model(inputs)

        # Calculate loss
        loss = criterion(predictions, targets)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)

    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            predictions = model(inputs)

            loss = criterion(predictions, targets)

            running_val_loss += loss.item() * inputs.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    # Store losses
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if epoch_val_loss < best_val_loss:

        best_val_loss = epoch_val_loss
        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss
            },
            "best_beam_mlp.pth"
        )

    # Print progress
    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {epoch_train_loss:.6f} "
            f"Val Loss: {epoch_val_loss:.6f}"
        )

print("\nTraining completed!")
print(f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}")

# Save training history
history = pd.DataFrame({
    "epoch": range(1, epochs + 1),
    "train_loss": train_losses,
    "val_loss": val_losses
})

history.to_csv("training_history.csv", index=False)

print("Training history saved successfully!")


