"""
train_nn.py

Training script for the Feedforward Neural Network surrogate model.

The neural network is trained to approximate the relationship between the
design variables of the compliant-base robotic system and the maximum
outreach response obtained from simulation.

Inputs:
    kr      : robot stiffness [N/m]
    zeta_r  : robot damping ratio [-]
    A       : chirp amplitude [m]
    f0      : initial chirp frequency [Hz]
    f1      : final chirp frequency [Hz]
    offset  : command offset [m]

Target:
    ymax    : maximum total displacement [m]

Authors: Maria Paulina Pantoja Gavidia
         Carmen Natalia de León Bercián
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "data/compliant_base_dataset.csv"

INPUT_COLS = ["kr", "zeta_r", "A", "f0", "f1", "offset"]
TARGET_COL = "ymax"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 1e-3

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "surrogate_nn.pth")
X_SCALER_PATH = os.path.join(MODEL_DIR, "x_scaler.pkl")
Y_SCALER_PATH = os.path.join(MODEL_DIR, "y_scaler.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.npz")


# ============================================================
# Neural network model
# ============================================================

class SurrogateNN(nn.Module):
    """
    Feedforward Neural Network surrogate model.
    """

    def __init__(self, input_dim: int = 6) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        """
        return self.net(x)


# ============================================================
# Data preparation
# ============================================================

def load_and_prepare_data(
    csv_path: str = DATASET_PATH,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Load, clean, split, normalize, and convert the dataset to tensors.
    """

    dataset = pd.read_csv(csv_path)

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    X = dataset[INPUT_COLS].values
    y = dataset[[TARGET_COL]].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + val_size,
        random_state=random_state,
    )

    relative_val_size = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - relative_val_size,
        random_state=random_state,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler


# ============================================================
# Training
# ============================================================

def train_model() -> None:
    """
    Train the surrogate neural network and save the trained artifacts.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler = (
        load_and_prepare_data()
    )

    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = SurrogateNN(input_dim=X_train.shape[1])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):

        model.train()
        epoch_train_loss = 0.0

        for X_batch, y_batch in train_loader:

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss /= len(train_loader.dataset)

        model.eval()

        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val).item()

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Train Loss: {epoch_train_loss:.6f} "
                f"Val Loss: {val_loss:.6f}"
            )

    if best_model_state is None:
        raise RuntimeError("Training failed: best model state was not saved.")

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test).item()

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)

    np.savez(
        HISTORY_PATH,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
    )

    print("\nFinal results")
    print("-------------")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Model saved as: {MODEL_PATH}")
    print(f"Input scaler saved as: {X_SCALER_PATH}")
    print(f"Target scaler saved as: {Y_SCALER_PATH}")
    print(f"Training history saved as: {HISTORY_PATH}")


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    train_model()