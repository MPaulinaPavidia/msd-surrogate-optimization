"""
evaluate_nn.py

Evaluation script for the trained Feedforward Neural Network surrogate model.

The script evaluates the trained neural network on the test dataset and
generates quantitative metrics and diagnostic plots.

Metrics:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Coefficient of determination (R²)
    - Worst-case prediction error

Authors: Maria Paulina Pantoja Gavidia
         Carmen Natalia de León Bercián
         Burak Turhan
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "data/compliant_base_dataset.csv"

INPUT_COLS = ["kr", "zeta_r", "A", "f0", "f1", "offset"]
TARGET_COL = "ymax"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "surrogate_nn.pth")
Y_SCALER_PATH = os.path.join(MODEL_DIR, "y_scaler.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.npz")

FIG_DIR = "figures/nn_evaluation"
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_DATA = "#102c53"
COLOR_REF = "#d97a1f"
COLOR_RES = "#6f4aa2"
COLOR_HIST = "#7fa87f"
COLOR_TRAIN = "#102c53"
COLOR_VAL = "#d97a1f"


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

def load_and_prepare_data():
    """
    Load, clean, split and normalize the dataset.

    The same split and normalization logic used during training is applied
    to reproduce the test set consistently.

    Returns
    -------
    tuple
        X_test, y_test, y_scaler
    """

    dataset = pd.read_csv(DATASET_PATH)

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    X = dataset[INPUT_COLS].values
    y = dataset[[TARGET_COL]].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    relative_val_size = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

    _, X_test, _, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - relative_val_size,
        random_state=RANDOM_STATE,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_train)
    y_scaler.fit(y_train)

    X_test = x_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_test, y_test, y_scaler


# ============================================================
# Evaluation
# ============================================================

def evaluate_model() -> None:
    """
    Evaluate the trained neural network and save diagnostic plots.
    """

    X_test, y_test, y_scaler = load_and_prepare_data()

    model = SurrogateNN(input_dim=X_test.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()

    y_true = y_scaler.inverse_transform(y_test.numpy()).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

    residuals = y_true - y_pred
    errors = np.abs(residuals)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    idx_worst = int(np.argmax(errors))

    print("\nEvaluation results")
    print("------------------")
    print(f"MAE        = {mae:.6f} m")
    print(f"RMSE       = {rmse:.6f} m")
    print(f"R²         = {r2:.6f}")
    print(f"Mean res.  = {np.mean(residuals):.6f} m")
    print(f"Std res.   = {np.std(residuals):.6f} m")
    print(f"Max error  = {np.max(errors):.6f} m")

    print("\nWorst case")
    print("----------")
    print(f"True  = {y_true[idx_worst]:.6f} m")
    print(f"Pred  = {y_pred[idx_worst]:.6f} m")
    print(f"Error = {errors[idx_worst]:.6f} m")

    metrics_path = os.path.join(FIG_DIR, "evaluation_metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("Evaluation results\n")
        file.write("------------------\n")
        file.write(f"MAE        = {mae:.6f} m\n")
        file.write(f"RMSE       = {rmse:.6f} m\n")
        file.write(f"R²         = {r2:.6f}\n")
        file.write(f"Mean res.  = {np.mean(residuals):.6f} m\n")
        file.write(f"Std res.   = {np.std(residuals):.6f} m\n")
        file.write(f"Max error  = {np.max(errors):.6f} m\n\n")
        file.write("Worst case\n")
        file.write("----------\n")
        file.write(f"True  = {y_true[idx_worst]:.6f} m\n")
        file.write(f"Pred  = {y_pred[idx_worst]:.6f} m\n")
        file.write(f"Error = {errors[idx_worst]:.6f} m\n")

    # True vs predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75, color=COLOR_DATA)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color=COLOR_REF,
        label="Ideal prediction",
    )

    plt.xlabel(r"True $y_{\max}$ [m]")
    plt.ylabel(r"Predicted $y_{\max}$ [m]")
    plt.title("NN Surrogate: True vs Predicted Response")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "true_vs_predicted.png"))
    plt.close()

    # Residual plot
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, residuals, s=18, alpha=0.75, color=COLOR_RES)
    plt.axhline(0.0, linestyle="--", color=COLOR_REF)

    plt.xlabel(r"True $y_{\max}$ [m]")
    plt.ylabel(r"Residual $y_{true} - y_{pred}$ [m]")
    plt.title("Prediction Residuals")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(FIG_DIR, "residuals.png"))
    plt.close()

    # Residual histogram
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30, color=COLOR_HIST, edgecolor="black")

    plt.xlabel(r"Residual $y_{true} - y_{pred}$ [m]")
    plt.ylabel("Count")
    plt.title("Distribution of Prediction Residuals")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(FIG_DIR, "residual_histogram.png"))
    plt.close()

    # Training history plot
    if os.path.exists(HISTORY_PATH):
        history = np.load(HISTORY_PATH)

        train_key = "train_losses" if "train_losses" in history else "train_loss"
        val_key = "val_losses" if "val_losses" in history else "val_loss"

        plt.figure(figsize=(7, 5))
        plt.plot(history[train_key], color=COLOR_TRAIN, label="Training loss")
        plt.plot(history[val_key], color=COLOR_VAL, label="Validation loss")

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Validation Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.savefig(os.path.join(FIG_DIR, "training_history.png"))
        plt.close()

    print(f"\nEvaluation figures saved in: {FIG_DIR}")


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    evaluate_model()