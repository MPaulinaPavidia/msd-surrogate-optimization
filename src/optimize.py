"""
optimize.py

Optimization script for the trained Feedforward Neural Network surrogate model.

The trained neural network is used as a surrogate model to predict the maximum
outreach response of the compliant-base robotic system. A Differential Evolution
optimizer is then applied to find the design variables that produce a target
maximum displacement.

Design variables:
    kr      : robot stiffness [N/m]
    zeta_r  : robot damping ratio [-]
    A       : chirp amplitude [m]
    f0      : initial chirp frequency [Hz]
    f1      : final chirp frequency [Hz]
    offset  : command offset [m]

Objective:
    Minimize (y_pred - Y_TARGET)^2

Authors: Maria Paulina Pantoja Gavidia
         Carmen Natalia de León Bercián
            Burak Turhan
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import torch
import torch.nn as nn

from scipy.optimize import differential_evolution


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "surrogate_nn.pth")
X_SCALER_PATH = os.path.join(MODEL_DIR, "x_scaler.pkl")
Y_SCALER_PATH = os.path.join(MODEL_DIR, "y_scaler.pkl")

RESULTS_DIR = "results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "optimization_results.txt")

Y_TARGET = 0.70
RANDOM_SEED = 42

BOUNDS = [
    (100.0, 2500.0),  # kr [N/m]
    (0.2, 1.1),       # zeta_r [-]
    (0.01, 0.08),     # A [m]
    (0.1, 1.0),       # f0 [Hz]
    (2.0, 8.0),       # f1 [Hz]
    (0.50, 0.70),     # offset [m]
]

PARAMETER_NAMES = [
    "kr",
    "zeta_r",
    "A",
    "f0",
    "f1",
    "offset",
]


# ============================================================
# Neural network model
# ============================================================

class SurrogateNN(nn.Module):
    """
    Feedforward Neural Network surrogate model.

    The architecture must match the one used during training.
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
# Model utilities
# ============================================================

def load_surrogate_model() -> tuple[SurrogateNN, object, object]:
    """
    Load the trained neural network and the input/output scalers.

    Returns
    -------
    tuple
        Trained model, input scaler and output scaler.
    """

    model = SurrogateNN(input_dim=6)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
        )
    )
    model.eval()

    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    return model, x_scaler, y_scaler


def predict_ymax(
    x: np.ndarray,
    model: SurrogateNN,
    x_scaler,
    y_scaler,
) -> float:
    """
    Predict ymax using the trained surrogate model.

    Parameters
    ----------
    x : numpy.ndarray
        Design vector [kr, zeta_r, A, f0, f1, offset].
    model : SurrogateNN
        Trained neural network.
    x_scaler : StandardScaler
        Input scaler fitted during training.
    y_scaler : StandardScaler
        Target scaler fitted during training.

    Returns
    -------
    float
        Predicted maximum displacement [m].
    """

    x = np.asarray(x, dtype=float).reshape(1, -1)
    x_scaled = x_scaler.transform(x)

    x_tensor = torch.tensor(
        x_scaled,
        dtype=torch.float32,
    )

    with torch.no_grad():
        y_scaled = model(x_tensor).numpy()

    y_pred = y_scaler.inverse_transform(y_scaled)

    return float(y_pred[0, 0])


def objective_function(
    x: np.ndarray,
    model: SurrogateNN,
    x_scaler,
    y_scaler,
) -> float:
    """
    Objective function minimized by Differential Evolution.
    """

    y_pred = predict_ymax(
        x,
        model,
        x_scaler,
        y_scaler,
    )

    return float((y_pred - Y_TARGET) ** 2)


# ============================================================
# Optimization
# ============================================================

def run_optimization() -> None:
    """
    Run Differential Evolution optimization and save the results.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, x_scaler, y_scaler = load_surrogate_model()

    result = differential_evolution(
        func=objective_function,
        bounds=BOUNDS,
        args=(model, x_scaler, y_scaler),
        seed=RANDOM_SEED,
        maxiter=300,
        popsize=20,
        tol=1e-7,
        polish=True,
    )

    x_opt = result.x

    y_opt = predict_ymax(
        x_opt,
        model,
        x_scaler,
        y_scaler,
    )

    abs_error = abs(y_opt - Y_TARGET)

    print("\nOptimization result")
    print("-------------------")
    print(f"Target ymax     : {Y_TARGET:.6f} m")
    print(f"Predicted ymax  : {y_opt:.6f} m")
    print(f"Absolute error  : {abs_error:.6f} m")

    print("\nOptimal parameters")
    print("------------------")
    for name, value in zip(PARAMETER_NAMES, x_opt):
        unit = " N/m" if name == "kr" else " m" if name in ["A", "offset"] else " Hz" if name in ["f0", "f1"] else ""
        print(f"{name:8s}= {value:.6f}{unit}")

    print("\nOptimizer information")
    print("---------------------")
    print(f"Objective value : {result.fun:.6e}")
    print(f"Iterations      : {result.nit}")
    print(f"Function evals. : {result.nfev}")
    print(f"Success         : {result.success}")
    print(f"Message         : {result.message}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("Optimization result\n")
        file.write("-------------------\n")
        file.write(f"Target ymax     : {Y_TARGET:.6f} m\n")
        file.write(f"Predicted ymax  : {y_opt:.6f} m\n")
        file.write(f"Absolute error  : {abs_error:.6f} m\n\n")

        file.write("Optimal parameters\n")
        file.write("------------------\n")
        for name, value in zip(PARAMETER_NAMES, x_opt):
            file.write(f"{name} = {value:.6f}\n")

        file.write("\nOptimizer information\n")
        file.write("---------------------\n")
        file.write(f"Objective value : {result.fun:.6e}\n")
        file.write(f"Iterations      : {result.nit}\n")
        file.write(f"Function evals. : {result.nfev}\n")
        file.write(f"Success         : {result.success}\n")
        file.write(f"Message         : {result.message}\n")

    print(f"\nOptimization results saved as: {RESULTS_PATH}")


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    run_optimization()