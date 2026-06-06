"""
generate_dataset.py

Dataset generation for the compliant robotic system.

The script samples the design space using random combinations of:

    - Robot stiffness (kr)
    - Robot damping ratio (zeta_r)
    - Chirp amplitude (A)
    - Initial frequency (f0)
    - Final frequency (f1)
    - Position offset

For each sampled design, the dynamic simulation is executed and
the maximum displacement ymax is stored as the target variable.

Authors: Maria Paulina Pantoja Gavidia
         Carmen Natalia de León Bercián
         Burak Turhan
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from simulation import simulate_case


# ============================================================
# Dataset configuration
# ============================================================

N_SAMPLES = 2000
RANDOM_SEED = 42

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "compliant_base_dataset.csv"
)

np.random.seed(RANDOM_SEED)


# ============================================================
# Design-space bounds
# ============================================================

BOUNDS = {
    "kr": (100.0, 2500.0),
    "zeta_r": (0.20, 1.10),
    "A": (0.01, 0.08),
    "f0": (0.10, 1.00),
    "f1": (2.00, 8.00),
    "offset": (0.50, 0.70)
}


# ============================================================
# Sampling
# ============================================================

def sample_design():
    """
    Generate a random design vector.

    Returns
    -------
    tuple
        (kr, zeta_r, A, f0, f1, offset)
    """

    return (
        np.random.uniform(*BOUNDS["kr"]),
        np.random.uniform(*BOUNDS["zeta_r"]),
        np.random.uniform(*BOUNDS["A"]),
        np.random.uniform(*BOUNDS["f0"]),
        np.random.uniform(*BOUNDS["f1"]),
        np.random.uniform(*BOUNDS["offset"])
    )


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(n_samples):
    """
    Generate simulation dataset.

    Parameters
    ----------
    n_samples : int
        Number of simulations.

    Returns
    -------
    pandas.DataFrame
        Generated dataset.
    """

    records = []

    for _ in tqdm(range(n_samples), desc="Generating dataset"):

        kr, zeta_r, A, f0, f1, offset = sample_design()

        results = simulate_case(
            kr=kr,
            zeta_r=zeta_r,
            amplitude=A,
            f0=f0,
            f1=f1,
            offset=offset
        )

        records.append({
            "kr": kr,
            "zeta_r": zeta_r,
            "A": A,
            "f0": f0,
            "f1": f1,
            "offset": offset,
            "ymax": results["ymax"]
        })

    return pd.DataFrame(records)


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = generate_dataset(N_SAMPLES)

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset generation completed")
    print("--------------------------------")
    print(f"Samples       : {len(dataset)}")
    print(f"Output file   : {OUTPUT_FILE}")
    print(f"Mean ymax     : {dataset['ymax'].mean():.6f} m")
    print(f"Maximum ymax  : {dataset['ymax'].max():.6f} m")


if __name__ == "__main__":
    main()