"""
simulation.py

Dynamic simulation of a two-degree-of-freedom compliant robotic system.

The system consists of:
    - A compliant base mass-spring-damper subsystem
    - A robot mass-spring-damper subsystem

The robot is excited through a chirp position command and the maximum
end-effector displacement is extracted as performance metric.

Authors: Maria Paulina Pantoja Gavidia
         Carmen Natalia de León Bercián
         Burak Turhan
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import chirp


# ============================================================
# Fixed system parameters
# ============================================================

MB = 10.0          # Base mass [kg]
MR = 10.0          # Robot mass [kg]

KB = 1000.0        # Base stiffness [N/m]
ZETA_B = 0.30      # Base damping ratio [-]

DB = 2.0 * ZETA_B * np.sqrt(KB * MB)

DEFAULT_T_SIM = 25.0
DEFAULT_N_POINTS = 3000


# ============================================================
# Command signal
# ============================================================

def chirp_command(
    t,
    amplitude,
    f0,
    f1,
    offset,
    t_sim
):
    """
    Generate chirp position command.

    Parameters
    ----------
    t : float or ndarray
        Time.
    amplitude : float
        Chirp amplitude [m].
    f0 : float
        Initial frequency [Hz].
    f1 : float
        Final frequency [Hz].
    offset : float
        Position offset [m].
    t_sim : float
        Total chirp duration [s].

    Returns
    -------
    float or ndarray
        Commanded position.
    """

    return offset + amplitude * chirp(
        t,
        f0=f0,
        f1=f1,
        t1=t_sim,
        method="linear"
    )


# ============================================================
# Simulation function
# ============================================================

def simulate_case(
    kr,
    zeta_r,
    amplitude,
    f0,
    f1,
    offset,
    t_sim=DEFAULT_T_SIM,
    n_points=DEFAULT_N_POINTS
):
    """
    Simulate the compliant robotic system.

    Parameters
    ----------
    kr : float
        Robot stiffness [N/m].
    zeta_r : float
        Robot damping ratio [-].
    amplitude : float
        Chirp amplitude [m].
    f0 : float
        Initial chirp frequency [Hz].
    f1 : float
        Final chirp frequency [Hz].
    offset : float
        Position offset [m].
    t_sim : float, optional
        Simulation time [s].
    n_points : int, optional
        Number of time samples.

    Returns
    -------
    dict
        Simulation metrics.
    """

    dr = 2.0 * zeta_r * np.sqrt(kr * MR)

    t_eval = np.linspace(0.0, t_sim, n_points)

    def dynamics(t, z):

        xb, xb_dot, xrd, xrd_dot = z

        xcmd = chirp_command(
            t,
            amplitude,
            f0,
            f1,
            offset,
            t_sim
        )

        xb_ddot = (
            -DB * xb_dot
            -KB * xb
            + kr * (xrd - xcmd)
            + dr * xrd_dot
        ) / MB

        xrd_ddot = (
            -dr * xrd_dot
            -kr * (xrd - xcmd)
        ) / MR - xb_ddot

        return [xb_dot, xb_ddot, xrd_dot, xrd_ddot]

    z0 = [
        0.0,
        0.0,
        chirp_command(
            0.0,
            amplitude,
            f0,
            f1,
            offset,
            t_sim
        ),
        0.0
    ]

    sol = solve_ivp(
        dynamics,
        [0.0, t_sim],
        z0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    xb = sol.y[0]
    xrd = sol.y[2]

    xcmd = chirp_command(
        sol.t,
        amplitude,
        f0,
        f1,
        offset,
        t_sim
    )

    y = xb + xrd
    xref = xb + xcmd

    idx_max = np.argmax(y)

    return {
        "ymax": float(y[idx_max]),
        "t_ymax": float(sol.t[idx_max]),
        "xb_max": float(np.max(xb)),
        "xb_min": float(np.min(xb)),
        "xrd_max": float(np.max(xrd)),
        "xref_max": float(np.max(xref)),
        "dr": float(dr)
    }