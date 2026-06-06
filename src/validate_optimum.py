"""
validate_optimum.py

Physical validation of the optimal design obtained with the trained
neural-network surrogate and Differential Evolution optimizer.

The optimized design variables are re-simulated using the original dynamic
model. The neural-network prediction is then compared against the direct
numerical simulation obtained with solve_ivp.

An optional mass-spring animation is displayed at the end of the script.
The animation is not saved.

Authors: Maria Paulina Pantoja Gavidia
         Carmen Natalia de León Bercián
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.signal import chirp

from simulation import MB, MR, KB, DB, T, T_EVAL


# ============================================================
# Plot configuration
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

FIG_DIR = "figures/optimal_solution"
os.makedirs(FIG_DIR, exist_ok=True)

COL_XRD = "#2ca02c"
COL_XB = "#ff7f0e"
COL_Y = "#9467bd"
COL_XCMD = "#1f77b4"
COL_REF = "#102c53"


# ============================================================
# Optimal design variables
# ============================================================

kr = 711.965256
zeta_r = 0.385453
A = 0.013616
f0 = 0.206698
f1 = 5.750233
offset = 0.677003

nn_predicted_ymax = 0.700001


# ============================================================
# Dynamic simulation
# ============================================================

def xrd_cmd(t: float | np.ndarray) -> float | np.ndarray:
    """
    Optimized linear chirp command.
    """

    return offset + A * chirp(
        t,
        f0=f0,
        f1=f1,
        t1=T,
        method="linear",
    )


def dynamics(_t: float, z: np.ndarray) -> list[float]:
    """
    State-space dynamics of the compliant-base robotic system.
    """

    xb, xb_dot, xrd, xrd_dot = z

    dr = 2.0 * zeta_r * np.sqrt(kr * MR)
    xcmd = xrd_cmd(_t)

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


def run_validation():
    """
    Run the physical simulation of the optimal design.

    Returns
    -------
    tuple
        Time vector and simulated response variables.
    """

    initial_state = [
        0.0,
        0.0,
        xrd_cmd(0.0),
        0.0,
    ]

    solution = solve_ivp(
        dynamics,
        [0.0, T],
        initial_state,
        t_eval=T_EVAL,
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(f"Simulation failed: {solution.message}")

    t = solution.t
    xb = solution.y[0]
    xrd = solution.y[2]
    xcmd = xrd_cmd(t)

    y = xb + xrd
    xref = xb + xcmd

    return t, xb, xrd, xcmd, y, xref


# ============================================================
# Results and figures
# ============================================================

def save_validation_results(
    t: np.ndarray,
    xb: np.ndarray,
    xrd: np.ndarray,
    xcmd: np.ndarray,
    y: np.ndarray,
    xref: np.ndarray,
) -> None:
    """
    Print and save validation metrics and response plots.
    """

    simulated_ymax = float(np.max(y))
    t_ymax = float(t[np.argmax(y)])
    difference = abs(simulated_ymax - nn_predicted_ymax)

    print("\nOptimal solution validation")
    print("---------------------------")
    print(f"NN predicted ymax = {nn_predicted_ymax:.6f} m")
    print(f"Simulated ymax    = {simulated_ymax:.6f} m")
    print(f"Difference        = {difference:.6f} m")
    print(f"t(ymax)           = {t_ymax:.6f} s")
    print(f"Max |xb|          = {np.max(np.abs(xb)):.6f} m")
    print(f"Max xrd           = {np.max(xrd):.6f} m")
    print(f"Max xref          = {np.max(xref):.6f} m")

    metrics_path = os.path.join(FIG_DIR, "optimal_solution_metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("Optimal solution validation\n")
        file.write("---------------------------\n")
        file.write(f"kr = {kr:.6f} N/m\n")
        file.write(f"zeta_r = {zeta_r:.6f}\n")
        file.write(f"A = {A:.6f} m\n")
        file.write(f"f0 = {f0:.6f} Hz\n")
        file.write(f"f1 = {f1:.6f} Hz\n")
        file.write(f"offset = {offset:.6f} m\n\n")
        file.write(f"NN predicted ymax = {nn_predicted_ymax:.6f} m\n")
        file.write(f"Simulated ymax    = {simulated_ymax:.6f} m\n")
        file.write(f"Difference        = {difference:.6f} m\n")
        file.write(f"t(ymax)           = {t_ymax:.6f} s\n")
        file.write(f"Max |xb|          = {np.max(np.abs(xb)):.6f} m\n")
        file.write(f"Max xrd           = {np.max(xrd):.6f} m\n")
        file.write(f"Max xref          = {np.max(xref):.6f} m\n")

    plt.figure(figsize=(8, 5))
    plt.plot(t, xb, color=COL_XB, linewidth=2.0, label=r"$x_b$")
    plt.plot(t, xrd, color=COL_XRD, linewidth=2.0, label=r"$x_{rd}$")
    plt.plot(t, y, color=COL_Y, linewidth=2.0, label=r"$y=x_b+x_{rd}$")
    plt.plot(t, xcmd, color=COL_XCMD, linestyle="--", linewidth=2.0, label=r"$x_{cmd}$")
    plt.axhline(nn_predicted_ymax, color=COL_REF, linestyle=":", linewidth=2.0, label=r"NN predicted $y_{\max}$")

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Optimal Solution Physical Validation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "optimal_solution_response.png"))
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(t, y, color=COL_Y, linewidth=2.0, label=r"Simulated $y$")
    plt.axhline(nn_predicted_ymax, color=COL_REF, linestyle=":", linewidth=2.0, label=r"NN predicted $y_{\max}$")
    plt.scatter(t_ymax, simulated_ymax, color=COL_REF, zorder=3, label=r"Simulated $y_{\max}$")

    plt.xlabel("Time [s]")
    plt.ylabel(r"$y$ [m]")
    plt.title("Maximum Outreach Validation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "maximum_outreach_validation.png"))
    plt.close()

    print(f"\nValidation results saved in: {FIG_DIR}")


# ============================================================
# Animation
# ============================================================

def spring_points(
    x_start: float,
    x_end: float,
    y0: float = 0.0,
    amp: float = 0.025,
    coils: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate points to draw a spring between two horizontal positions.
    """

    xs = np.linspace(x_start, x_end, 2 * coils + 2)
    ys = np.zeros_like(xs) + y0

    if len(xs) > 2:
        ys[1:-1:2] = amp
        ys[2:-1:2] = -amp

    return xs, ys


def animate_system(
    t: np.ndarray,
    xb: np.ndarray,
    xrd: np.ndarray,
    xcmd: np.ndarray,
    y: np.ndarray,
    xref: np.ndarray,
) -> None:
    """
    Display a mass-spring animation of the optimal system response.

    The animation is shown only and is not saved.
    """

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.set_xlim(-0.15, 0.85)
    ax.set_ylim(-0.18, 0.18)
    ax.set_xlabel("Position [m]")
    ax.set_yticks([])
    ax.set_title("Optimized Mass-Spring-Damper System Response")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)

    ax.axvline(
        0.0,
        color="black",
        linewidth=2.5,
        label="Ground",
    )

    cmd_line = ax.axvline(
        xref[0],
        color=COL_XCMD,
        linestyle="--",
        linewidth=2.0,
        label=r"$x_b+x_{cmd}$",
    )

    ax.axvline(
        nn_predicted_ymax,
        color=COL_REF,
        linestyle=":",
        linewidth=2.0,
        label=r"NN predicted $y_{\max}$",
    )

    base_mass, = ax.plot(
        [],
        [],
        "s",
        color=COL_XB,
        markersize=18,
        label="Base mass",
    )

    robot_mass, = ax.plot(
        [],
        [],
        "s",
        color=COL_Y,
        markersize=18,
        label="Robot mass",
    )

    base_spring, = ax.plot([], [], color=COL_XB, linewidth=2.0)
    robot_spring, = ax.plot([], [], color=COL_XRD, linewidth=2.0)

    time_text = ax.text(0.02, 0.88, "", transform=ax.transAxes)
    state_text = ax.text(0.02, 0.08, "", transform=ax.transAxes)

    ax.legend(loc="upper right", frameon=True)

    frame_step = max(1, len(t) // 400)
    frames = range(0, len(t), frame_step)

    def update(i: int):
        xb_i = xb[i]
        y_i = y[i]
        xref_i = xref[i]

        cmd_line.set_xdata([xref_i, xref_i])

        base_mass.set_data([xb_i], [0.0])
        robot_mass.set_data([y_i], [0.0])

        xs_base, ys_base = spring_points(
            0.0,
            xb_i,
            amp=0.018,
            coils=5,
        )

        xs_robot, ys_robot = spring_points(
            xb_i,
            y_i,
            amp=0.025,
            coils=8,
        )

        base_spring.set_data(xs_base, ys_base)
        robot_spring.set_data(xs_robot, ys_robot)

        time_text.set_text(
            f"t = {t[i]:.2f} s | Optimized chirp input"
        )

        state_text.set_text(
            rf"$x_b$ = {xb_i:.3f} m   |   "
            rf"$x_{{rd}}$ = {xrd[i]:.3f} m   |   "
            rf"$y$ = {y_i:.3f} m"
        )

        return (
            cmd_line,
            base_mass,
            robot_mass,
            base_spring,
            robot_spring,
            time_text,
            state_text,
        )

    FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=40,
        blit=True,
    )

    plt.show()


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    """
    Run physical validation and display the system animation.
    """

    t, xb, xrd, xcmd, y, xref = run_validation()

    save_validation_results(
        t,
        xb,
        xrd,
        xcmd,
        y,
        xref,
    )

    animate_system(
        t,
        xb,
        xrd,
        xcmd,
        y,
        xref,
    )


if __name__ == "__main__":
    main()