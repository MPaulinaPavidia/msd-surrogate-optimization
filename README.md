# msd-surrogate-optimization

Neural-network-based surrogate modeling and optimization of a two-degree-of-freedom Mass-Spring-Damper (MSD) robotic system.

=====================================================================
PROJECT OVERVIEW
=====================================================================

This project investigates the dynamic behavior of a compliant robotic
platform modeled as a two-degree-of-freedom mass-spring-damper system.

A Feedforward Neural Network (FFNN) is trained as a surrogate model to
approximate the maximum displacement response of the system, enabling
efficient design optimization without repeatedly solving the full
dynamic simulation.

The workflow combines:

- Physics-based dynamic simulation
- Dataset generation through parameter sampling
- Feedforward Neural Network surrogate modeling
- Differential Evolution optimization
- Physical validation of the optimal solution

=====================================================================
SYSTEM DESCRIPTION
=====================================================================

The robotic platform is represented by:

- Compliant base subsystem
- Robot subsystem
- Spring-damper coupling between both masses
- Chirp excitation input

The total displacement of the robot is defined as:

y = xb + xrd

where:

xb  = base displacement
xrd = robot relative displacement

The optimization objective is to maximize the robot motion by exploiting
the compliant dynamics of the system.

=====================================================================
REPOSITORY STRUCTURE
=====================================================================

.
│
├── simulation.py
├── generate_dataset.py
├── train_nn.py
├── evaluate_nn.py
├── optimize.py
├── validate_optimum.py
├── requirements.txt
│
├── data/
├── models/

=====================================================================
INSTALLATION
=====================================================================

Install the required packages:

pip install -r requirements.txt

=====================================================================
WORKFLOW
=====================================================================

1. Generate Dataset
-------------------

Generate simulation samples covering the design space.

Command:

python generate_dataset.py

Output:

data/compliant_base_dataset.csv

---------------------------------------------------------------------

2. Train Neural Network
-----------------------

Train the Feedforward Neural Network surrogate model.

Command:

python train_nn.py

Outputs:

models/surrogate_nn.pth
models/x_scaler.pkl
models/y_scaler.pkl
models/training_history.npz

---------------------------------------------------------------------

3. Evaluate Surrogate Model
---------------------------

Evaluate the prediction capability of the trained network.

Command:

python evaluate_nn.py

Outputs:

figures/nn_evaluation/

Including:

- True vs Predicted response
- Residual analysis
- Residual histogram
- Training history

---------------------------------------------------------------------

4. Optimize Design Variables
----------------------------

Perform design optimization using Differential Evolution.

Command:

python optimize.py

Output:

results/optimization_results.txt

---------------------------------------------------------------------

5. Validate Optimal Solution
----------------------------

Validate the optimized design using the original physics-based
simulation.

Command:

python validate_optimum.py

Outputs:

figures/optimal_solution/

Including:

- Dynamic response
- Maximum outreach validation
- Physical validation metrics
- Interactive mass-spring animation

=====================================================================
DESIGN VARIABLES
=====================================================================

kr      : Robot stiffness [N/m]
zeta_r  : Robot damping ratio [-]
A       : Chirp amplitude [m]
f0      : Initial chirp frequency [Hz]
f1      : Final chirp frequency [Hz]
offset  : Position offset [m]

=====================================================================
NEURAL NETWORK ARCHITECTURE
=====================================================================

Input Layer (6)
      ↓
Hidden Layer (32) + ReLU
      ↓
Hidden Layer (32) + ReLU
      ↓
Output Layer (1)

Loss Function:
Mean Squared Error (MSE)

Optimizer:
Adam

=====================================================================
OPTIMIZATION METHOD
=====================================================================

The trained surrogate model is coupled with Differential Evolution to
identify the design variables that achieve a target maximum displacement
while avoiding repeated evaluations of the full dynamic simulation.

=====================================================================
AUTHORS
=====================================================================

Maria Paulina Pantoja Gavidia
Carmen Natalia de León Bercián
Burak Turhan

Politecnico di Milano
Machine Learning for Mechanical Systems
2025/2026
