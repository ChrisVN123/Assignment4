import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 0.9            # State transition coefficient
b = 1              # Bias term
sigma1 = 1         # Process noise standard deviation
X0 = 5             # Initial state
n = 100            # Time steps
num_realizations = 5  # Number of trajectories

np.random.seed(2025)  # For reproducibility

# Container to hold all simulated trajectories
trajectories = []

# Generate 5 independent trajectories
for _ in range(num_realizations):
    X = np.zeros(n)
    X[0] = X0
    for t in range(1, n):
        e1 = np.random.normal(0, sigma1)
        X[t] = a * X[t-1] + b + e1
    trajectories.append(X)

# Plotting
plt.figure(figsize=(10, 6))
for i, traj in enumerate(trajectories):
    plt.plot(traj, label=f"Trajectory {i+1}")
plt.title("5 Independent Realizations of the State Process $X_t$")
plt.xlabel("Time step")
plt.ylabel("$X_t$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
