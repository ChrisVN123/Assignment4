import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data
df = pd.read_csv("transformer_data.csv")
matrix = df.to_numpy()
u = matrix[:, 2:]  # Use last three columns as exogenous inputs
Y = df["Y"].values

# Optional: Normalize input and output for better optimization
# Y = (Y - np.mean(Y)) / np.std(Y)
# u = (u - np.mean(u, axis=0)) / np.std(u, axis=0)

# Updated parameter unpacking with observation bias `e`
def unpack_parameters(theta):
    A = theta[0]
    B = np.array(theta[1:4])  # B1, B2, B3
    C = theta[4]
    Q = theta[5]  # Process noise variance
    R = theta[6]  # Measurement noise variance
    x0 = theta[7]  # Initial latent state
    e = theta[8]   # Observation bias
    return A, B, C, Q, R, x0, e

# Kalman filter implementation
def kalman_filter(y, u, theta):
    A, B, C, Q, R, x0, e = unpack_parameters(theta)
    n = len(y)

    x_pred = np.zeros(n)
    P_pred = np.zeros(n)
    x_filt = np.zeros(n)
    P_filt = np.zeros(n)
    log_likelihoods = np.zeros(n)

    # Initialization
    x_filt[0] = x0
    P_filt[0] = 1.0  # Could be estimated too

    for t in range(1, n):
        # Predict
        x_pred[t] = A * x_filt[t-1] + np.dot(B, u[t])
        P_pred[t] = A * P_filt[t-1] * A + Q

        # Update
        y_pred = C * x_pred[t] + e
        innovation = y[t] - y_pred
        S_t = C * P_pred[t] * C + R
        K_t = P_pred[t] * C / S_t

        x_filt[t] = x_pred[t] + K_t * innovation
        P_filt[t] = (1 - K_t * C) * P_pred[t]

        # Log-likelihood contribution
        log_likelihoods[t] = -0.5 * (np.log(2 * np.pi) + np.log(S_t) + innovation**2 / S_t)

    return log_likelihoods, x_filt

# Negative log-likelihood for optimization
def negative_log_likelihood(theta, y, u):
    log_likelihoods, _ = kalman_filter(y, u, theta)
    return -np.sum(log_likelihoods)

# Model estimation
def estimate_model(y, u):
    # Parameters: A, B1, B2, B3, C, Q, R, x0, e
    initial_guess = np.array([0.9, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 20.0, 0.0])
    lower_bounds = [-1.5] * 5 + [1e-6, 1e-6, -np.inf, -np.inf]
    upper_bounds = [1.5] * 5 + [np.inf, np.inf, np.inf, np.inf]

    result = minimize(negative_log_likelihood, initial_guess,
                      args=(y, u), bounds=list(zip(lower_bounds, upper_bounds)))

    return result

# Estimate parameters
result = estimate_model(Y, u)
print("Estimated parameters:", result.x)

# Run filter with estimated parameters
log_lik, x_filtered = kalman_filter(Y, u, result.x)

C = result.x[4]
e = result.x[8]
Y_pred = C * x_filtered + e

plt.plot(Y, label="Observed")
plt.plot(Y_pred, label="Filtered prediction ($C \\cdot x + e$)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Kalman Filter: Observed vs Filtered Prediction")
plt.grid(True)
plt.show()

