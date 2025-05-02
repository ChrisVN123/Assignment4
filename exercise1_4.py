import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Load data
data = pd.read_csv("transformer_data.csv")
observations = data["Y"].values

plt.plot(observations)
# plt.show()

def kalman_filter(y, theta):
    a, b, Q = theta
    n = len(y)

    # Constants
    R = 1          # Observation noise variance
    x0 = 5         # Initial state estimate
    P0 = 1         # Initial covariance estimate

    # Allocate arrays
    x_pred = np.zeros(n)      # Predicted state estimates
    P_pred = np.zeros(n)      # Predicted covariances
    x_filt = np.zeros(n)      # Filtered state estimates
    P_filt = np.zeros(n)      # Filtered covariances
    innovations = np.zeros(n) # Measurement residuals
    S_t = np.zeros(n)         # Innovation variances
    log_likelihoods = np.zeros(n)  # Log-likelihoods

    # Initialization
    x_filt[0] = x0
    P_filt[0] = P0
    S_t[0] = P_filt[0] + R
    kalman_gain = P_filt[0] / S_t[0]
    innovations[0] = y[0] - x_filt[0]
    log_likelihoods[0] = -0.5 * (np.log(2 * np.pi) + np.log(S_t[0]) + (innovations[0]**2) / S_t[0])

    for t in range(1, n):
        # Predict
        x_pred[t] = a * x_filt[t-1] + b
        P_pred[t] = a**2 * P_filt[t-1] + Q

        # Innovation
        innovations[t] = y[t] - x_pred[t]
        S_t[t] = P_pred[t] + R

        # Kalman Gain
        kalman_gain = P_pred[t] / S_t[t]

        # Update
        x_filt[t] = x_pred[t] + kalman_gain * innovations[t]
        P_filt[t] = (1 - kalman_gain) * P_pred[t]

        # Log-likelihood
        log_likelihoods[t] = -0.5 * (np.log(2 * np.pi) + np.log(S_t[t]) + (innovations[t]**2) / S_t[t])

    return log_likelihoods, x_filt, innovations

# Example usage
theta = [0.5, 0.5, 0.5]  # a, b, Q
logL, filtered_states, residuals = kalman_filter(observations, theta)
print("Log-likelihood sum:", np.sum(logL))

def negative_log_likelihood(theta, y):
    log_likelihoods, _, _ = kalman_filter(y, theta)
    return -np.sum(log_likelihoods)

def optimal_kalman(initial_guess, train_data, plotting=False):
    bounds = [(1e-5, None)] * 3  # a, b, Q (all should be positive)

    result = minimize(negative_log_likelihood, initial_guess, args=(train_data,), bounds=bounds)
    theta_hat = result.x

    print("Estimated parameters:")
    print(f"a (mean reversion): {theta_hat[0]:.4f}")
    print(f"b (long-term mean): {theta_hat[1]:.4f}")
    print(f"Q (state noise var): {theta_hat[2]:.4f}")

    # Optionally plot filtered results
    if plotting:
        _, filtered_states, _ = kalman_filter(train_data, theta_hat)
        plt.plot(train_data, label='Observed')
        plt.plot(filtered_states, label='Filtered')
        plt.legend()
        plt.title("Kalman Filter Fit")
        plt.show()

    return theta_hat


optimal_kalman(theta, observations, plotting=True)
