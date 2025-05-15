import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from exercise1_2 import simulated


def kalman_filter(y, theta):
    a, b, Q = theta
    n = len(y)

    R = 1          # Observation noise variance
    x0 = 5         # Initial state estimate
    P0 = 1         # Initial covariance estimate

    x_pred = np.zeros(n)
    P_pred = np.zeros(n)
    x_filt = np.zeros(n)
    P_filt = np.zeros(n)
    innovations = np.zeros(n)
    S_t = np.zeros(n)
    log_likelihoods = np.zeros(n)

    x_filt[0] = x0
    P_filt[0] = P0
    S_t[0] = P_filt[0] + R
    kalman_gain = P_filt[0] / S_t[0]
    innovations[0] = y[0] - x_filt[0]
    log_likelihoods[0] = -0.5 * (np.log(2 * np.pi) + np.log(S_t[0]) + (innovations[0]**2) / S_t[0])

    for t in range(1, n):
        x_pred[t] = a * x_filt[t-1] + b
        P_pred[t] = a**2 * P_filt[t-1] + Q

        innovations[t] = y[t] - x_pred[t]
        S_t[t] = P_pred[t] + R
        kalman_gain = P_pred[t] / S_t[t]

        x_filt[t] = x_pred[t] + kalman_gain * innovations[t]
        P_filt[t] = (1 - kalman_gain) * P_pred[t]

        log_likelihoods[t] = -0.5 * (np.log(2 * np.pi) + np.log(S_t[t]) + (innovations[t]**2) / S_t[t])

    return log_likelihoods, x_filt, innovations

def negative_log_likelihood(theta, y):
    log_likelihoods, _, _ = kalman_filter(y, theta)
    return -np.sum(log_likelihoods)

def optimal_kalman(initial_guess, train_data, plotting=False):
    bounds = [(1e-5, None)] * 3  # a, b, Q

    result = minimize(negative_log_likelihood, initial_guess, args=(train_data,), bounds=bounds)
    theta_hat = result.x

    if plotting:
        _, filtered_states, _ = kalman_filter(train_data, theta_hat)
        plt.plot(train_data, label='Observed')
        plt.plot(filtered_states, label='Filtered')
        plt.legend()
        plt.title("Kalman Filter Fit")
        plt.show()

    return theta_hat


# Allocate result storage: rows = iterations, cols = [a, b, Q]
theta_results = np.zeros((100, 3))

# Loop over simulations
for i in range(100):
    np.random.seed(i + 2025)  # Slightly different seed each time
    X, Y = simulated(a=1, b=0.9, sigma1=5)
    initial_theta = [0.5, 0.5, 0.5]
    if i == 1:
        optimal_kalman(initial_theta, X, plotting=True)
    theta_est = optimal_kalman(initial_theta, X, plotting=False)
    theta_results[i, :] = theta_est

# Show results
# Boxplot
plt.boxplot([theta_results[:, 0], theta_results[:, 1], theta_results[:, 2]])
plt.xticks([1, 2, 3], ['a (mean reversion)', 'b (long-term mean)', 'Q (state noise)'])
plt.ylabel('Estimated Value')
plt.title('Distribution of Estimated Kalman Parameters over 100 Simulations')
plt.grid(True)
plt.show()
