import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv("transformer_data.csv")
matrix = df.to_numpy()
u = matrix[:,2:] #we only want the last three columns


print(u)


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